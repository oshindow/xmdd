#!/usr/bin/env python3
"""4-GPU Whisper+OTC MDD baseline. Launch directly; Trainer creates DDP workers."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

import k2, torch, torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
import whisper

from mdd.metrics import MddMetrics
from mdd.paired_data import audio_to_mel
from otc_graph_compiler import OtcTrainingGraphCompiler


class BaselineDataset(Dataset):
    def __init__(self, path, n_mels=80):
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
        self.n_mels = n_mels
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; mel, length = audio_to_mel(row["l2_wav"], self.n_mels)
        return row, mel, length


def collate(batch):
    rows, mels, lengths = zip(*batch)
    return list(rows), torch.stack(mels), torch.tensor(lengths, dtype=torch.long)


class Baseline(pl.LightningModule):
    def __init__(self, args):
        super().__init__(); self.save_hyperparameters(vars(args)); self.args = args
        self.symbols = k2.SymbolTable.from_file(args.tokens)
        self.id_to_phone = {i: self.symbols._id2sym[i] for i in range(len(self.symbols))}
        self.model = whisper.load_model(args.model, ctc_vocab=len(self.symbols), ctc_layers=args.ctc_layers)
        self.graph = None; self.validation_outputs = []

    def on_fit_start(self):
        self.graph = OtcTrainingGraphCompiler(
            Path(self.args.tokens).parent, "<star>", self.device,
            self.args.bypass_weight, self.args.self_loop_weight,
            self.args.bypass_decay, self.args.self_loop_decay)

    def encode(self, mel):
        features, _ = self.model.encoder(mel)
        _, logits = self.model.ctc_head(features)
        return features, logits

    def otc_loss(self, logits, rows, lengths):
        batch, _, vocab = logits.shape
        log_probs = logits.log_softmax(-1)
        star = torch.logsumexp(log_probs[:, :, 1:], -1, keepdim=True) - math.log(vocab - 1)
        dense_scores = torch.cat((log_probs, star), -1)
        targets = [[p for p in row["actual_phones"] if p] for row in rows]
        ids = [[[self.symbols[p]] for p in sequence] for sequence in targets]
        graph = self.graph.compile(targets, True, True,
            self.args.bypass_weight * self.args.bypass_decay ** max(0, self.current_epoch),
            self.args.self_loop_weight * self.args.self_loop_decay ** max(0, self.current_epoch), ids).to(self.device)
        encoder_lengths = torch.clamp((lengths + 1) // 2, max=logits.size(1)).to(torch.int32)
        supervision = torch.stack((torch.arange(batch, dtype=torch.int32), torch.zeros(batch, dtype=torch.int32),
                                   encoder_lengths.cpu()), 1)
        dense = k2.DenseFsaVec(dense_scores, supervision, allow_truncate=3)
        target_lengths = torch.tensor([len(x) for x in targets], dtype=torch.int32, device=self.device)
        return k2.ctc_loss(graph, dense, target_lengths, output_beam=self.args.output_beam,
                           reduction="mean", use_double_scores=True)

    def training_step(self, batch, _):
        rows, mel, lengths = batch; _, logits = self.encode(mel)
        loss = self.otc_loss(logits, rows, lengths)
        self.log("train/otc_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    @staticmethod
    def greedy(logits, lengths, id_to_phone):
        output = []
        for sequence, length in zip(logits.argmax(-1), lengths):
            result, previous = [], -1
            for token in sequence[:int(length)]:
                token = int(token)
                if token != 0 and token != previous: result.append(id_to_phone[token])
                previous = token
            output.append(result)
        return output

    def validation_step(self, batch, _):
        rows, mel, lengths = batch; features, logits = self.encode(mel)
        enc_lengths = torch.clamp((lengths + 1) // 2, max=logits.size(1))
        hypotheses = self.greedy(logits, enc_lengths, self.id_to_phone)
        for b, (row, hypothesis) in enumerate(zip(rows, hypotheses)):
            self.validation_outputs.append((row, hypothesis))

    def on_validation_epoch_end(self):
        gathered = [None] * dist.get_world_size() if dist.is_initialized() else [self.validation_outputs]
        if dist.is_initialized(): dist.all_gather_object(gathered, self.validation_outputs)
        if self.global_rank == 0:
            metrics = {
                "overall": MddMetrics(),
                "vowels": MddMetrics(lambda phone: phone[-1:] in "12345"),
                "consonants": MddMetrics(lambda phone: phone[-1:] not in "12345"),
            }
            for shard in gathered:
                for row, hypothesis in shard:
                    for metric in metrics.values():
                        metric.update(row["canonical_phones"], row["actual_phones"], row["error_labels"], hypothesis)
            values = {group: metric.compute() for group, metric in metrics.items()}
            for group, group_values in values.items():
                for key, value in group_values.items():
                    self.log(f"val/{group}/{key}", float(value),
                             prog_bar=group == "overall" and key in {"per", "f1"}, rank_zero_only=True)
            print("VALIDATION", json.dumps(values))
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * self.args.warmup_ratio), steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/mdd_paired_manifest/train.jsonl")
    p.add_argument("--val", default="data/mdd_paired_manifest/test_seen.jsonl")
    p.add_argument("--tokens", default="data/mdd_phone/tokens.txt")
    p.add_argument("--exp-dir", default="exp/mdd_otc_baseline")
    p.add_argument("--model", default="small"); p.add_argument("--ctc-layers", type=int, default=2)
    p.add_argument("--devices", type=int, default=4); p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--accumulate", type=int, default=1); p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=.1)
    p.add_argument("--warmup-ratio", type=float, default=.05); p.add_argument("--output-beam", type=float, default=10)
    p.add_argument("--bypass-weight", type=float, default=-19); p.add_argument("--self-loop-weight", type=float, default=3.75)
    p.add_argument("--bypass-decay", type=float, default=.975); p.add_argument("--self-loop-decay", type=float, default=.999)
    args = p.parse_args(); pl.seed_everything(2025, workers=True); Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    model = Baseline(args)
    train = DataLoader(BaselineDataset(args.train), args.batch_size, shuffle=True, drop_last=True,
                       num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    val = DataLoader(BaselineDataset(args.val), args.batch_size, num_workers=args.workers, collate_fn=collate)
    callbacks = [ModelCheckpoint(dirpath=f"{args.exp_dir}/checkpoints", monitor="val/f1", mode="max",
                                 save_last=True, save_top_k=3, filename="epoch{epoch:02d}-f1{val/f1:.4f}", auto_insert_metric_name=False),
                 LearningRateMonitor("step")]
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy="ddp", max_epochs=args.epochs,
                         precision=args.precision, accumulate_grad_batches=args.accumulate, gradient_clip_val=1.0,
                         callbacks=callbacks, logger=TensorBoardLogger(args.exp_dir, name="logs"))
    trainer.fit(model, train, val)

if __name__ == "__main__": main()
