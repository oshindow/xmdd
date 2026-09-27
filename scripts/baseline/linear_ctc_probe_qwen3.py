#!/usr/bin/env python3
"""Linear CTC probing of one frozen Qwen3-ASR audio-encoder layer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from scripts.baseline.finetuning_pinyin_ctc_qwen3 import (
    PhoneTable,
    QwenCollator,
    QwenCTCDataset,
    load_audio_encoder,
)


class ProbeCheckpoint(Callback):
    """Save only the small linear probe, never a copy of the frozen encoder."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.best_per = float("inf")
        self.history = {}

    @staticmethod
    def _payload(module, per):
        return {
            "layer": module.layer,
            "num_encoder_layers": len(module.encoder.layers),
            "d_model": module.encoder.config.d_model,
            "ctc_vocab": module.args.ctc_vocab,
            "model_name": module.args.model_name,
            "val_per": float(per),
            "global_step": module.global_step,
            "ctc_head": {k: v.detach().cpu() for k, v in module.ctc_head.state_dict().items()},
        }

    def on_validation_end(self, trainer, module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val/per")
        if value is None:
            return
        per = float(value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history[str(module.global_step)] = per
        (self.output_dir / "metrics_history.json").write_text(
            json.dumps({"layer": module.layer, "per_by_step": self.history}, indent=2) + "\n",
            encoding="utf-8")
        if per < self.best_per:
            self.best_per = per
            torch.save(self._payload(module, per), self.output_dir / "probe_best.pt")
            (self.output_dir / "best_metrics.json").write_text(
                json.dumps({"layer": module.layer, "step": module.global_step,
                            "val_per": per}, indent=2) + "\n", encoding="utf-8")

    def on_train_end(self, trainer, module):
        if trainer.is_global_zero:
            value = trainer.callback_metrics.get("val/per", float("nan"))
            torch.save(self._payload(module, float(value)), self.output_dir / "probe_last.pt")


class FrozenLayerCTCProbe(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.layer = args.layer
        self.phones = PhoneTable(args.token_table_path)
        if len(self.phones) != args.ctc_vocab:
            raise ValueError(f"Token table has {len(self.phones)} symbols, expected {args.ctc_vocab}")

        self.encoder = load_audio_encoder(args.model_name)
        count = len(self.encoder.layers)
        if not 1 <= self.layer <= count:
            raise ValueError(f"--layer must be in [1, {count}], got {self.layer}")
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.ctc_head = nn.Linear(self.encoder.config.d_model, args.ctc_vocab)
        nn.init.normal_(self.ctc_head.weight, 0.0, self.encoder.config.initializer_range)
        nn.init.zeros_(self.ctc_head.bias)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self._selected_output = None
        self._hook = self.encoder.layers[self.layer - 1].register_forward_hook(self._capture_layer)
        self.register_buffer("val_errors", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("val_phones", torch.tensor(0, dtype=torch.long), persistent=False)
        print(f"Linear probe: block {self.layer}/{count}; encoder frozen; trainable parameters="
              f"{sum(p.numel() for p in self.ctc_head.parameters())}")

    def train(self, mode=True):
        super().train(mode)
        # Lightning recursively switches children to train mode; keep the backbone deterministic.
        self.encoder.eval()
        return self

    def _capture_layer(self, _module, _inputs, output):
        self._selected_output = output[0] if isinstance(output, tuple) else output

    def encode(self, features, feature_mask):
        lengths = feature_mask.sum(-1).long()
        self._selected_output = None
        with torch.no_grad():
            self.encoder(input_features=features, input_features_mask=feature_mask,
                         return_dict=True)
        if self._selected_output is None:
            raise RuntimeError(f"Encoder layer {self.layer} hook did not run")
        packed = self._selected_output.detach()
        output_lengths = self.encoder._post_cnn_length(
            lengths.remainder(self.encoder.n_window * 2))
        output_lengths += lengths.div(self.encoder.n_window * 2, rounding_mode="floor") * 13
        hidden = pad_sequence(packed.split(output_lengths.tolist()), batch_first=True)
        return self.ctc_head(hidden), output_lengths

    def targets(self, rows):
        sequences = []
        for row in rows:
            phones = [p for p in row["actual_phones"] if p not in (None, "sil")]
            unknown = sorted({p for p in phones if p not in self.phones.phone_to_id})
            if unknown:
                raise KeyError(f"Phones absent from token table: {unknown}")
            sequences.append([self.phones.phone_to_id[p] for p in phones])
        flat = torch.tensor([x for seq in sequences for x in seq], device=self.device,
                            dtype=torch.long)
        lengths = torch.tensor([len(seq) for seq in sequences], device=self.device,
                               dtype=torch.long)
        return flat, lengths, sequences

    def training_step(self, batch, _batch_idx):
        rows, features, mask = batch
        logits, output_lengths = self.encode(features, mask)
        targets, target_lengths, _ = self.targets(rows)
        loss = self.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                             output_lengths, target_lengths)
        self.log("train/ctc_loss", loss, on_step=True, prog_bar=True)
        return loss

    @staticmethod
    def edit_distance(reference, hypothesis):
        previous = list(range(len(hypothesis) + 1))
        for i, ref in enumerate(reference, 1):
            current = [i]
            for j, hyp in enumerate(hypothesis, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (ref != hyp)))
            previous = current
        return previous[-1]

    def on_validation_epoch_start(self):
        self.val_errors.zero_()
        self.val_phones.zero_()

    def validation_step(self, batch, _batch_idx):
        rows, features, mask = batch
        logits, output_lengths = self.encode(features, mask)
        _, _, references = self.targets(rows)
        for prediction, length, reference in zip(logits.argmax(-1), output_lengths, references):
            hypothesis, previous = [], -1
            for token in prediction[:int(length)]:
                token = int(token)
                if token != 0 and token != previous:
                    hypothesis.append(token)
                previous = token
            self.val_errors += self.edit_distance(reference, hypothesis)
            self.val_phones += len(reference)

    def on_validation_epoch_end(self):
        totals = torch.stack((self.val_errors, self.val_phones))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        self.log("val/per", totals[0].float() / totals[1].clamp_min(1), prog_bar=True)

    def configure_optimizers(self):
        # This assertion makes accidental backbone training fail loudly.
        trainable = [p for p in self.parameters() if p.requires_grad]
        if {id(p) for p in trainable} != {id(p) for p in self.ctc_head.parameters()}:
            raise RuntimeError("Trainable parameters must be exactly ctc_head.weight and ctc_head.bias")
        optimizer = AdamW(self.ctc_head.parameters(), lr=self.args.learning_rate,
                          betas=(0.9, 0.98), eps=self.args.adam_epsilon, weight_decay=0.0)

        def schedule(step):
            if step < self.args.warmup_steps:
                return step / max(1, self.args.warmup_steps)
            return max(0.0, (self.args.max_steps - step) /
                       max(1, self.args.max_steps - self.args.warmup_steps))
        return [optimizer], [{"scheduler": LambdaLR(optimizer, schedule), "interval": "step"}]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layer", type=int, required=True,
                        help="1-based Transformer block whose output is probed")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--token-table-path", default="data/lang_jsonl/tokens.txt")
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--ctc-vocab", type=int, default=188)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    output_dir = Path(args.exp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n",
                                           encoding="utf-8")
    collator = QwenCollator(args.model_name)
    train_set = QwenCTCDataset(args.train_path, args.data_root)
    val_set = QwenCTCDataset(args.val_path, args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers, collate_fn=collator,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collator,
                            persistent_workers=args.workers > 0)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        strategy="auto", precision=args.precision, max_steps=args.max_steps, max_epochs=-1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0, val_check_interval=args.eval_steps * args.gradient_accumulation_steps,
        check_val_every_n_epoch=None, callbacks=[ProbeCheckpoint(args.exp_dir)],
        logger=TensorBoardLogger(args.exp_dir, name="logs"),
    )
    trainer.fit(FrozenLayerCTCProbe(args), train_loader, val_loader)


if __name__ == "__main__":
    main()
