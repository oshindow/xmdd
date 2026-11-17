"""The X-MDD model: Qwen3-ASR with the fine-tuned X-MDD weights, plus phone decoding."""
from __future__ import annotations

import re
from time import perf_counter
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

from xmdd.constrain import make_phone_constraint
from xmdd.prompt import PromptBuilder
from xmdd.latency import ModuleTimer


class XMDD(torch.nn.Module):
    def __init__(self, cfg, tokenizer_size, asr_config=None):
        super().__init__()
        mc = cfg["model"]
        dtype = torch.bfloat16 if mc["dtype"] == "bfloat16" else torch.float16
        if asr_config is None:
            asr_config = Qwen3ASRConfig.from_json_file(Path(__file__).with_name("asr_config.json"))
        self.asr = Qwen3ASRForConditionalGeneration._from_config(
            asr_config, dtype=dtype, attn_implementation=mc["attention_implementation"])
        self.asr.config.use_cache = bool(mc.get("use_cache", False))
        self.asr.config.text_config.use_cache = bool(mc.get("use_cache", False))
        if tokenizer_size > self.asr.get_input_embeddings().num_embeddings:  # added prompt-role tokens
            self.asr.resize_token_embeddings(tokenizer_size)
        self._chunk_audio_encoding(32)

    def _chunk_audio_encoding(self, chunk):
        """Encode the audios of many-shot prompts in groups of `chunk` (the encoder treats each audio
        independently, so this equals encoding them all at once, with far less memory)."""
        audio_model = self.asr.model
        whole = audio_model.get_audio_features

        def chunked(input_features, input_features_mask, **kwargs):
            if input_features.shape[0] <= chunk:
                return whole(input_features, input_features_mask, **kwargs)
            parts = [whole(input_features[i:i + chunk], input_features_mask[i:i + chunk], **kwargs)
                     for i in range(0, input_features.shape[0], chunk)]
            out = parts[0]
            out.last_hidden_state = torch.cat([p.last_hidden_state for p in parts])
            out.pooler_output = torch.cat([p.pooler_output for p in parts])
            return out
        audio_model.get_audio_features = chunked

    @classmethod
    def from_pretrained(cls, model_dir, device="cuda", attn_implementation=None):
        """Load a released model folder (config.yaml, model.safetensors, processor/, phone_vocabulary.txt)."""
        model_dir = Path(model_dir)
        cfg = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
        if attn_implementation:
            cfg["model"]["attention_implementation"] = attn_implementation
        processor = AutoProcessor.from_pretrained(model_dir / cfg["data"].get("processor_path", "processor"),
                                                  local_files_only=True)
        # Older releases omit config.json; use the bundled backbone architecture.
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            config_path = Path(__file__).with_name("asr_config.json")
        model = cls(cfg, len(processor.tokenizer), Qwen3ASRConfig.from_json_file(config_path))
        weights = load_file(str(model_dir / "model.safetensors"))
        # ctc_head.*: an unused CTC head inherited from an earlier training stage
        model.load_state_dict({k: v for k, v in weights.items() if not k.startswith("ctc_head.")}, strict=True)
        model.eval().to(device)
        model.cfg, model.processor, model.device = cfg, processor, torch.device(device)
        model.prompt = PromptBuilder(processor, cfg)
        vocabulary = model_dir / cfg["data"].get("phone_vocabulary_path", "phone_vocabulary.txt")
        model.vocabulary = set(vocabulary.read_text(encoding="utf-8").splitlines())
        model.constraint = None
        return model

    def set_phone_constraint(self, symbols_or_path):
        """Constrain decoding to a phone set (list of symbols, .txt with one per line, or a manifest)."""
        self.constraint = make_phone_constraint(self.processor.tokenizer, symbols_or_path)

    @torch.inference_mode()
    def predict(self, query, demos=(), raw_phones=False, return_latency=False):
        """Predicted actual phones of `query` (an utterance from xmdd.load_manifest) given `demos`.

        raw_phones: keep every output symbol (unseen languages); otherwise keep only phones of the
        training vocabulary.
        """
        def sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        sync()
        started = perf_counter()
        inputs = self.prompt(query, demos)
        prepared = perf_counter()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        sync()
        transferred = perf_counter()
        prompt_length = inputs["input_ids"].shape[1]
        extra = {"prefix_allowed_tokens_fn": self.constraint(prompt_length)} if self.constraint else {}
        timers = {name: ModuleTimer(module, self.device) for name, module in (
            ("aut", self.asr.model.audio_tower),
            ("projector", self.asr.model.multi_modal_projector),
            ("llm", self.asr.model.language_model))} if return_latency else {}
        generation_started = perf_counter()
        try:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                generated = self.asr.generate(**inputs, max_new_tokens=self.cfg["evaluation"]["max_new_tokens"],
                                              do_sample=False, use_cache=True, **extra)
            sync()
        finally:
            for timer in timers.values():
                timer.close()
        generation_ended = perf_counter()
        text = self.processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
        prefix = self.cfg["data"]["target_prefix_by_language"].get(query["language_id"],
                                                                   f"language {query['language_name']}<asr_text>")
        if text.startswith(prefix):
            text = text[len(prefix):]
        phones = (re.sub(r"^\s*language [A-Za-z]+<asr_text>", "", text).split() if raw_phones
                  else [p for p in text.strip().split() if p in self.vocabulary])
        if not return_latency:
            return phones
        llm = timers["llm"].seconds()
        latency = {
            "num_demos": len(demos), "num_audios": int(inputs["input_features"].shape[0]),
            "prompt_tokens": int(prompt_length),
            "generated_tokens": int(generated.shape[1] - prompt_length),
            "preprocess_s": prepared - started, "transfer_s": transferred - prepared,
            "aut_s": sum(timers["aut"].seconds()),
            "projector_s": sum(timers["projector"].seconds()),
            "llm_s": sum(llm), "llm_prefill_s": llm[0] if llm else 0.0,
            "llm_decode_s": sum(llm[1:]),
            "generation_s": generation_ended - generation_started,
            "postprocess_s": perf_counter() - generation_ended,
            "total_s": perf_counter() - started,
        }
        return phones, latency
