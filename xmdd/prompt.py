"""Prompt construction: task block, demonstrations and query in one Qwen3-ASR chat turn.

Each pair (demonstration or query) lists its fields in config `data.prompt_fields` order, e.g.
<|canonical_audio|>[audio]<|canonical_phone|>p1 p2 ...<|learner_audio|>[audio]; a demonstration
adds <|actual_phone|> and its annotated phones and is wrapped in <|demo_start|> ... <|demo_end|>.
"""
from __future__ import annotations

import soundfile as sf
import torch
import torchaudio

ROLE_TOKENS = {"canonical_phone": "<|canonical_phone|>", "canonical_speech": "<|canonical_audio|>",
               "learner_speech": "<|learner_audio|>"}
AUDIO_SOURCE = {"canonical_speech": "canonical_audio", "learner_speech": "audio"}
DEMO_START, ACTUAL_PHONE, DEMO_END = "<|demo_start|>", "<|actual_phone|>", "<|demo_end|>"
TASK_START, TASK_END = "<|task_start|>", "<|task_end|>"
USER_START = "<|im_start|>user\n"


def expand_audio_placeholders(text, audio_token, lengths):
    """Expand the i-th audio token in `text` to lengths[i] tokens."""
    parts = text.split(audio_token)
    if len(parts) - 1 != len(lengths):
        raise ValueError(f"{len(parts) - 1} audio placeholders but {len(lengths)} audios")
    return parts[0] + "".join(audio_token * int(n) + tail for n, tail in zip(lengths, parts[1:]))


class PromptBuilder:
    def __init__(self, processor, cfg):
        dc = cfg["data"]
        self.processor = processor
        self.sample_rate = int(dc["sample_rate"])
        self.fields = list(dc.get("prompt_fields") or ["canonical_speech", "canonical_phone", "learner_speech"])
        unknown = set(self.fields) - set(ROLE_TOKENS)
        if unknown or "learner_speech" not in self.fields:
            raise ValueError(f"prompt_fields must include learner_speech and only use {sorted(ROLE_TOKENS)}")
        self.speech_fields = [f for f in self.fields if f.endswith("_speech")]
        self.chat_shell = ("<|im_start|>system\n" + dc.get("system_prompt", "") + "<|im_end|>\n"
                           + USER_START + "__PAIRED_PROMPT__<|im_end|>\n<|im_start|>assistant\n")
        self.placeholder = processor.audio_bos_token + processor.audio_token + processor.audio_eos_token

    def load_audio(self, path):
        # soundfile reads the same samples as torchaudio.load, ~20x faster
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data).mean(1)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.numpy()

    def pair_body(self, row, include_answer):
        body = "".join(ROLE_TOKENS[f] + (" ".join(row["canonical_phones"]) if f == "canonical_phone"
                                         else self.placeholder) for f in self.fields)
        if include_answer:
            body += ACTUAL_PHONE + " ".join(row["actual_phones"])
        return body

    @staticmethod
    def task_text(query):
        return (TASK_START + f"language: {query['language_name']}\naccent: {query['accent_id']}\n"
                + f"task: {query['task_description']}\n" + TASK_END)

    def __call__(self, query, demos=()):
        """Model inputs (input_ids, attention_mask, input_features, input_features_mask) for one query."""
        chunks, audios = [], []
        for row, include_answer in [(d, True) for d in demos] + [(query, False)]:
            body = self.pair_body(row, include_answer)
            chunks.append(DEMO_START + body + DEMO_END if include_answer else body)
            audios.extend(self.load_audio(row[AUDIO_SOURCE[f]]) for f in self.speech_fields)
        features = self.processor.feature_extractor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
            return_attention_mask=True)
        feature_mask = features.pop("attention_mask")
        lengths = self.processor._get_audio_token_length(feature_mask.sum(-1)).tolist()
        prompt = expand_audio_placeholders(self.chat_shell.replace("__PAIRED_PROMPT__", "".join(chunks)),
                                           self.processor.audio_token, lengths)
        prompt = prompt.replace(USER_START, USER_START + self.task_text(query), 1)
        tokens = self.processor.tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
                "input_features": features["input_features"], "input_features_mask": feature_mask}
