# X-MDD

**X-MDD: LLM-Based Mispronunciation Detection and Diagnosis for Unseen Languages with Canonical References**
Xintong Wang, Junchuan Zhao, Ye Wang (National University of Singapore)
[[Paper]](https://arxiv.org/abs/2610.23079) [[Model]](https://huggingface.co/walston/X-MDD)

Mispronunciation detection and diagnosis (MDD) gives language learners phone-level feedback. Extending it
to a new language usually means collecting and annotating learner speech for that language. X-MDD asks
whether MDD skills learned from annotated learner speech can carry over to languages never seen in MDD
training, as long as canonical pronunciation references exist.

X-MDD is built on Qwen3-ASR-1.7B. For each utterance it receives:

- **canonical speech**: the target text synthesized by TTS, cloning the learner's voice;
- **canonical phones**: the expected phone sequence;
- **learner speech**: the learner's recording;
- optionally, **in-context demonstrations**: the same three inputs plus the annotated actual phones.

It generates the phone sequence the learner actually produced. Aligning that sequence to the canonical
phones gives detection (which phones are wrong) and diagnosis (what was said instead). Training uses
supervised fine-tuning followed by GRPO, with a reward that combines detection F1 and phone error rate.

## Pretrained model

| Model | Backbone | Inputs | Training | Link |
|---|---|---|---|---|
| X-MDD | Qwen3-ASR-1.7B | canonical speech + canonical phones + learner speech | SFT on Mandarin + English → GRPO | [walston/X-MDD](https://huggingface.co/walston/X-MDD) |

```bash
hf download walston/X-MDD --local-dir pretrained/X-MDD
```

Inference loads weights and the processor from `pretrained/X-MDD` without downloading
Qwen3-ASR backbone weights. If `config.json` is absent, it uses the bundled architecture config.

## Installation

Linux x86_64; uses a prebuilt flash-attn wheel.

```bash
conda create -n xmdd python=3.11 -y
conda activate xmdd
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install -r requirements.txt --no-build-isolation
```

## Quick start: Arabic example

`examples/arabic/` holds a small example from the [QuranMB.v2 test set](https://huggingface.co/datasets/IqraEval/QuranMB.v2):

- `queries.jsonl`: 3 query utterances;
- `demos.jsonl`: 4 demonstrations from the same speaker;
- `wav/`: the learner speech and canonical speech (TTS) of each utterance;
- `phone_symbols.txt`: the Arabic phone set for optional constrained decoding.

The canonical
speech was synthesized with [SILMA TTS](https://huggingface.co/silma-ai/silma-tts). See
`examples/arabic/LICENSE.md`.

```bash
python -m xmdd.infer --language ar \
    --manifest examples/arabic/queries.jsonl --support-manifest examples/arabic/demos.jsonl \
    --shots 0 4 --out-dir results/ar_example
```

## Chinese example

`examples/chinese/` contains 3 queries and 4 demos from the same [LATIC](https://doi.org/10.21227/mtqj-qh10) test speaker,
with learner speech, CosyVoice3 canonical speech, and tonal-phone annotations.
All 3 queries and 4 demos have a **Korean accent** (unseen during training).
The example data is shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
see `examples/chinese/README.md` for attribution.

```bash
python -m xmdd.infer --language zh \
    --manifest examples/chinese/queries.jsonl --support-manifest examples/chinese/demos.jsonl \
    --shots 0 4 --out-dir results/zh_example
```

    
## Acknowledgements

This project builds on the following open-source work:

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B): model backbone (audio encoder, projector and LLM decoder), loaded through [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) and [SILMA TTS](https://huggingface.co/silma-ai/silma-tts): canonical speech synthesis
- GRPO, following [DeepSeekMath (Shao et al., 2024)](https://arxiv.org/abs/2402.03300), and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for training

Please also follow the licenses and terms of use of these projects, models and datasets.

## Citation

```bibtex
@article{wang2026xmdd,
  title   = {X-MDD: LLM-Based Mispronunciation Detection and Diagnosis for Unseen Languages with Canonical References},
  author  = {Wang, Xintong and Zhao, Junchuan and Wang, Ye},
  journal = {arXiv preprint arXiv:2610.23079},
  year    = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The model weights are derived from Qwen3-ASR-1.7B
and are subject to its license.
