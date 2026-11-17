# Example data license

The learner recordings (`wav/*_learner.wav`), canonical texts and phone annotations in this folder are
7 utterances of speaker `quranmb_00015` from
[IqraEval/QuranMB.v2](https://huggingface.co/datasets/IqraEval/QuranMB.v2), the test set of the Iqra'Eval
shared task. They are redistributed unchanged, apart from the manifest format, under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
**Non-commercial use only.**

The canonical speech (`wav/*_canonical_tts.wav`) was synthesized from the canonical texts with
[SILMA TTS](https://huggingface.co/silma-ai/silma-tts) (Apache-2.0), using each learner recording as the
voice-cloning reference. It is a derivative of the data above and is distributed under the same
CC BY-NC 4.0 license.

If you use this data, please cite the Iqra'Eval shared task:

```bibtex
@inproceedings{iqraeval2025,
  title     = {Iqra'Eval: A Shared Task on Qur'anic Pronunciation Assessment},
  booktitle = {Proceedings of the Third Arabic Natural Language Processing Conference: Shared Tasks},
  year      = {2025},
  url       = {https://aclanthology.org/2025.arabicnlp-sharedtasks.61}
}
```

The rest of this repository is released under the MIT License (see `LICENSE` at the repository root).
