# Chinese example data

Three queries and four separate demos from LATIC speaker `0010` (Korean accent),
from the local `zh_test_unseen.jsonl` split. All seven utterances have distinct texts.
Learner recordings and phone annotations come from
[LATIC](https://doi.org/10.21227/mtqj-qh10) by Xiao Zhang (2021); canonical speech was
generated with CosyVoice3. Original annotations and WAV contents are preserved.

Source manifest: `/home/xintong/whisper_pinyin/data/mdd_manifest_v4_local/zh_test_unseen.jsonl`.
Audio paths in the example manifests are relative to this folder.
`phone_symbols.txt` contains the Mandarin phone inventory from the local Chinese
manifests, excluding `sil` and `sp`.

The LATIC recordings and annotations are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The canonical TTS recordings are derived from these examples and shared under
the same license. The repository's MIT license applies to the code.
