# 中英双语 BPE ICL（canonical dropout）版本：新服务器复现指南

本文说明如何在一台新的 GPU 服务器上复现 `zh_en_canonical_dropout.log` 对应的训练：

- 配置：[config/qwen3_zh_en_bpe_icl12_canonical_dropout_5k_2gpu.yaml](config/qwen3_zh_en_bpe_icl12_canonical_dropout_5k_2gpu.yaml)
- 训练脚本：[scripts/train_qwen3_multilingual_bpe_icl.py](scripts/train_qwen3_multilingual_bpe_icl.py)
- 分支：`final`

模型从 HuggingFace 上的 `Qwen/Qwen3-ASR-0.6B-hf` 预训练权重开始训练，**不需要任何之前训练过的 checkpoint**。

---

## 0. 需要准备的东西一览

| 类别 | 内容 | 大小 | 来源 |
|---|---|---:|---|
| 代码与标注 | 本仓库 `final` 分支（含 `data/jsonl_actual_split/`、`l2_arctic_peng2021/`、`data/lang_jsonl/tokens.txt`） | < 100 MB | `git clone` |
| 环境 | Python 3.11 + torch 2.5.1/cu121 + transformers 5.16.1 + flash-attn 2.8.3 | — | 第 2 节 |
| 预训练模型 | `Qwen/Qwen3-ASR-0.6B-hf` | 1.5 GB | HuggingFace |
| 中文学习者音频 | AISHELL-3、LATIC、MagicHub 新加坡与四川 子集 | 7.5 GB（实际用到的文件） | 原服务器 rsync，或官方下载 |
| 中文 canonical 语音 | CosyVoice3 合成的同音色标准发音 + 生成日志 | 8.9 GB | **只能从原服务器 rsync**（自行合成的数据） |
| 英文学习者音频 | L2-ARCTIC v5.0 的 wav | 1.1 GB（实际用到的文件） | 原服务器 rsync，或官方申请 |
| 英文 canonical 语音 | CosyVoice3 合成的 L2-ARCTIC 标准发音 | 0.5 GB | **只能从原服务器 rsync** |

硬件：原实验使用 2 × NVIDIA A40（48 GB），每卡 batch 1，梯度累积 24，bf16 混合精度。

---

## 1. 拉代码

```bash
git clone -b final https://github.com/oshindow/whisper_pinyin.git
cd whisper_pinyin
```

标注文件已经在仓库中：

| 路径 | 内容 |
|---|---|
| `data/jsonl_actual_split/train.jsonl` / `dev.jsonl` | 中文：65,797 / 2,035 条，含 `canonical_phones`、`actual_phones`、`speaker_id`、`accent_id`、`l2_wav` |
| `l2_arctic_peng2021/{train,dev,test}.jsonl` | 英文 L2-ARCTIC（Peng et al. 2021 划分）：2,436 / 263 / 900 条，统计见 `l2_arctic_peng2021/STATS.md` |
| `data/lang_jsonl/tokens.txt` | 音素表 |

---

## 2. 环境

原服务器使用 conda 环境 `qwen3_omni`，以下为其中实际使用的版本：

| 包 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch / torchaudio | 2.5.1+cu121 |
| transformers | 5.16.1（原生 Qwen3-ASR 需要 ≥ 5.13.0） |
| pytorch-lightning | 2.4.0 |
| flash-attn | 2.8.3.post1 |
| accelerate | 1.14.0 |
| huggingface_hub | 1.30.0 |
| tokenizers | 0.23.2 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| PyYAML | 6.0.3 |
| soundfile | 0.14.0 |
| tensorboard | 2.21.0 |
| torchmetrics | 1.9.0 |

```bash
conda create -n qwen3_omni python=3.11 -y
conda activate qwen3_omni

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==5.16.1 pytorch-lightning==2.4.0 accelerate==1.14.0 \
    huggingface_hub==1.30.0 tokenizers==0.23.2 safetensors==0.8.0 \
    numpy==2.4.6 PyYAML==6.0.3 soundfile==0.14.0 tensorboard==2.21.0 torchmetrics==1.9.0

# flash-attn 需要与 torch 2.5 / CUDA 12 / Python 3.11 匹配。
# 有 nvcc 时可直接编译；否则去 https://github.com/Dao-AILab/flash-attention/releases
# 下载 cu12 + torch2.5 + cp311 + cxx11abiFALSE 的 wheel 安装。
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

检查：

```bash
python -c "import torch, transformers, flash_attn; \
from transformers import Qwen3ASRForConditionalGeneration; \
print(torch.__version__, transformers.__version__, flash_attn.__version__, torch.cuda.is_available())"
```

---

## 3. 预训练模型

配置里 `local_files_only: true`，训练时**不会联网下载**，必须先下好。原服务器把缓存放在仓库内的 `.cache-qwen3/huggingface`（已被 `.gitignore` 忽略），snapshot 为 `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`。

```bash
export HF_HOME=$PWD/.cache-qwen3/huggingface
hf download Qwen/Qwen3-ASR-0.6B-hf          # 旧版 CLI：huggingface-cli download Qwen/Qwen3-ASR-0.6B-hf
```

也可以直接从原服务器拷贝：

```bash
rsync -a OLD_HOST:/home/xintong/whisper_pinyin/.cache-qwen3/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf \
      .cache-qwen3/huggingface/hub/
```

训练和评测时都需要设置同一个 `HF_HOME`。

---

## 4. 音频数据

### 4.1 四个音频根目录

配置里有四个根目录，新服务器上需要全部准备好：

| 配置项 | 原服务器路径 | 目录结构 |
|---|---|---|
| zh `data_root` | `/data2/xintong/datasets/datasets`（`/data2/xintong/mandarin_accent` 是指向它的软链接） | `aishell3/train/wav_16k/<spk>/<utt>.wav`、`LATIC/WAVE/...`、`magichub_multiaccent/{magichub_singapore,Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence}/...` |
| zh `canonical_audio_root` | `/data2/xintong/datasets/cosyvoice3/cosyvoice3` | 与上面**相同的相对路径**，外加 `logs/*.jsonl` |
| en `l2_audio_root` | `/data1/xintong/datasets/l2-arctic-release-v5.0` | `<SPEAKER>/wav/<utt>.wav` |
| en `canonical_audio_root` | `/data2/xintong/datasets/l2-arctic-release-v5.0` | `<SPEAKER>/wav/<utt>.wav` |

> ⚠️ **中文数据的目录名有硬性要求。** 中文 jsonl 中的 `l2_wav` 是原服务器上的绝对路径。代码会截取其中 `/datasets/datasets/` 之后的部分，拼到 `data_root` 下面；canonical 语音也是按这个 L2 路径中 `/datasets/datasets/` 之后的部分，到 `canonical_audio_root` 下查找。所以：
>
> 1. **新服务器上的中文 `data_root` 路径必须以 `/datasets/datasets` 结尾**（例如 `/mnt/disk1/xintong/datasets/datasets`），否则会报 `No canonical TTS audio for ... rows`。
> 2. `canonical_audio_root/logs/` 下至少要有一个 `*.jsonl`，否则会报 `No CosyVoice logs matched`。
>
> 最省事的做法：新服务器上建立**与原服务器完全相同的绝对路径**（可用软链接），配置里的路径就一个都不用改。

### 4.2 从原服务器同步（推荐）

只同步训练和评测实际用到的文件，总共约 18 GB。先在原服务器上生成文件列表（列表中都是相对路径，两台服务器通用）：

```bash
# 在原服务器的仓库目录下
python scripts/list_required_audio.py \
    --config config/qwen3_zh_en_bpe_icl12_canonical_dropout_5k_2gpu.yaml \
    --extra-manifest l2_arctic_peng2021/test.jsonl \
    --output audio_lists
# 生成 audio_lists/{zh_l2,zh_canonical,en_l2,en_canonical}.txt
# 条数：67,832 / 67,833（多出的一条是 logs/ 目录）/ 3,599 / 3,599
```

把 `audio_lists/` 拷到新服务器后，在新服务器上执行（`NEW_*` 换成你新服务器上的目录）：

```bash
OLD=OLD_HOST   # 原服务器的 ssh 地址
NEW_ZH=/path/to/datasets/datasets              # 必须以 /datasets/datasets 结尾
NEW_ZH_CANON=/path/to/cosyvoice3/cosyvoice3
NEW_EN=/path/to/l2-arctic-release-v5.0
NEW_EN_CANON=/path/to/l2-arctic-cosyvoice3

rsync -a --files-from=audio_lists/zh_l2.txt        $OLD:/data2/xintong/datasets/datasets/                 $NEW_ZH/
rsync -a -r --files-from=audio_lists/zh_canonical.txt $OLD:/data2/xintong/datasets/cosyvoice3/cosyvoice3/ $NEW_ZH_CANON/
rsync -a --files-from=audio_lists/en_l2.txt        $OLD:/data1/xintong/datasets/l2-arctic-release-v5.0/  $NEW_EN/
rsync -a --files-from=audio_lists/en_canonical.txt $OLD:/data2/xintong/datasets/l2-arctic-release-v5.0/  $NEW_EN_CANON/
```

`zh_canonical` 那一行多了 `-r`，是为了把 `logs/` 整个目录同步过去。

### 4.3 从官方来源下载（原服务器不可用时）

学习者音频可以重新下载，但 **canonical 语音是本项目用 CosyVoice3 合成的，只能从原服务器拷贝，或者重新合成**（见 4.4）。

| 数据 | 来源 | 放置位置 |
|---|---|---|
| AISHELL-3 | OpenSLR SLR93 | `$NEW_ZH/aishell3/train/wav_16k/<spk>/*.wav`（16 kHz 单声道；原始包的 `wav/` 为 44.1 kHz，需要自行重采样到 `wav_16k/`） |
| LATIC | MagicHub | `$NEW_ZH/LATIC/WAVE/...` |
| MagicHub 新加坡口音普通话、四川方言朗读语料 | MagicHub（需注册） | `$NEW_ZH/magichub_multiaccent/magichub_singapore/...`、`$NEW_ZH/magichub_multiaccent/Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence/...` |
| L2-ARCTIC v5.0 | 官网申请（TAMU PSI Lab） | `$NEW_EN/<SPEAKER>/wav/*.wav`，按说话人解压 |

放好后，可以用 4.2 中生成的文件列表检查是否有缺失：

```bash
while read p; do [ -e "$NEW_ZH/$p" ] || echo "MISSING $p"; done < audio_lists/zh_l2.txt | head
```

### 4.4 重新合成 canonical 语音（仅在无法拷贝时）

中文使用 [scripts/mdd/generate_cosyvoice_pairs.py](scripts/mdd/generate_cosyvoice_pairs.py)：以每条学习者语音自身作为 zero-shot prompt，合成同音色的标准发音。该脚本依赖原服务器上仓库之外的 `/home/xintong/Joycent-code`，以及 CosyVoice3 的模型权重，新服务器上需要另外准备。合成结果和原数据不会完全一致，**严格复现请直接拷贝**。

---

## 5. 修改配置

复制一份配置，只改路径相关的几项：

```bash
cp config/qwen3_zh_en_bpe_icl12_canonical_dropout_5k_2gpu.yaml config/my_zh_en_canonical_dropout.yaml
```

| 配置项 | 改成 |
|---|---|
| `experiment.output_root` | 新服务器上存放 checkpoint 和日志的目录 |
| `data.datasets[0].data_root`（zh） | `$NEW_ZH`（以 `/datasets/datasets` 结尾） |
| `data.datasets[0].canonical_audio_root`（zh） | `$NEW_ZH_CANON` |
| `data.datasets[1].train_path` / `val_path`（en） | `l2_arctic_peng2021/train.jsonl` / `l2_arctic_peng2021/dev.jsonl`（仓库内的版本，与原服务器上 `/data1/xintong/datasets/english_mdd_jsonl/l2_arctic_peng2021/` 下的内容逐字节相同） |
| `data.datasets[1].l2_audio_root`（en） | `$NEW_EN` |
| `data.datasets[1].canonical_audio_root`（en） | `$NEW_EN_CANON` |
| `training.devices` | 使用的 GPU 数（原实验为 2） |

其余超参数保持不变即可复现。若 GPU 数不是 2，想保持有效 batch 48，需要相应调整 `gradient_accumulation_steps`（有效 batch = GPU 数 × `per_device_batch_size` × `gradient_accumulation_steps`）。

---

## 6. 训练

```bash
conda activate qwen3_omni
export HF_HOME=$PWD/.cache-qwen3/huggingface
CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/train_qwen3_multilingual_bpe_icl.py \
    --config config/my_zh_en_canonical_dropout.yaml > zh_en_canonical_dropout.log 2>&1 &
```

- 直接用 `python` 启动即可，Lightning 会按 `training.devices` 自动拉起 DDP 进程，不需要 `torchrun`。
- 正常启动时日志开头应为：

  ```
  Loaded train: zh 65797 rows from data/jsonl_actual_split/train.jsonl
  Loaded train: en 2436 rows from .../l2_arctic_peng2021/train.jsonl
  Loaded val: zh 2035 rows ...
  Loaded val: en 263 rows ...
  samples=68,233, effective_global_batch=48, steps/epoch=1422, max_steps=5000
  ```

- 共 5,000 步，每 500 步在 dev 上做一次 0-shot 验证，按 `val/icl_score` 保存最好的 3 个 checkpoint 以及 `last.ckpt`，位置是 `<output_root>/checkpoints/`；TensorBoard 日志在 `<output_root>/logs/`。
- 从中断处继续：把配置中的 `experiment.resume` 设为 `auto`（自动找最新的 checkpoint），或加参数 `--resume-from <ckpt>`。

---

## 7. 评测

```bash
export HF_HOME=$PWD/.cache-qwen3/huggingface
# dev，0/1 shot
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qwen3_multilingual_bpe_icl_dev.py \
    --config config/my_zh_en_canonical_dropout.yaml \
    --checkpoint <output_root>/checkpoints/<best>.ckpt \
    --output results/zh_en_canonical_dropout_dev \
    --shots 0 1

# L2-ARCTIC 测试集（Peng 2021 的 6 位测试说话人）
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qwen3_multilingual_bpe_icl_dev.py \
    --config config/my_zh_en_canonical_dropout.yaml \
    --checkpoint <output_root>/checkpoints/<best>.ckpt \
    --output results/zh_en_canonical_dropout_en_test \
    --languages en --split test --manifest l2_arctic_peng2021/test.jsonl \
    --shots 0 1
```

---

## 8. 常见报错

| 报错 | 原因 | 解决 |
|---|---|---|
| `No CosyVoice logs matched .../logs/*.jsonl` | zh `canonical_audio_root` 下没有 `logs/` | 同步 `logs/` 目录（4.2 中 `zh_canonical` 那一行） |
| `No canonical TTS audio for N/M rows` | 中文 `data_root` 路径不以 `/datasets/datasets` 结尾，或 canonical 文件缺失 | 按 4.1 的要求调整目录 |
| `Missing L2/canonical WAV for N L2-ARCTIC rows` | 英文某个根目录的结构不是 `<SPEAKER>/wav/<utt>.wav` | 检查 `l2_audio_root` 和 `canonical_audio_root` |
| `OSError: ... local_files_only` / 找不到模型 | 模型没下载，或 `HF_HOME` 没设 | 按第 3 节下载，并设置同一个 `HF_HOME` |
| `ImportError: Qwen3ASRForConditionalGeneration` | transformers 版本太低 | 安装 5.16.1（至少 5.13.0） |
| flash-attn 相关报错 | wheel 与 torch/CUDA/Python 版本不匹配 | 换成匹配的 wheel；应急时把 `model.attention_implementation` 改为 `sdpa`（速度和显存会变，结果不保证与原实验完全一致） |
