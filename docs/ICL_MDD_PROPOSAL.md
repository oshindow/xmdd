# 多模态 In-Context Learning 用于跨语言发音错误检测（MDD）——研究方案

> 目标：ACL 主会 strong accept。
> 前提：下文所列各项预期结果均成立；本文件只讨论方案设计。

---

## 0. 一句话主张（Thesis）

**语音 LLM 的多模态 in-context demonstration 主要传递的是"错误模式先验"，而不是声学或音色上的相似性。据此设计的错误归纳式 ICL 训练，可以在没见过的语言和口音上，不做任何参数更新就完成发音错误检测，并且效果优于用同样 k 条样本做的 fine-tuning。**

整篇论文围绕一条主线展开，而不是四个并列的问题：

```
现象 (RQ1)          机理 (RQ3, RQ4)                   方法 (RQ2)                    推广
模型用了 demo，  →   demo 提供的是错误模式先验；    →   EI-ICL：让"不看 demo 就     →   同一机理在第二个任务
越陌生越依赖         canonical 提供的是比对参照          答不对"成为训练目标              和第二个 backbone 上成立
```

建议题目：
*What Do Multimodal Demonstrations Teach Speech LLMs? Error-Inductive In-Context Learning for Cross-Lingual Mispronunciation Detection*

---

## 1. 贡献（写在 Introduction 末尾）

1. **评测协议与 benchmark**：第一个多语言、多模态 ICL 的 MDD 评测协议。包含三种泛化层级（见过语言+见过口音 / 见过语言+没见过口音 / 没见过的语言），speaker 严格分离，demo 池固定。公开划分、demo 列表和评测脚本。
2. **机理发现**：用干预实验（而不是相关性分析）证明 demo 的主要作用是提供错误模式先验和错误率校准；canonical 语音的作用是提供比对参照，不是音色。
3. **方法**：Error-Inductive ICL（EI-ICL）。它由机理发现直接推出，让模型只能从 demo 中推断当前 episode 的错误规则。
4. **效果**：在没见过的语言（阿拉伯语、isiZulu）和没见过的 L1 口音上，不更新参数就优于标准 ICL 训练和同 k 条样本的 LoRA fine-tuning。

---

## 2. 任务定义与输入格式

**输入**（交错排列，每条 demo 自成一段）：

```
[demo_i] <canonical_phone> c_i  <canonical_audio> A^c_i  <learner_audio> A^l_i  <actual_phone> y_i
...（k 条）
[query]  <canonical_phone> c_q  <canonical_audio> A^c_q  <learner_audio> A^l_q  → 预测 y_q
```

- `c`：canonical 音素序列，经 BPE 编码。中文、英文、阿拉伯语、isiZulu 的音素都写成统一的类 IPA 字符串，再交给同一个 BPE tokenizer，这样没见过的语言无需扩充词表。
- `A^c`：canonical 语音（TTS 合成的标准发音）。
- `A^l`：学习者语音。
- `y`：学习者实际发出的音素序列。MDD 结果由 `c` 和 `y` 对齐后得到。
- 训练时只对 query 的 `y_q` 计算 loss；demo 的答案只作为上下文。

现有实现：[scripts/train_qwen3_canonical_prompt_bpe_icl.py](../scripts/train_qwen3_canonical_prompt_bpe_icl.py)，作为本方案的"标准 ICL 训练"基线。

---

## 3. 数据与评测协议

### 3.1 数据

| 语言 | 角色 | 数据集 | 标注类型 | 本地路径 |
|---|---|---|---|---|
| 中文 | 见过 | LATIC、Sichuan、SG 等学习者或方言数据，AISHELL-3 | canonical + actual | `data/jsonl_actual_split/` |
| 英文 | 见过 | L2-ARCTIC（6 个 L1、24 名说话人）、speechocean762 | canonical + actual（L2-ARCTIC 有人工标注子集） | `/data1/xintong/datasets/l2-arctic-release-v5.0`、`speechocean762` |
| 英文 | canonical 语音与母语参照 | CMU-Arctic（与 L2-ARCTIC 句子相同） | 母语者录音 | `/data1/xintong/datasets/cmu_arctic` |
| 阿拉伯语 | **没见过**（只做测试和 demo） | L2-KSU、IqraEval.2 真实错误子集（IS26）、QuranMB.v2 + Test_GT | canonical + actual | `/data1/xintong/datasets/l2_ksu`、`iqraeval_2025_2026` |
| isiZulu | **没见过**（只做检测） | IsiZulu L2 Learner Speech Corpus | 音素是否正确（二值）+ 声调错误 | `/data1/xintong/datasets/isiZulu_L2` |

> IqraEval 的 TTS 合成错误子集（`Iqra_TTS`）**不参与训练**，只用于第 6.3 节的受控干预实验。否则"没见过的语言"这个设定就不成立。

### 3.2 三种泛化层级（speaker 严格分离）

| 层级 | 定义 | 构造方式 |
|---|---|---|
| **L1 见过语言+见过口音** | 语言和口音都在训练中出现过，但**说话人没出现过** | 中文、英文按说话人划分 |
| **L2 见过语言+没见过口音** | 语言见过，这个 L1 背景没见过 | L2-ARCTIC 留出 2 个 L1（如越南语、西班牙语）的全部说话人；中文留出 SG 口音 |
| **L3 没见过的语言** | 训练中完全没出现过的语言 | 阿拉伯语、isiZulu |

### 3.3 Demo 池（个性化协议）

- 每个测试说话人的前 N 条语音（如 N=20）作为 **enrollment 池**，只用来当 demo，不作为 query。
- 其余语音作为 query。
- demo 按固定种子从 enrollment 池中选取，列表随 benchmark 一起公开。
- 这样测到的就是"模型在上下文中适应这个人"，不会和训练时见过的样本混在一起。

### 3.4 指标

- **主要指标**：检测的 Precision、Recall、F1；诊断准确率（检测到的错误中，实际音素预测对的比例）。
- **辅助指标**：False Rejection Rate、False Acceptance Rate；actual 音素的 PER。
- isiZulu 只报检测 F1 和声调错误检测 F1。
- 所有结果用 3 个随机种子 × 3 次 demo 重采样，报告均值、标准差和 paired bootstrap 显著性。

---

## 4. RQ1：模型有多依赖 demo？

**目标**：证明模型确实使用了 demo 的**内容**，而不只是因为上下文变长；并且依赖程度随泛化层级递增。

### 4.1 反事实 demo 实验

同一个 query，只改 demo，其余条件不变：

| 编号 | demo 条件 | 用来检验什么 |
|---|---|---|
| D0 | 0-shot | 下限 |
| D1 | 正确的 demo（同一 speaker） | 上限参照 |
| D2 | 标签打乱（demo 之间互换 `y`） | 模型是否依赖音频和标签的对应关系 |
| D3 | 标签换成 canonical（相当于"全部发对"） | 模型是否依赖 demo 中的错误信息 |
| D4 | 换成其他 speaker、同一 L1 | 说话人特有信息的贡献 |
| D5 | 换成其他 L1 | 口音特有信息的贡献 |
| D6 | 只有文本（去掉两段音频） | 文本通道的贡献 |
| D7 | 只有音频（去掉 `y`） | 声学适应的贡献 |
| D8 | token 长度相同的无关内容 | 排除上下文长度的影响 |

### 4.2 定义三个量化指标

- **Demo Reliance（DR）** = F1(D1) − F1(D3)，即去掉 demo 里的错误信息后损失了多少。
- **错误迁移率（Error Transfer Rate, ETR）**：在预测出的错误中，混淆对（canonical→actual）曾出现在 demo 里的比例，再和随机基线比较。
- **重合 / 非重合增益**：把 query 的真实错误分成两类（混淆对在 demo 中出现过 vs 没出现过），分别计算相对 0-shot 的 recall 增益。

### 4.3 预期结果

- 依赖程度：DR(L3) > DR(L2) > DR(L1)。
- 增益主要集中在"重合"的那类错误上。
- D8 ≈ D0，说明提升不是上下文变长带来的。

**图 1**：三个泛化层级 × shot 数（0/1/2/4/8/12）的 F1 曲线，并叠加 D3 条件作为对照。

---

## 5. RQ2：方法——Error-Inductive ICL（EI-ICL）

**设计原则**：由 RQ3 的机理推出——既然 demo 的作用是提供错误模式，训练就要让模型**只有从 demo 中推断出错误规则才能答对**。标准 ICL 训练的问题是 query 本身就足以预测，模型会学会忽略 demo。

### 5.1 组件一：合成口音 episode（核心）

- 每个训练 episode 随机抽一套**错误规则** R，比如 {zh→z, ing→in, 声调 3→2, 删除末尾 n}，共 1–5 条。规则来源：
  - 从真实学习者数据统计出的混淆矩阵中采样（保证规则真实可信）；
  - 加一部分完全随机的规则（迫使模型真正归纳规则，而不是记住常见错误）。
- 把 R 同时应用到这个 episode 的所有 demo 和 query 上：
  - 标签：`y = R(c)`；
  - 音频：用 CosyVoice（`third_party/CosyVoice`）按 `R(c)` 合成学习者语音，并随机克隆一个说话人的音色，把"错误模式"和"音色"分开。
- 因为 R 每个 episode 都重新抽，模型不可能记住它，只能从 demo 中归纳。这对应 AudioICL-Bench 提出的 **task learning**（从 demo 学新映射），而不是 **task recognition**（demo 只是提示已有能力）。

### 5.2 组件二：有信息的 demo 与干扰 demo 混合

- 训练时一个 episode 里的 demo 有 3 种：与 query 同一规则（有用）、不同规则（干扰）、无错误（中性）。比例随机。
- 模型必须学会判断哪些 demo 可用。这为 L2、L3 的真实场景做准备，因为推理时的 demo 不一定都相关。

### 5.3 组件三：demo 对比损失

同一个 query 分别配"有用的 demo"和"干扰 demo"，要求有用 demo 下的负对数似然比干扰 demo 下至少低 m：

```
L = L_LM(y_q | D+) + λ · max(0, m − [L_LM(y_q | D−) − L_LM(y_q | D+)])
```

这条损失把"利用 demo"直接写进了训练目标。

### 5.4 组件四：音素符号置换（模拟没见过的语言）

- 一部分 episode 对**输出音素符号**做随机置换，demo 和 query 使用同一套置换。
- 模型必须从 demo 中学出这次的符号对应关系，相当于在训练中模拟"没见过的音素集合"，直接对应 L3 场景。

### 5.5 训练课程

1. 阶段 1：标准 MDD 训练（0-shot），学会基本任务。
2. 阶段 2：真实数据的 ICL episode 与合成口音 episode 混合训练，逐步提高合成口音 episode 的比例和规则复杂度。
3. 全程随机 0–12 shot，只对 query 计算 loss。

### 5.6 消融

EI-ICL 完整版，逐一去掉组件一、二、三、四，观察在各泛化层级上的 F1 和 DR 变化。**预期**：组件一贡献最大；组件四主要提升 L3。

---

## 6. RQ3：demo 到底提供了什么？

### 6.1 四个竞争假设

| 假设 | demo 提供的东西 | 预期结论 |
|---|---|---|
| H1 错误模式先验 | 这个人或这个 L1 的混淆对 | **主要作用** |
| H2 声学或说话人适应 | 音色、信道、语速 | 次要 |
| H3 音素集合与输出格式 | 没见过的语言用哪些符号 | 只在 L3 上显著 |
| H4 错误率校准 | 这个人大概错多少 | 显著，并解释了 recall 的提升 |

### 6.2 因子分解实验（核心证据）

用 CosyVoice 合成受控 demo，逐个改变四个因子，其余保持不变：

| 因子 | 取值 |
|---|---|
| 错误模式 | 与 query 相同 / 不同 |
| 音色 | 克隆 query 说话人 / 其他音色 |
| 音频 | 有 / 无 |
| 错误率 | 0% / 10% / 30% |

- 对每个因子计算主效应和交互效应，得到每个因子对 F1 的贡献，做成方差分解图（**图 2**）。
- **预期**：错误模式的主效应最大，音色的主效应接近 0。

### 6.3 干预实验：只改 demo 的错误规则

- 在 IqraEval TTS 子集和合成中文、英文数据上，保持 query 不变，只改 demo 里的规则 R。
- 测量模型对 query 的预测是否跟着 R 变（例如 demo 中是 θ→s，模型预测里出现 θ→s 的比例是否上升）。
- 这是**因果证据**，回应"只是相关性"的质疑（**图 3**：规则迁移率随 shot 数的变化）。

### 6.4 校准分析（H4）

- 横轴是 demo 的错误率，纵轴是模型在 query 上预测的错误率，画斜率（**图 4**）。
- **预期**：斜率明显大于 0，说明 demo 在校准模型"敢不敢判错"，这解释了 MDD 中常见的偏向输出 canonical、recall 偏低的问题为什么会被缓解。

### 6.5 注意力与归因分析

- query 中被判为错误的音素 token，对 demo 各部分（音频、canonical、actual 标签中对应的混淆位置）的注意力分布和梯度归因。
- **预期**：注意力集中在 demo 中相同混淆对的 actual 标签 token 上。

---

## 7. RQ4：canonical 音素和 canonical 语音提供了什么？

### 7.1 输入消融

| 条件 | canonical 文本 | canonical 语音 |
|---|---|---|
| C0 | × | × |
| C1 | ✓ | × |
| C2 | × | ✓ |
| C3 | ✓ | ✓ |

### 7.2 canonical 语音的干预实验

| 编号 | 对 canonical 语音的改动 | 检验什么 |
|---|---|---|
| V1 | 克隆学习者音色 / 固定音色 / 随机音色 | 音色是否有作用（预期：无显著差异） |
| V2 | 换成另一句话的 TTS | 模型是否在做逐音素比对 |
| V3 | **在 canonical 语音里注入错误**（TTS 读改动过的音素序列，文本不变） | 模型是否在做声学上的差异检测：预期被注入的位置更容易被判错或判对 |
| V4 | 变速 0.8× / 1.2×、加噪声 | 对齐锚点的作用和鲁棒性 |

### 7.3 canonical 文本的作用

- 分别统计有/无 canonical 文本时的 recall 和 False Acceptance，量化"照抄 canonical"的偏差。
- **预期**：canonical 文本提供对齐和音素集合，但也带来照抄偏差；demo（H4 校准）可以抵消这个偏差。这把 RQ3 和 RQ4 连成一个结论。

**预期结论**：canonical 语音的作用是比对参照（V2、V3 显著），不是音色（V1 不显著）。

---

## 8. 基线

| 类别 | 基线 | 目的 |
|---|---|---|
| 同模型 | 0-shot；标准 ICL 训练（现有实现） | 证明 EI-ICL 的增益 |
| **参数更新** | 用同一批 k 条 demo 做 LoRA fine-tuning（每个说话人或每个 L1 单独训练）；说话人 adapter | 回答"为什么不直接 fine-tune" |
| 检索 | kNN：从 demo 中检索最相近的音素片段，直接投票 | 排除"只是检索"的解释 |
| 传统 MDD | wav2vec2 / HuBERT + CTC 的 MDD SOTA；IqraEval 官方基线 | 在标准 benchmark 上有竞争力 |
| 通用音频 LLM | Qwen2.5-Omni、Gemini 做 few-shot prompting | 说明通用模型的 ICL 做不了这个任务 |

**预期**：
- EI-ICL 在 L2、L3 上优于 k≤8 的 LoRA fine-tuning；k 较大时两者接近。
- 与 LoRA 相比，EI-ICL 不需要训练，部署成本低（**表 3**：F1 与适应时间、存储开销的对比）。

---

## 9. 模型

| 模型 | 作用 |
|---|---|
| Qwen3-ASR 0.6B | 主实验 |
| Qwen3-ASR 1.7B | 规模影响（本地已有 1.7B 的 CTC 实验） |
| 另一家族的语音 LLM（如 Qwen2.5-Omni-3B 或 Phi-4-Multimodal） | 证明结论不依赖具体 backbone |

主要结论（RQ1 的依赖梯度、RQ3 的 H1、RQ4 的比对参照）必须在三个模型上都成立。

---

## 10. 推广性实验

在第二个任务上重复 RQ1 和 RQ3 的核心实验，证明"demo 提供错误模式先验"不是 MDD 特有的：

- **首选**：带口音的音素识别或 ASR（L2-ARCTIC、中文方言），只用学习者的音频和转写作为 demo。
- **可选**：speechocean762 的发音评分（demo 提供评分尺度的校准，对应 H4）。

**预期**：同样观察到依赖梯度和错误模式迁移，结论可以提升为"语音 LLM 的多模态 ICL"层面的发现。

---

## 11. 预期结果与图表规划

| 编号 | 内容 | 支撑哪个结论 |
|---|---|---|
| 表 1 | 主结果：三个泛化层级 × 各方法 × 各语言的 F1 | EI-ICL 有效 |
| 表 2 | 标准 benchmark 上与 SOTA 比较（L2-ARCTIC、speechocean762、IqraEval） | 竞争力 |
| 表 3 | EI-ICL 与 LoRA 的 F1、适应成本比较 | ICL 相对 fine-tuning 的优势 |
| 表 4 | EI-ICL 消融 | 各组件的贡献 |
| 图 1 | 依赖梯度：shot 曲线 + 反事实条件 | RQ1 |
| 图 2 | 因子分解的方差分解 | RQ3 H1 |
| 图 3 | 干预后的规则迁移率 | RQ3 因果 |
| 图 4 | 校准斜率 | RQ3 H4 |
| 图 5 | canonical 干预（V1–V4） | RQ4 |
| 图 6 | 三个 backbone 上的结论一致性 | 推广性 |

---

## 12. 风险与应对

| 风险 | 应对 |
|---|---|
| 审稿人认为适合 Interspeech 而非 ACL | 标题、摘要、引言都以"多模态 ICL 的机理"为中心，MDD 作为测试场景；与 NLP 中 ICL 机理的工作（Min et al. 2022；task recognition vs task learning）直接对话 |
| 合成口音 episode 与真实错误分布不一致 | 规则主要从真实混淆矩阵采样；主要结论全部在真实学习者数据上报告；合成数据只用于训练和受控干预 |
| 没见过语言的音素集合与训练语言差异大 | 统一类 IPA 表示；组件四（符号置换）专门针对这一点；分析 L3 中"训练见过的音素"和"没见过的音素"的表现差别 |
| 12-shot 上下文太长（约 26 段音频） | demo 只保留学习者音频和标签，去掉 demo 的 canonical 语音作为高效变体；报告性能与上下文长度的权衡 |
| isiZulu 没有实际音素标注 | 只做检测和声调检测，明确写出限制 |
| 结论只在一个模型上成立 | 三个 backbone；若另一家族模型的 ICL 能力太弱，就在它上面也做 EI-ICL 训练 |

---

## 13. 与已有工作的定位（Related Work 结构）

1. **NLP 中的 ICL 机理**：Min et al. 2022（demo 中的标签是否重要）；task recognition vs task learning；Chan et al. 2022（数据分布如何使 ICL 出现）。
2. **语音与音频 ICL**：WavPrompt；SICL（Whisper 方言 ICL）；SALM（speech supervised in-context training）；COSMIC；Hsu et al. 2024（textless SLM 的 ICL）；Audio Flamingo；MiMo-Audio；评测类工作 ALICE、AudioICL-Bench。
3. **MDD**：CTC 和 wav2vec2 系列、基于 LLM 的 MDD、IqraEval 系列。
4. **本文定位**：据我们所知，这是第一个多模态 ICL 做 MDD 的工作，也是第一个用干预实验解释语音 LLM 中 demo 作用的工作。（投稿前需要做一次系统的文献检索确认。）

> 以上相关工作的出处和具体结论在写作前需逐篇核对原文。

---

## 14. 时间规划（约 16 周，目标 ARR 投稿周期）

| 周 | 任务 |
|---|---|
| 1–2 | 固定评测协议：各语言的说话人划分、enrollment 池、统一音素表示、评测脚本 |
| 3–4 | 基线：标准 ICL（现有）、0-shot、LoRA、kNN、传统 MDD；RQ1 反事实实验 |
| 5–7 | EI-ICL：合成口音 episode 的数据生成（CosyVoice）与训练 |
| 8–9 | RQ3 因子分解、干预、校准实验 |
| 10 | RQ4 canonical 实验 |
| 11–12 | 第二、第三个 backbone；推广性任务 |
| 13 | 显著性检验、所有图表 |
| 14–16 | 写作、内部审阅、公开 benchmark |

---

## 15. 从 accept 到 strong accept 的检查清单

- [ ] 一条主线：现象 → 机理 → 方法 → 推广，而不是四个并列的问题
- [ ] 方法由机理直接推出（EI-ICL ← H1）
- [ ] 机理结论有**干预实验**支撑，不只是相关性
- [ ] 没见过的语言用**真实学习者数据**测试（L2-KSU、IqraEval IS26）
- [ ] 与同 k 条样本的 fine-tuning 公平比较
- [ ] 三个 backbone 上结论一致
- [ ] 至少一个 MDD 之外的任务上结论成立
- [ ] 说话人严格分离，demo 池独立
- [ ] 多种子、bootstrap 显著性
- [ ] 公开 benchmark、划分和代码
