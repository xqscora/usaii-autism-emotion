# Autism Emotion Recognition — USAII Global AI Hackathon 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xqscora/usaii-autism-emotion/blob/master/demo_colab.ipynb)

**▶️ Live demo, no install — click the badge above, run the cells top to bottom, speak into your mic, and it reads the emotion in your voice.**

> Cora Zeng × Mujahid · voice-first emotion recognition for autistic children, with face as calibration.
> 建于 2026-06-15。这是项目蓝图 + 代码。训练在 **Kaggle GPU**（本地 Intel Arc 跑不动 wav2vec2 微调）。

---

## 🎯 目标

一个能读出**自闭症儿童**情绪的模型：麦克风听语音 → 识别情绪 → （胸前点阵屏显示）。
**核心难点（诚实）**：自闭症语音情绪**几乎没有标注数据**，且神经典型(NT)训练的模型会**自信地读错**自闭症声音（Apple 2025）。所以这不是"套一个现成 SER"就行——要专门设计绕过 bias。

---

## 📦 数据现状

| 数据集 | 内容 | 情绪标签? | 状态 | 用途 |
|---|---|---|---|---|
| **ASDSpeech** | 197 自闭症儿童语音的 49 维声学特征（123 录音 .mat） | ❌ 标签是 ADOS 严重度 | ✅ 已下 `datasets/ASDSpeech/` | **熟悉自闭症声音分布**（SSL/迁移的目标域） |
| **FER-Autism** | 自闭症儿童面部，6 类情绪，1200+220 图 | ✅ 6 类 | ⏳ 待手动下（Mendeley 要网页点 Download All）| **面部校准**信号 |
| **RAVDESS** | 通用成人语音，8 类情绪，有音频 | ✅ 8 类 | ⏳ 待下（Zenodo）| 训**情绪分类头**的标注源（NT） |
| **CALMED / CoSAm** | 自闭症多模态/语音情绪 | ✅ | 📧 已申请作者（等回复，大概率慢）| 理想的自闭症情绪标注，若拿到则金 |

> 关键：能直接训"情绪"的标签来自 **FER-Autism(面部) + RAVDESS(NT语音)**；自闭症**语音**情绪标注是 gap。ASDSpeech 给的是**无标签的自闭症声音**（正好做自监督/域适配）。

---

## 🏗️ 架构（voice-first + face 校准 + 防 bias）

```
                   ┌─────────────────────────────────────┐
   麦克风(语音) ──▶ │ SSL backbone (wav2vec2/HuBERT)        │  ← 在大规模语音上自监督预训练(现成)
                   │   + 继续自监督 on ASDSpeech 声音      │  ← 让它熟悉"自闭症孩子怎么发声"
                   └───────────────┬─────────────────────┘
                                   ▼
                   ┌─────────────────────────────────────┐
                   │ 情绪分类头 (fine-tune on RAVDESS)     │  ← 学"声音→情绪"映射
                   └───────────────┬─────────────────────┘
                       ┌───────────┴───────────┐
                       ▼                       ▼
            ┌────────────────────┐   ┌──────────────────────────┐
            │ per-wearer 校准      │   │ 不确定性弃权 (conformal)   │
            │ 每个孩子几个样本     │   │ 不确定就不显示/显示"?"     │  ← 堵住 confident-but-wrong
            └────────────────────┘   └──────────────────────────┘
                                   ▲
            面部(FER-Autism CNN) ───┘  ← 偶尔校准，不是常开摄像头
```

四个支柱（都有论文撑，见 `code-CogArch/docs/Emotion_Neuroscience_Canon/SER非典型语音_技术报告.md`）：
1. **SSL 预训练**熟悉自闭症声音（用 ASDSpeech）→ 再 fine-tune 情绪（RAVDESS）
2. **per-wearer 校准**：少样本适配个体（绕过"无单一自闭症声音"）
3. **不确定性弃权**：conformal prediction，不确定就别下结论（堵 confident-but-wrong）
4. **面部校准**：FER-Autism 做偶尔的 ground-truth check

---

## 📁 结构

```
USAII_autism_emotion/
├── README.md                  # 本文件（蓝图）
├── datasets/
│   ├── ASDSpeech/             # ✅ 已下（自闭症语音特征 + 代码）
│   ├── FER-Autism.zip         # ⏳ 待手动下
│   └── RAVDESS/               # ⏳ 待下
├── code/
│   ├── asdspeech_loader.py    # ASDSpeech 特征加载
│   ├── ser_model.py           # SSL backbone + 情绪头（待写）
│   ├── train_ser_kaggle.py    # Kaggle GPU 训练脚本（待写）
│   └── calibration.py         # per-wearer 校准 + conformal 弃权（待写）
└── notebooks/
    └── kaggle_train.ipynb     # Kaggle notebook（待写）
```

---

## 🚀 跑法（计划）

1. **本地**：准备数据、写/调代码、ASDSpeech 特征探索（Intel Arc 够）
2. **Kaggle GPU**：跑 wav2vec2 SSL 继续预训练 + 情绪 fine-tune（免费 T4/P100）
3. **demo**：本地加载训好的模型做推理 + （硬件原型：ESP32 + mic + LED 点阵）

---

## ⚖️ 诚实边界（写给评委也写给自己）

- 自闭症**语音**情绪无公开标注 → baseline 的情绪标签来自 NT 数据(RAVDESS)，**这正是要解决的 bias**，不藏着
- hackathon 产出是 **prototype**，不声称临床级
- 设备只在孩子/家长**同意**下用，显示**可纠正**（不是黑箱断言）
