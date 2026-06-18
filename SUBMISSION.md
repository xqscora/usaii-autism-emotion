# 提交指南 — USAII Global AI Hackathon 2026

## ⏰ Deadline & 必交物
- **提交：6/21 11:59 PM ET**（还剩 ~3 天）｜获奖公布 6/27
- 必交：① 在 **Devpost** 提交项目　② **3–5 分钟 demo video**（YouTube / Vimeo / Loom）
- 要 **qualifier code** 才能交；**team leader** 在 Devpost 创建项目 → 用每个人的**注册邮箱**邀请队友

## 📊 评分 rubric（照这个优化，别瞎堆技术）
| 维度 | 权重 | 我们怎么打满 |
|---|---|---|
| **Problem Understanding & Context** | **30%** | 我们最强：自闭症孩子表达非典型→不被理解→拿不到支持；**被理解是被支持的前提**。加上诚实的 bias 分析（NT 模型对自闭症 *confident-but-wrong*，Apple 2025）|
| AI Reasoning | 20% | SSL(wav2vec2) embedding → 分类 → **per-child 个性化** → 不确定**弃权** |
| Solution Design & Architecture | 20% | voice-first + 个性化层 + 长期 Cerome；human-in-the-loop |
| Impact & Insight | 20% | 个性化**救得最多的正是通用模型读得最差的人**（+19pt）= 自闭症案例 |

> 🎯 30% 在 Problem Understanding——**video 和提交文案要重点讲"问题有多真、我们多懂它 + 多诚实"**，不是炫准确率。这是你的主场。

## 📋 提交时要披露的
- **数据源**：RAVDESS（公开，Zenodo）、ASDSpeech（公开，GitHub）——都公开，合规
- **工具 / AI**：wav2vec2-superb（HuggingFace）、transformers、scikit-learn
- ⚠️ 不能用私人 / 个人 / 敏感数据 —— 我们没用，合规

## 🎬 Demo video 脚本（3–5 分钟，分镜 + 旁白）

> 评委想看 **input → AI 处理 → output** 的完整 workflow。旁白可你自己念，或我用 TTS 配。

**① 问题（0:00–0:40）** — 画面：一个孩子的简单场景 / 文字卡
> "When an autistic kid is overwhelmed, it often comes out in a way the people around them can't read. The support is right there — but it never lands, because the kid isn't understood. Being understood is the precondition for being supported. And generic emotion AI makes it worse: trained on neurotypical speech, it's *confidently wrong* on autistic kids."

**② 方案 + 现场 demo（0:40–1:50）** — 画面：跑 `demo_emotion.py` 或 Colab，语音进、情绪出
> "So we built a voice-first emotion reader. Speech goes in, it reads the emotion — and when it's not sure, it says so instead of guessing. Here it is running: [说一句话 → 屏幕出 emotion + confidence]. Baseline is 64% speaker-independent on RAVDESS — an honest number, chance is 12.5%."

**③ 核心：个性化（1:50–3:00）** — 画面：personalize_sweep 曲线 + 那张 67%→77% 表
> "But the real idea isn't one generic model — it's that *every child gets a model that learns them*. Give it just a few samples of one kid and it understands *them* better. We measured it: 67% → 77% with three samples per emotion. And here's the part that matters — the speakers the generic model read *worst* improved the *most*, +19 points for the hardest one. That is exactly the autistic case: atypical expression, generic model fails, personalization rescues it."

**④ 长期 vision + 诚实边界（3:00–3:50）** — 画面：架构图 / 文字
> "Long-term, each child builds up a personalized model over time — a framework we call Cerome — so even when they express things in a blurry, non-standard way, the model that's been learning them still gets it. To be honest about limits: today's model is trained on neurotypical speech, the autism-specific adaptation is next, it's a prototype not a clinical tool, and a human always confirms."

**⑤ 收尾（3:50–4:10）**
> "Being understood is the first step to being supported. That's what we're building. Repo and live demo in the description."

## ✅ 提交前 checklist
- [ ] 确认 **team leader** + **qualifier code**（你还是 Mujahid 当 leader？）
- [ ] Devpost 创建项目 + 邀请队友（注册邮箱）
- [ ] 录 demo video（3–5min）→ 传 YouTube（unlisted 即可）
- [ ] 填：问题、方案、数据源（RAVDESS/ASDSpeech）、工具（wav2vec2 等）
- [ ] 贴 **GitHub** + **Colab** 链接
- [ ] 标 inspiration / what's next（个性化 Cerome）

---
*技术细节 → `README.md`；项目叙事 → `PROJECT_BRIEF.md`。*
