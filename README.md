# Aspect-Based Sentiment Analysis of Digital-Health App Reviews

Code, data and lexicons for the paper:

> **基于 BERT-BiLSTM-Attention 的数字健康平台用户体验方面级情感分析研究**
> *(Aspect-Based Sentiment Analysis of Digital-Health Platform User Experience based on BERT-BiLSTM-Attention)*

This repository is the official reproducibility release. It contains everything needed to
reproduce the data pipeline, the aspect taxonomy, the lexical annotation rules and all
model results reported in the paper.

---

## 1. Overview

The study analyses user reviews of **20 digital-health mobile apps** (Google Play + Apple
App Store) at the **aspect level**, using a 7-aspect taxonomy and 3-way polarity
(negative / neutral / positive). The proposed model is **BERT-BiLSTM-Attention**; it is
compared with TextCNN, BiLSTM, BERT fine-tuning, BERT + attention pooling and an
LLM few-shot baseline.

### Dataset at a glance (figures reported in the paper)

| Item | Value |
|---|---|
| Apps | 20 (Google Play + Apple App Store) |
| Aspect–sentiment pairs (final) | **31,349** |
| Train / Val / Test | **25,066 / 3,182 / 3,101** |
| High-confidence subset (all) | 10,137 |
| High-confidence subset (test) | 1,036 |
| Split strategy | `GroupShuffleSplit` grouped by `review_id` (no cross-split leakage) |
| Labels | negative / neutral / positive |
| Label mapping | 1–2★ → negative, 3★ → neutral, 4–5★ → positive |

> The raw long-table `data/absa_long.csv` contains **35,717** aspect–sentiment rows.
> After the cleaning/filtering step in `code/data_prep.py` (valid aspect ∧ valid label ∧
> non-empty text) it yields exactly **31,349** rows — the number used everywhere in the paper.
> Running `code/data_prep.py` reproduces the split 25,066 / 3,182 / 3,101 and the
> high-confidence counts above.

---

## 2. Repository structure

```
digital-health-absa/
├── README.md                     ← this file
├── ENVIRONMENT.md                ← software environment / how to run
├── LICENSE                       ← MIT (source code)
├── NOTE-ON-DATA.md               ← data provenance & usage terms
├── CITATION.cff                  ← machine-readable citation
├── code/                         ← modelling code (paper pipeline)
│   ├── data_prep.py              ← clean + 8:1:1 grouped split + vocab
│   ├── encode_bert.py            ← BERT encoding (masked mean pooling)
│   ├── models.py                 ← BERT-BiLSTM-Attention, TextCNN, BiLSTM, …
│   ├── train.py                  ← training loop
│   ├── run_all.py                ← full experiment runner (all tables)
│   ├── run_experiments.py        ← ablation / multi-seed experiments
│   ├── run_extras.py             ← auxiliary analyses
│   ├── eda.py                    ← descriptive statistics
│   ├── llm_baseline.py           ← LLM few-shot baseline (optional, needs API key)
│   ├── sanity_check.py
│   └── requirements.txt
├── preprocess/                   ← data-preparation scripts (steps 1–9)
│   ├── preprocess.py             ← step 1: cleaning & tokenisation
│   ├── step2_labeling.py         ← step 2: sentiment labels
│   ├── step3_aspect.py           ← step 3: ASPECT LEXICON + extraction  ★
│   ├── step4_absa_labeling.py    ← step 4: SENTIMENT LEXICON + window ±5  ★
│   ├── step5_7_dedup_split.py    ← steps 5–7: dedup & split
│   ├── step8_visualization.py    ← step 8: figures
│   └── step9_bert_format.py      ← step 9: BERT input formatting
├── lexicon/                      ← extracted, machine-readable rules  ★
│   ├── aspect_lexicon.json       ← 7-aspect trigger-word lexicon (EN + ZH)
│   ├── sentiment_lexicon.json    ← positive/negative words, negators, intensifiers
│   └── config.json               ← window size, scoring rule, split ratio, labels
└── data/
    ├── absa_long.csv             ← aspect-level long table (one row per aspect–sentiment pair)
    └── README_data.md            ← field description
```

★ = the three artefacts explicitly requested by the reviewer for reproducibility
(aspect lexicon, sentiment lexicon, window size).

---

## 3. The 7-aspect taxonomy

| ID | Aspect (EN) | 中文 | Core trigger examples |
|---|---|---|---|
| A1 | UI Design | 界面设计 | ui, interface, layout, navigation, button, 界面, 布局, 导航 |
| A2 | Functionality | 功能实用性 | feature, tracker, reminder, sync, 功能, 提醒, 同步, 记录 |
| A3 | Cost / Subscription | 费用订阅 | price, fee, subscription, premium, refund, 价格, 订阅, 会员, 退款 |
| A4 | Privacy / Security | 隐私安全 | privacy, data, permission, security, 隐私, 数据, 权限, 安全 |
| A5 | Performance | 性能稳定 | crash, freeze, lag, bug, glitch, 崩溃, 闪退, 卡顿, 错误 |
| A6 | Customer Support | 客服支持 | support, response, reply, feedback, 客服, 回复, 反馈, 投诉 |
| A7 | Content Quality | 内容质量 | content, article, course, audio, quality, 内容, 课程, 音频, 质量 |

The full, non-abridged word lists (bilingual) are in `lexicon/aspect_lexicon.json`.

---

## 4. Aspect extraction and aspect-level sentiment annotation

**Aspect extraction** (multi-label) matches every trigger word of the taxonomy against the
cleaned review text — see `preprocess/step3_aspect.py::extract_aspects`.

**Aspect-level sentiment** is assigned by a lexicon + window rule
(`preprocess/step4_absa_labeling.py`):

1. locate each matched aspect keyword in the token sequence;
2. take a window of **± 5 tokens** around the keyword (`WINDOW = 5`);
3. score = Σ(sentiment word) with **negation reversal** and **intensifier weighting**;
4. map the score to negative / neutral / positive;
5. if no sentiment word occurs in the window, **fall back to the document-level label**
   and mark the row `confidence = low` (rows with a sentiment word in the window are
   `confidence = high` — the high-confidence subset used as a quasi-gold set).

All thresholds and word lists are externalised in `lexicon/` so the rule can be audited
and re-run without reading the source code.

---

## 5. Quick start

```bash
pip install -r code/requirements.txt

# 1) data pipeline: cleaning → labels → aspects → ABSA labels → split → BERT format
python preprocess/preprocess.py
python preprocess/step2_labeling.py
python preprocess/step3_aspect.py
python preprocess/step4_absa_labeling.py
python preprocess/step5_7_dedup_split.py
python preprocess/step8_visualization.py
python preprocess/step9_bert_format.py

# 2) modelling: build cache, encode with BERT, run all models & tables
cd code
python data_prep.py
python encode_bert.py
python run_all.py
```

> The pre-processing scripts read/write with relative paths under `data/`.
> `code/encode_bert.py` requires `bert-base-uncased` (downloaded automatically by
> HuggingFace `transformers`).

---

## 6. Environment

See [`ENVIRONMENT.md`](ENVIRONMENT.md). Short version: Python 3.10+ with
`numpy / pandas / scipy / scikit-learn / torch / transformers`; the LLM baseline
additionally needs `openai` and an `OPENAI_API_KEY` (it is skipped automatically if the
key is absent — no results are fabricated).

---

## 7. Citation

If you use this code or data, please cite the paper (see `CITATION.cff`):

```bibtex
@misc{digitalhealth_absa,
  title  = {Code and data for "Aspect-Based Sentiment Analysis of Digital-Health Platform User Experience based on BERT-BiLSTM-Attention"},
  author = {Yu, Jingyi}、doi = {10.5281/zenodo.23347005},
  year   = {2026},
  note   = {Zenodo/GitHub release}
}
```

---

## 8. License

- **Source code** — MIT License (`LICENSE`).
- **Data & lexicons** — released for academic reproducibility only; see
  [`NOTE-ON-DATA.md`](NOTE-ON-DATA.md).

---

## 中文说明（简）

本仓库是论文《基于 BERT-BiLSTM-Attention 的数字健康平台用户体验方面级情感分析研究》
的**可复现性发布**，包含：

- **代码**（`code/` 建模 + `preprocess/` 数据处理全流程）
- **数据**（`data/absa_long.csv`，方面级长表）
- **规则文件**（`lexicon/`：七类方面触发词表、情感词表、否定词/程度副词、窗口大小 ±5 及标注规则）

复现步骤见第 5 节。数据经 `code/data_prep.py` 清洗后得到论文所用 **31,349** 个
方面-情感对，按 `review_id` 分组划分为 **25,066 / 3,182 / 3,101**。
