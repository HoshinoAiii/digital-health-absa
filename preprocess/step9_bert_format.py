# -*- coding: utf-8 -*-
"""
============================================================
 第9步: BERT输入格式化 (BERT Input Formatting for ABSA)
 输入: dataset/train.csv, val.csv, test.csv (第7步输出)
 输出: dataset/bert_*/  目录下 input_ids.npy, attention_mask.npy, labels.npy
------------------------------------------------============
 ABSA输入格式 (BERT-BiLSTM-Attention):
   [CLS] review_text [SEP] aspect_category [SEP]
   - 每条评论的每个方面 -> 一个独立样本
   - label = aspect_sentiment (0=negative, 1=neutral, 2=positive)
   - max_length = 128 (评论中位数20token, 128覆盖>99%)

 ⚠️ 运行前提: 需安装 transformers + torch
   pip install transformers torch
   首次运行会自动下载 bert-base-uncased 模型(~440MB)
============================================================
"""
import os
import pandas as pd
import numpy as np

BASE     = r"D:\codex\nlp_thesis"
DATA_DIR = os.path.join(BASE, "dataset")
OUT_DIR  = os.path.join(BASE, "dataset", "bert_ready")
REPORT   = os.path.join(BASE, "bert_format_report.txt")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LENGTH = 128
BERT_MODEL = "bert-base-uncased"

# 方面类别 -> 英文描述(作为BERT输入的aspect文本)
ASPECT_TEXT = {
    "A1_UI_Design": "ui design",
    "A2_Functionality": "functionality",
    "A3_Cost_Subscription": "cost and subscription",
    "A4_Privacy_Security": "privacy and security",
    "A5_Performance": "performance",
    "A6_Customer_Support": "customer support",
    "A7_Content_Quality": "content quality",
}


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(str(s))

    log("=" * 64)
    log("  第9步: BERT输入格式化 报告")
    log("=" * 64)

    # ---- 检查依赖 ----
    try:
        from transformers import BertTokenizer
    except ImportError:
        log("\n[!] 未安装 transformers, 脚本已生成但无法执行。")
        log("    请先安装: pip install transformers torch")
        log("    然后重新运行本脚本。")
        log("\n    脚本功能说明:")
        log(f"    - 使用 {BERT_MODEL} tokenizer")
        log(f"    - ABSA输入格式: [CLS] text [SEP] aspect [SEP]")
        log(f"    - max_length = {MAX_LENGTH}")
        log(f"    - 输出: {OUT_DIR}/")
        log(f"      train: input_ids.npy, attention_mask.npy, labels.npy, aspects.npy")
        log(f"      val/test 同理")
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    log(f"\n【1. Tokenizer】{BERT_MODEL}")
    log(f"  vocab_size: {tokenizer.vocab_size}")
    log(f"  max_length: {MAX_LENGTH}")

    # ---- 处理每个划分 ----
    SENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}

    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
        log(f"\n【{split}】{len(df)} 行")

        # 展开为方面级样本: 每条评论 x 每个方面
        samples_text = []
        samples_aspect = []
        samples_label = []
        for _, row in df.iterrows():
            asps = [a.strip() for a in str(row.get("aspects", "")).split(",") if a.strip()]
            if not asps:
                continue  # 跳过无方面的评论(BERT-BiLSTM-Att需要aspect输入)
            for a in asps:
                if a not in ASPECT_TEXT:
                    continue
                # 方面级情感标签(宽表列)
                col = f"{a}_sentiment"
                lbl = row.get(col, "")
                if lbl not in SENT_MAP:
                    # 回退到文档级
                    lbl = row.get("sentiment_3label", "")
                if lbl not in SENT_MAP:
                    continue
                samples_text.append(str(row["clean_text"]))
                samples_aspect.append(a)
                samples_label.append(SENT_MAP[lbl])

        n_samples = len(samples_text)
        log(f"  方面级样本数: {n_samples}")

        if n_samples == 0:
            log(f"  [跳过] 无有效样本")
            continue

        # Tokenize
        aspect_texts = [ASPECT_TEXT[a] for a in samples_aspect]
        encodings = tokenizer(
            samples_text, aspect_texts,
            padding="max_length", truncation=True, max_length=MAX_LENGTH,
            return_tensors="np"
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = np.array(samples_label, dtype=np.int64)
        # 方面类别索引(辅助特征)
        aspect_list = list(ASPECT_TEXT.keys())
        aspect_idx = np.array([aspect_list.index(a) for a in samples_aspect], dtype=np.int64)

        # 保存
        split_dir = os.path.join(OUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "input_ids.npy"), input_ids)
        np.save(os.path.join(split_dir, "attention_mask.npy"), attention_mask)
        np.save(os.path.join(split_dir, "labels.npy"), labels)
        np.save(os.path.join(split_dir, "aspects.npy"), aspect_idx)
        log(f"  input_ids:       {input_ids.shape}")
        log(f"  attention_mask:  {attention_mask.shape}")
        log(f"  labels:          {labels.shape}")
        log(f"  aspects:         {aspect_idx.shape}")

        # 标签分布
        from collections import Counter
        lc = Counter(labels)
        for lbl in [0, 1, 2]:
            log(f"    {['negative','neutral','positive'][lbl]}: {lc.get(lbl,0)}")

    log(f"\n【输出】{OUT_DIR}/")
    log(f"\n  训练时加载方式:")
    log(f"    input_ids = np.load('bert_ready/train/input_ids.npy')")
    log(f"    attention_mask = np.load('bert_ready/train/attention_mask.npy')")
    log(f"    labels = np.load('bert_ready/train/labels.npy')")
    log(f"    aspects = np.load('bert_ready/train/aspects.npy')  # 辅助特征")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n{'='*64}")
    log(f"  第9步(BERT输入格式化)完成 ✓")
    log(f"{'='*64}")


if __name__ == "__main__":
    main()
