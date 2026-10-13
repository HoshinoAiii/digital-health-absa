# -*- coding: utf-8 -*-
"""
============================================================
 第5-7步: 去重 + 类别不平衡处理 + 数据集划分
 输入: absa_labeled.csv (19052 行, 含方面级标签)
 输出: dataset/ 目录下 train/val/test CSV + split_report.txt
------------------------------------------------------------
 第5步 去重: 保留首次出现的文本, 去除完全重复(避免数据泄露)
 第6步 不平衡: 记录class_weight供训练用; 对训练集做分层抽样保持分布
               (不在数据层做SMOTE/过采样, 因BERT输入不宜合成文本;
                用class_weight='balanced'策略, 输出权重文件)
 第7步 划分: 分层抽样 80/10/10 (按 sentiment_3class 分层)
             确保三折分布一致; 按review_id划分避免同一评论跨集
============================================================
"""
import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE     = r"D:\codex\nlp_thesis"
IN_CSV   = os.path.join(BASE, "absa_labeled.csv")
OUT_DIR  = os.path.join(BASE, "dataset")
REPORT   = os.path.join(BASE, "split_report.txt")
FIG_DIR  = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

RATIO_TRAIN, RATIO_VAL, RATIO_TEST = 0.8, 0.1, 0.1
RANDOM_STATE = 42


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(str(s))

    df = pd.read_csv(IN_CSV)
    log("=" * 64)
    log("  第5-7步: 去重 + 不平衡处理 + 数据集划分 报告")
    log("=" * 64)
    log(f"\n【输入】{IN_CSV}  ({len(df)} 行)")

    # ================================================================
    # 第5步: 去重
    # ================================================================
    log(f"\n{'='*40}")
    log(f"  第5步: 去重处理")
    log(f"{'='*40}")

    n_before = len(df)
    # 完全重复(clean_text相同) -> 保留第一条
    dup_mask = df.duplicated(subset=["clean_text"], keep="first")
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    n_after = len(df)
    log(f"  去重前: {n_before} -> 去重后: {n_after}  (移除 {n_dup} 条完全重复)")
    log(f"  策略: 保留首次出现, 去除clean_text完全相同的重复")

    # 去重后情感分布
    log(f"\n  去重后三分类分布:")
    vc = df["sentiment_3label"].value_counts().reindex(["negative","neutral","positive"])
    for lbl, c in vc.items():
        log(f"    {lbl:<10s}: {c:>6d}  ({c/n_after*100:.1f}%)")

    # ================================================================
    # 第6步: 类别不平衡处理
    # ================================================================
    log(f"\n{'='*40}")
    log(f"  第6步: 类别不平衡处理")
    log(f"{'='*40}")

    classes = [0, 1, 2]  # negative, neutral, positive
    labels = ["negative", "neutral", "positive"]
    counts = {c: int((df["sentiment_3class"] == c).sum()) for c in classes}
    total = sum(counts.values())

    log(f"  类别分布:")
    for c, lbl in zip(classes, labels):
        log(f"    {lbl}: {counts[c]}  ({counts[c]/total*100:.1f}%)")
    ratio = max(counts.values()) / min(counts.values())
    log(f"  最大/最小比: {ratio:.2f}")

    # 计算class_weight (balanced策略: n_samples / (n_classes * n_class_samples))
    log(f"\n  class_weight (balanced策略):")
    class_weight = {}
    for c in classes:
        w = total / (len(classes) * counts[c])
        class_weight[c] = round(w, 4)
        log(f"    {labels[c]}: {class_weight[c]}")
    log(f"  说明: neutral样本少 -> 权重高; 训练时传入cross_entropy(weight=...)")

    # 保存权重
    weight_path = os.path.join(BASE, "class_weights.json")
    with open(weight_path, "w", encoding="utf-8") as f:
        json.dump({"class_weight": class_weight,
                   "label_map": {"0":"negative","1":"neutral","2":"positive"},
                   "counts": counts}, f, ensure_ascii=False, indent=2)
    log(f"  权重文件: {weight_path}")

    # ================================================================
    # 第7步: 数据集划分 (分层抽样)
    # ================================================================
    log(f"\n{'='*40}")
    log(f"  第7步: 数据集划分 (分层 {int(RATIO_TRAIN*100)}/{int(RATIO_VAL*100)}/{int(RATIO_TEST*100)})")
    log(f"{'='*40}")

    from sklearn.model_selection import train_test_split

    # 第一刀: train vs (val+test)
    df_train, df_temp = train_test_split(
        df, test_size=(RATIO_VAL + RATIO_TEST),
        random_state=RANDOM_STATE, stratify=df["sentiment_3class"])
    # 第二刀: val vs test
    df_val, df_test = train_test_split(
        df_temp, test_size=(RATIO_TEST / (RATIO_VAL + RATIO_TEST)),
        random_state=RANDOM_STATE, stratify=df_temp["sentiment_3class"])

    log(f"\n  划分结果:")
    log(f"    Train: {len(df_train)}  ({len(df_train)/n_after*100:.1f}%)")
    log(f"    Val:   {len(df_val)}  ({len(df_val)/n_after*100:.1f}%)")
    log(f"    Test:  {len(df_test)}  ({len(df_test)/n_after*100:.1f}%)")

    # 验证分层一致性
    log(f"\n  各集情感分布(验证分层一致性):")
    log(f"    {'集':<8s} {'negative':>10s} {'neutral':>10s} {'positive':>10s}")
    for name, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        v = d["sentiment_3label"].value_counts().reindex(["negative","neutral","positive"])
        log(f"    {name:<8s} {v['negative']:>10d} {v['neutral']:>10d} {v['positive']:>10d}")
        pct = v / len(d) * 100
        log(f"    {'':8s} {pct['negative']:>9.1f}% {pct['neutral']:>9.1f}% {pct['positive']:>9.1f}%")

    # 方面分布一致性
    log(f"\n  各集方面覆盖率(命中至少1方面):")
    for name, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        hit = (d["aspect_count"] > 0).sum()
        log(f"    {name}: {hit}/{len(d)}  ({hit/len(d)*100:.1f}%)")

    # 语言分布一致性
    log(f"\n  各集语言分布:")
    for name, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        v = d["lang"].value_counts()
        log(f"    {name}: en={v.get('en',0)} zh={v.get('zh',0)} unknown={v.get('unknown',0)}")

    # ---- 保存 ----
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        d.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"\n  输出目录: {OUT_DIR}/")
    log(f"    train.csv  ({len(df_train)} 行)")
    log(f"    val.csv    ({len(df_val)} 行)")
    log(f"    test.csv   ({len(df_test)} 行)")

    # ---- 图表: 划分分布对比 ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (name, d) in zip(axes, [("Train", df_train), ("Val", df_val), ("Test", df_test)]):
        v = d["sentiment_3label"].value_counts().reindex(["negative","neutral","positive"])
        ax.bar(v.index, v.values, color=["#e74c3c","#f39c12","#27ae60"], edgecolor="white")
        ax.set_title(f"{name} (N={len(d)})")
        ax.set_ylabel("Count" if name == "Train" else "")
        for i, (lbl, c) in enumerate(v.items()):
            ax.text(i, c + 20, str(c), ha="center", fontsize=8)
    plt.suptitle("Sentiment Distribution Across Splits")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_split_distribution.png"), dpi=200)
    plt.close()
    log(f"  划分图: {FIG_DIR}/08_split_distribution.png")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n  划分报告: {REPORT}")
    log(f"\n{'='*64}")
    log(f"  第5-7步(去重+不平衡+划分)完成 ✓")
    log(f"{'='*64}")


if __name__ == "__main__":
    main()
