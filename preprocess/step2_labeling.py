# -*- coding: utf-8 -*-
"""
============================================================
 第2步: 情感标签构建
 输入: cleaned_reviews.csv (19052 行)
 输出: labeled_reviews.csv + label_report.txt + 分布图表
------------------------------------------------------------
 标签方案:
   三分类 sentiment_3class: 1-2星->negative(0), 3星->neutral(1), 4-5星->positive(2)
   二分类 sentiment_2class: 1-2星->negative(0), 4-5星->positive(1), 3星丢弃
 论文采用三分类(保留中评, 更贴近真实分布), 二分类作对比实验备用
============================================================
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE    = r"D:\codex\nlp_thesis"
IN_CSV  = os.path.join(BASE, "cleaned_reviews.csv")
OUT_CSV = os.path.join(BASE, "labeled_reviews.csv")
REPORT  = os.path.join(BASE, "label_report.txt")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def build_labels(df):
    """构建三分类与二分类标签."""
    # 三分类: 1-2->negative(0), 3->neutral(1), 4-5->positive(2)
    def to_3class(r):
        if r <= 2: return 0
        if r == 3: return 1
        return 2
    df["sentiment_3class"] = df["rating"].apply(to_3class)
    df["sentiment_3label"] = df["sentiment_3class"].map(
        {0: "negative", 1: "neutral", 2: "positive"})

    # 二分类: 1-2->negative(0), 4-5->positive(1), 3->NaN(丢弃)
    def to_2class(r):
        if r <= 2: return 0
        if r >= 4: return 1
        return None
    df["sentiment_2class"] = df["rating"].apply(to_2class)
    df["sentiment_2label"] = df["sentiment_2class"].map(
        {0: "negative", 1: "positive"})
    return df


def plot_distributions(df):
    """生成论文用分布图."""
    colors_3 = {"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#27ae60"}

    # 图1: 评分分布(1-5星)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vc = df["rating"].value_counts().sort_index()
    bars = ax.bar(vc.index, vc.values, color=["#e74c3c","#e67e22","#f39c12","#2ecc71","#27ae60"], edgecolor="white")
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x()+b.get_width()/2, v+50, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Rating (star)")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution of Health App Reviews (N=19052)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_rating_distribution.png"), dpi=200)
    plt.close()

    # 图2: 三分类情感分布
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vc3 = df["sentiment_3label"].value_counts().reindex(["negative","neutral","positive"])
    bars = ax.bar(vc3.index, vc3.values, color=[colors_3[k] for k in vc3.index], edgecolor="white")
    for b, v in zip(bars, vc3.values):
        pct = v / len(df) * 100
        ax.text(b.get_x()+b.get_width()/2, v+50, f"{v}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution (3-class)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_sentiment_3class.png"), dpi=200)
    plt.close()

    # 图3: 情感 x 语言 交叉堆叠图
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cross = pd.crosstab(df["lang"], df["sentiment_3label"])
    cross = cross.reindex(columns=["negative","neutral","positive"], fill_value=0)
    cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).index]
    cross.plot(kind="barh", stacked=True, ax=ax,
               color=[colors_3[k] for k in cross.columns], edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_ylabel("Language")
    ax.set_title("Sentiment by Language")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_sentiment_by_lang.png"), dpi=200)
    plt.close()

    # 图4: 各App情感分布(横向堆叠, Top 12)
    fig, ax = plt.subplots(figsize=(8, 6))
    app_cross = pd.crosstab(df["app_name"], df["sentiment_3label"])
    app_cross = app_cross.reindex(columns=["negative","neutral","positive"], fill_value=0)
    app_cross = app_cross.loc[app_cross.sum(axis=1).sort_values().tail(12).index]
    app_cross.plot(kind="barh", stacked=True, ax=ax,
                   color=[colors_3[k] for k in app_cross.columns], edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_ylabel("App")
    ax.set_title("Sentiment Distribution by App (Top 12)")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_sentiment_by_app.png"), dpi=200)
    plt.close()


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(str(s))

    df = pd.read_csv(IN_CSV)
    log("=" * 64)
    log("  第2步: 情感标签构建 报告")
    log("=" * 64)

    log(f"\n【输入】{IN_CSV}  ({len(df)} 行)")

    # ---- 标签构建 ----
    df = build_labels(df)
    log(f"\n【1. 标签映射方案】")
    log(f"  三分类 sentiment_3class:")
    log(f"    rating 1-2 -> negative(0)")
    log(f"    rating 3   -> neutral(1)")
    log(f"    rating 4-5 -> positive(2)")
    log(f"  二分类 sentiment_2class (对比实验备用):")
    log(f"    rating 1-2 -> negative(0)")
    log(f"    rating 4-5 -> positive(1)")
    log(f"    rating 3   -> 丢弃")

    # ---- 三分类分布 ----
    log(f"\n【2. 三分类分布】")
    vc3 = df["sentiment_3label"].value_counts().reindex(["negative","neutral","positive"])
    for lbl, c in vc3.items():
        log(f"    {lbl:<10s}: {c:>6d}  ({c/len(df)*100:.1f}%)")
    log(f"    合计: {vc3.sum()}")

    # ---- 二分类分布 ----
    log(f"\n【3. 二分类分布(丢弃3星后)】")
    df_2 = df.dropna(subset=["sentiment_2class"])
    vc2 = df_2["sentiment_2label"].value_counts().reindex(["negative","positive"])
    for lbl, c in vc2.items():
        log(f"    {lbl:<10s}: {c:>6d}  ({c/len(df_2)*100:.1f}%)")
    log(f"    丢弃3星: {len(df)-len(df_2)} 条")
    log(f"    合计: {vc2.sum()}")

    # ---- 不平衡度量 ----
    log(f"\n【4. 类别不平衡分析】")
    ratio3 = vc3.max() / vc3.min()
    log(f"  三分类 最大/最小比: {ratio3:.2f}  "
        f"({vc3.idxmax()}={vc3.max()} vs {vc3.idxmin()}={vc3.min()})")
    ratio2 = vc2.max() / vc2.min()
    log(f"  二分类 最大/最小比: {ratio2:.2f}  "
        f"({vc2.idxmax()}={vc2.max()} vs {vc2.idxmin()}={vc2.min()})")
    log(f"  建议: 三分类不平衡更严重(neutral极少), 训练时用class_weight或focal loss")

    # ---- 情感 x 语言 ----
    log(f"\n【5. 情感 x 语言 交叉表(三分类)】")
    ct = pd.crosstab(df["lang"], df["sentiment_3label"], margins=True)
    ct = ct.reindex(columns=["negative","neutral","positive","All"], fill_value=0)
    log(ct.to_string())

    # ---- 情感 x 平台 ----
    log(f"\n【6. 情感 x 平台 交叉表(三分类)】")
    ct2 = pd.crosstab(df["platform"], df["sentiment_3label"], margins=True)
    ct2 = ct2.reindex(columns=["negative","neutral","positive","All"], fill_value=0)
    log(ct2.to_string())

    # ---- 各App情感分布 ----
    log(f"\n【7. 各App情感分布(Top 15)】")
    app_ct = pd.crosstab(df["app_name"], df["sentiment_3label"])
    app_ct = app_ct.reindex(columns=["negative","neutral","positive"], fill_value=0)
    app_ct["total"] = app_ct.sum(axis=1)
    app_ct = app_ct.sort_values("total", ascending=False)
    log(app_ct.head(15).to_string())

    # ---- 保存 ----
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"\n【8. 输出】")
    log(f"  带标签数据: {OUT_CSV}  ({len(df)} 行)")

    # ---- 图表 ----
    plot_distributions(df)
    log(f"  分布图表: {FIG_DIR}/  (4张PNG)")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  标签报告: {REPORT}")
    log(f"\n{'='*64}")
    log(f"  第2步(情感标签构建)完成 ✓")
    log(f"{'='*64}")


if __name__ == "__main__":
    main()
