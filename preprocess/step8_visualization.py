# -*- coding: utf-8 -*-
"""
============================================================
 第8步: 数据统计可视化 (论文"数据描述"章节用图)
 输入: absa_labeled.csv + absa_long.csv
 输出: figures/ 目录下全套图表 + visualization_report.txt
------------------------------------------------------------
 生成图表:
   09  文本长度分布直方图(中英文对比)
   10  词云-英文(正面 vs 负面)
   11  词云-中文
   12  App来源分布(横向柱状图)
   13  方面热力图(方面 x 情感占比)
   14  评分 vs 方面命中数箱线图
   15  各方面负向率排序图(核心发现)
============================================================
"""
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

BASE     = r"D:\codex\nlp_thesis"
IN_CSV   = os.path.join(BASE, "absa_labeled.csv")
LONG_CSV = os.path.join(BASE, "absa_long.csv")
FIG_DIR  = os.path.join(BASE, "figures")
REPORT   = os.path.join(BASE, "visualization_report.txt")
os.makedirs(FIG_DIR, exist_ok=True)

# 字体
for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        ZH_FONT = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# 中文字体路径(wordcloud需要)
ZH_FONT_PATH = None
for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
          r"C:\Windows\Fonts\msyh.ttf"]:
    if os.path.exists(p):
        ZH_FONT_PATH = p
        break

ASPECT_NAMES_CN = {
    "A1_UI_Design": "UI Design", "A2_Functionality": "Functionality",
    "A3_Cost_Subscription": "Cost/Sub", "A4_Privacy_Security": "Privacy",
    "A5_Performance": "Performance", "A6_Customer_Support": "Support",
    "A7_Content_Quality": "Content",
}

EN_STOP = set("the a an and or but if of to in on at for with by is are was were "
              "be been being this that these those it its i my we our you your he "
              "she they them his her their as so not no do does did done have has "
              "had will would can could should shall may might must from into out "
              "up down over under again further then once here there all any both "
              "each few more most other some such only own same than too very just "
              "don t s m re ve ll d am about what when which who whom whose how why "
              "been being doing having".split())
CN_STOP = set("的 了 是 在 我 有 和 就 不 人 都 一 上 也 很 到 说 要 去 你 会 "
              "着 没有 看 好 自己 这 那 他 她 它 我们 你们 他们 这个 那个 什么 "
              "可以 因为 所以 如果 虽然 但是 而且 或者 还是 已经 一直 一些 一下 "
              "没 不 很 非常 太 真 就 才 还 再 又 只".split())


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(str(s))

    df = pd.read_csv(IN_CSV)
    df_long = pd.read_csv(LONG_CSV)
    log("=" * 64)
    log("  第8步: 数据统计可视化 报告")
    log("=" * 64)
    log(f"\n【输入】{IN_CSV} ({len(df)} 行) + {LONG_CSV} ({len(df_long)} 行)")

    # ---- 09 文本长度分布 ----
    log(f"\n[09] 文本长度分布直方图")
    df["text_len"] = df["clean_text"].str.len()
    df_en = df[df["lang"] == "en"]
    df_zh = df[df["lang"] == "zh"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, d, color) in zip(axes, [("English", df_en, "#3498db"),
                                            ("Chinese", df_zh, "#e74c3c")]):
        ax.hist(d["text_len"].clip(upper=500), bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(d["text_len"].median(), color="black", linestyle="--", label=f"median={d['text_len'].median():.0f}")
        ax.set_title(f"{name} Review Length (N={len(d)})")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_text_length_dist.png"), dpi=200)
    plt.close()
    log(f"  -> 09_text_length_dist.png")

    # ---- 10 英文词云 (正面 vs 负面) ----
    log(f"\n[10] 英文词云(正面 vs 负面)")
    for sent, color, fname in [("positive", "#27ae60", "10_wordcloud_en_pos"),
                                ("negative", "#e74c3c", "10_wordcloud_en_neg")]:
        sub = df_en[df_en["sentiment_3label"] == sent]
        words = []
        for toks in sub["tokens"].fillna(""):
            words += [t for t in str(toks).split()
                      if t.lower() not in EN_STOP and len(t) > 2 and t.isalpha()]
        freq = Counter(words)
        if freq:
            wc = WordCloud(width=800, height=400, background_color="white",
                           max_words=100, colormap="Greens" if sent == "positive" else "Reds",
                           collocations=False).generate_from_frequencies(freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"English {sent.capitalize()} Reviews Word Cloud (N={len(sub)})")
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"{fname}.png"), dpi=200)
            plt.close()
            log(f"  -> {fname}.png  (top: {', '.join(w for w,_ in freq.most_common(10))})")

    # ---- 11 中文词云 ----
    log(f"\n[11] 中文词云")
    zh_words = []
    for toks in df_zh["tokens"].fillna(""):
        zh_words += [t for t in str(toks).split()
                     if t not in CN_STOP and len(t.strip()) >= 2
                     and not re.fullmatch(r"[^\u4e00-\u9fa5]+", t)]
    freq_zh = Counter(zh_words)
    if freq_zh and ZH_FONT_PATH:
        wc = WordCloud(width=800, height=400, background_color="white",
                       max_words=80, font_path=ZH_FONT_PATH,
                       colormap="Blues", collocations=False).generate_from_frequencies(freq_zh)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Chinese Reviews Word Cloud (N={len(df_zh)})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "11_wordcloud_zh.png"), dpi=200)
        plt.close()
        log(f"  -> 11_wordcloud_zh.png  (top: {', '.join(w for w,_ in freq_zh.most_common(10))})")
    else:
        log(f"  中文词云跳过(字体或词频为空)")

    # ---- 12 App来源分布 ----
    log(f"\n[12] App来源分布")
    fig, ax = plt.subplots(figsize=(9, 6))
    app_vc = df.groupby(["platform", "app_name"]).size().sort_values()
    colors = ["#3498db" if "Google" in str(idx) or "Apple" == str(idx)[0]
              else "#e74c3c" for idx in app_vc.index]
    # 按平台着色
    bar_colors = []
    for (plat, _) in app_vc.index:
        bar_colors.append("#3498db" if plat == "Google Play" else "#e67e22")
    app_labels = [f"{app[:30]}" for (_, app) in app_vc.index]
    ax.barh(range(len(app_vc)), app_vc.values, color=bar_colors, edgecolor="white")
    ax.set_yticks(range(len(app_vc)))
    ax.set_yticklabels(app_labels, fontsize=7)
    ax.set_xlabel("Review Count")
    ax.set_title("Review Count by App and Platform")
    # 图例
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#3498db", label="Google Play"),
                       Patch(color="#e67e22", label="Apple App Store")],
              loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "12_app_distribution.png"), dpi=200)
    plt.close()
    log(f"  -> 12_app_distribution.png")

    # ---- 13 方面 x 情感 热力图(占比) ----
    log(f"\n[13] 方面情感热力图")
    ct = pd.crosstab(df_long["aspect"], df_long["aspect_sentiment"])
    ct = ct.reindex(columns=["negative","neutral","positive"], fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.reindex(list(ASPECT_NAMES_CN.keys()))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ct_pct.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Negative","Neutral","Positive"])
    ax.set_yticks(range(len(ASPECT_NAMES_CN)))
    ax.set_yticklabels([f"{a}\n{ASPECT_NAMES_CN[a]}" for a in ASPECT_NAMES_CN], fontsize=8)
    for i in range(len(ASPECT_NAMES_CN)):
        for j in range(3):
            ax.text(j, i, f"{ct_pct.values[i,j]:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if ct_pct.values[i,j] > 60 else "black")
    ax.set_title("Aspect-Level Sentiment Heatmap (%)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "13_aspect_sentiment_heatmap.png"), dpi=200)
    plt.close()
    log(f"  -> 13_aspect_sentiment_heatmap.png")

    # ---- 14 评分 vs 方面命中数 ----
    log(f"\n[14] 评分 vs 方面命中数")
    fig, ax = plt.subplots(figsize=(8, 5))
    ratings = sorted(df["rating"].unique())
    data_box = [df[df["rating"] == r]["aspect_count"].values for r in ratings]
    bp = ax.boxplot(data_box, labels=[f"{r}★" for r in ratings], patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="yellow",
                                                   markersize=5))
    colors_r = ["#e74c3c","#e67e22","#f39c12","#2ecc71","#27ae60"]
    for patch, c in zip(bp["boxes"], colors_r):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Number of Aspects Detected")
    ax.set_xlabel("Rating")
    ax.set_title("Aspect Count Distribution by Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "14_rating_vs_aspects.png"), dpi=200)
    plt.close()
    log(f"  -> 14_rating_vs_aspects.png")

    # ---- 15 各方面负向率排序(核心发现图) ----
    log(f"\n[15] 各方面负向率排序(核心发现)")
    neg_rates = {}
    for a in ASPECT_NAMES_CN:
        sub = df_long[df_long["aspect"] == a]
        neg_rates[a] = (sub["aspect_sentiment"] == "negative").sum() / len(sub) * 100
    sorted_a = sorted(neg_rates.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{a}\n{ASPECT_NAMES_CN[a]}" for a, _ in sorted_a]
    vals = [v for _, v in sorted_a]
    bars = ax.barh(range(len(sorted_a)), vals, color="#e74c3c", edgecolor="white")
    ax.set_yticks(range(len(sorted_a)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Negative Sentiment Rate (%)")
    ax.set_title("Negative Sentiment Rate by Aspect (Sorted)")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="50% line")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "15_aspect_neg_rate.png"), dpi=200)
    plt.close()
    log(f"  -> 15_aspect_neg_rate.png")

    # ---- 汇总 ----
    n_figs = len([f for f in os.listdir(FIG_DIR) if f.endswith(".png")])
    log(f"\n【输出】{FIG_DIR}/  (共 {n_figs} 张图表)")
    log(f"  本次新增: 09-15 (7张)")
    log(f"  此前已生成: 01-08 (8张)")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n  可视化报告: {REPORT}")
    log(f"\n{'='*64}")
    log(f"  第8步(数据统计可视化)完成 ✓")
    log(f"{'='*64}")


if __name__ == "__main__":
    main()
