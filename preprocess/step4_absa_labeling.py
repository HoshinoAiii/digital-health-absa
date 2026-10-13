# -*- coding: utf-8 -*-
"""
============================================================
 第4步: 方面级情感标注 (Aspect-Based Sentiment Labeling)
 输入: aspect_extracted.csv (含 aspects / aspect_detail)
 输出: absa_labeled.csv + absa_report.txt + 方面情感分布图
------------------------------------------------------------
 对每条评论中命中的每个方面, 判定其情感极性:
   positive(2) / neutral(1) / negative(0)
 标注方法: 方面级情感词典 + 否定词窗口 + 程度副词修正
   1. 定位方面关键词在文本中的位置
   2. 取该位置前后 window=5 个token的窗口
   3. 窗口内情感词得分 + 否定词反转 + 程度副词加权
   4. 得分>0 -> positive, <0 -> negative, ==0 -> neutral
   5. 无情感词时回退到评论整体情感标签(文档级)
 附加: 标注置信度(high=窗口内有情感词, low=回退到文档级)
============================================================
"""
import os
import re
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE    = r"D:\codex\nlp_thesis"
IN_CSV  = os.path.join(BASE, "aspect_extracted.csv")
OUT_CSV = os.path.join(BASE, "absa_labeled.csv")
REPORT  = os.path.join(BASE, "absa_report.txt")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

for f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ==================================================================
# 情感词典 (中英文, 方面级)
# ==================================================================
POS_WORDS = {
    "en": ["good", "great", "excellent", "amazing", "awesome", "love", "loved",
           "like", "liked", "best", "perfect", "wonderful", "fantastic", "helpful",
           "useful", "easy", "nice", "beautiful", "clean", "smooth", "fast",
           "reliable", "accurate", "recommend", "recommended", "satisfied",
           "impressed", "happy", "glad", "enjoy", "enjoyed", "worth", "value",
           "convenient", "intuitive", "simple", "effective", "works", "working",
           "cool", "fun", "brilliant", "superb", "outstanding", "flawless",
           "comfortable", "pleased", "delighted", "grateful", "thankful"],
    "zh": ["好", "好用", "很好", "不错", "棒", "赞", "喜欢", "满意", "方便",
           "简单", "容易", "实用", "有用", "有效", "准确", "推荐", "优秀",
           "出色", "完美", "精彩", "舒适", "流畅", "快速", "稳定", "值得",
           "感谢", "开心", "喜欢", "美观", "漂亮", "好看", "贴心", "省心",
           "专业", "靠谱", "良心", "给力", "牛", "强大", "清晰"]
}
NEG_WORDS = {
    "en": ["bad", "terrible", "awful", "horrible", "hate", "hated", "worst",
           "useless", "broken", "crash", "crashes", "crashed", "freeze",
           "freezes", "frozen", "bug", "buggy", "slow", "laggy", "glitch",
           "error", "fail", "fails", "failed", "poor", "disappointing",
           "frustrated", "frustrating", "annoying", "annoyed", "difficult",
           "hard", "confusing", "complicated", "expensive", "overpriced",
           "pricey", "unfair", "scam", "fraud", "fake", "wrong", "incorrect",
           "unreliable", "unstable", "outdated", "old", "waste", "worthless",
           "sucks", "suck", "garbage", "trash", "rubbish", "junk", "mess",
           "ripoff", "rip off", "greedy", "deceptive", "misleading", "beware"],
    "zh": ["差", "差劲", "垃圾", "糟", "糟糕", "烂", "讨厌", "不喜欢",
           "难用", "难看", "麻烦", "复杂", "混乱", "卡", "卡顿", "崩溃",
           "闪退", "死机", "慢", "延迟", "错误", "故障", "bug", "失效",
           "无法", "不能", "不行", "不好", "没用", "无效", "不准确",
           "骗", "骗子", "骗钱", "坑", "坑爹", "套路", "虚假", "误导",
           "贵", "太贵", "不值", "不划算", "浪费", "恶心", "烦", "烦人",
           "广告", "弹窗", "退款", "侵权", "泄露", "危险", "不稳定",
           "过时", "陈旧", "失望", "后悔", "吐槽", "投诉"]
}
# 否定词
NEGATORS = {
    "en": ["not", "no", "never", "n't", "none", "neither", "nor", "without",
           "cannot", "can't", "don't", "doesn't", "didn't", "won't", "isn't",
           "wasn't", "aren't", "weren't", "haven't", "hasn't", "hadn't"],
    "zh": ["不", "没", "没有", "别", "莫", "无", "非", "未", "勿", "毫不"]
}
# 程度副词(增强/减弱)
INTENSIFIERS = {
    "en": {"very": 1.5, "really": 1.5, "so": 1.3, "extremely": 2.0,
           "too": 1.3, "quite": 1.2, "incredibly": 1.8, "super": 1.5,
           "highly": 1.5, "absolutely": 1.8, "completely": 1.5, "totally": 1.5},
    "zh": {"很": 1.5, "非常": 1.5, "太": 1.5, "特别": 1.5, "极其": 2.0,
           "超级": 1.5, "十分": 1.5, "相当": 1.3, "格外": 1.5, "真心": 1.5,
           "真的": 1.3, "比较": 1.2, "稍微": 0.7, "有点": 0.8, "略微": 0.7}
}

ASPECT_NAMES_CN = {
    "A1_UI_Design": "界面设计", "A2_Functionality": "功能实用性",
    "A3_Cost_Subscription": "费用订阅", "A4_Privacy_Security": "隐私安全",
    "A5_Performance": "性能稳定", "A6_Customer_Support": "客服支持",
    "A7_Content_Quality": "内容质量",
}

# 方面关键词(从第3步词典中取核心词, 用于定位)
ASPECT_KEYWORDS = {
    "A1_UI_Design": ["ui","interface","design","layout","screen","navigation",
                     "menu","button","icon","界面","布局","美观","设计","操作","交互"],
    "A2_Functionality": ["feature","function","tool","tracker","tracking","remind",
                         "reminder","notification","scan","search","sync","monitor",
                         "功能","提醒","通知","搜索","同步","监测","记录"],
    "A3_Cost_Subscription": ["price","cost","expensive","cheap","fee","subscription",
                             "subscribe","pay","paid","free","premium","charge",
                             "refund","worth","membership","member","价格","费用",
                             "收费","订阅","付费","免费","会员","扣费","退款"],
    "A4_Privacy_Security": ["privacy","data","permission","security","secure",
                            "track","consent","隐私","数据","权限","安全","跟踪"],
    "A5_Performance": ["crash","freeze","lag","slow","bug","glitch","error",
                       "broken","performance","stable","崩溃","闪退","卡顿","慢",
                       "bug","错误","稳定"],
    "A6_Customer_Support": ["support","customer service","help","response","reply",
                            "contact","feedback","fix","resolve","客服","服务",
                            "帮助","回复","反馈","解决"],
    "A7_Content_Quality": ["content","information","article","guide","resource",
                           "course","audio","sound","quality","accurate","reliable",
                           "informative","内容","信息","课程","音频","声音","质量",
                           "准确","专业"],
}

WINDOW = 5  # 方面前后窗口大小


def tokenize_simple(text, lang):
    """简单分词(用于窗口定位)."""
    if lang == "zh":
        # 中文按字符+标点分割
        return list(text)
    else:
        return text.split()


def find_keyword_positions(tokens, keywords, lang):
    """在tokens中找关键词位置."""
    positions = []
    if lang == "zh":
        text = "".join(tokens)
        for kw in keywords:
            idx = text.find(kw)
            while idx != -1:
                positions.append(idx)
                idx = text.find(kw, idx + 1)
    else:
        tokens_lower = [t.lower().strip(".,!?;:'\"()") for t in tokens]
        for kw in keywords:
            kw_l = kw.lower()
            for i, t in enumerate(tokens_lower):
                if kw_l in t and t:
                    positions.append(i)
    return positions


def score_window(tokens, positions, lang):
    """对方面关键词窗口内打分."""
    pos_set = set(POS_WORDS.get(lang, []) + POS_WORDS.get("en", []))
    neg_set = set(NEG_WORDS.get(lang, []) + NEG_WORDS.get("en", []))
    negators = set(NEGATORS.get(lang, []) + NEGATORS.get("en", []))
    intensifiers = INTENSIFIERS.get(lang, {})

    if lang == "zh":
        tokens_lower = tokens  # 中文不分大小写
        check_fn = lambda t, w: w in t
    else:
        tokens_lower = [t.lower().strip(".,!?;:'\"()") for t in tokens]
        check_fn = lambda t, w: w == t or w in t

    score = 0
    n_hits = 0
    for pos in positions:
        start = max(0, pos - WINDOW)
        end = min(len(tokens_lower), pos + WINDOW + 1)
        for i in range(start, end):
            t = tokens_lower[i]
            if not t:
                continue
            # 检查程度副词
            intens = 1.0
            for iw, iv in intensifiers.items():
                if check_fn(t, iw):
                    intens = iv
                    break
            # 检查否定词(前2个token)
            neg = False
            for j in range(max(0, i - 2), i):
                for neg_w in negators:
                    if check_fn(tokens_lower[j], neg_w):
                        neg = True
                        break
            # 情感词
            for w in pos_set:
                if check_fn(t, w):
                    score += intens * (-1 if neg else 1)
                    n_hits += 1
                    break
            else:
                for w in neg_set:
                    if check_fn(t, w):
                        score -= intens * (-1 if neg else 1)
                        n_hits += 1
                        break
    return score, n_hits


def label_aspect_sentiment(text, aspect, lang, doc_sentiment):
    """对单个方面做情感标注. 返回 (label, confidence)."""
    if not aspect:
        return None, "none"
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    tokens = tokenize_simple(text, lang)
    positions = find_keyword_positions(tokens, keywords, lang)

    if positions:
        score, n_hits = score_window(tokens, positions, lang)
        if n_hits > 0:
            if score > 0:
                return "positive", "high"
            elif score < 0:
                return "negative", "high"
            else:
                return "neutral", "high"
    # 回退到文档级情感
    return doc_sentiment, "low"


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(str(s))

    df = pd.read_csv(IN_CSV)
    log("=" * 64)
    log("  第4步: 方面级情感标注 报告")
    log("=" * 64)
    log(f"\n【输入】{IN_CSV}  ({len(df)} 行)")

    # ---- 方面级标注 ----
    log(f"\n【1. 标注方法】")
    log(f"  方面关键词窗口(±{WINDOW}token) + 情感词典 + 否定词反转 + 程度副词加权")
    log(f"  无窗口情感词时回退到文档级情感标签(置信度=low)")

    absa_records = []  # 方面级长表
    aspect_cols = {}   # 宽表列

    for idx, row in df.iterrows():
        asps = [a.strip() for a in str(row["aspects"]).split(",") if a.strip()]
        lang = row["lang"] if row["lang"] in ("en", "zh") else "en"
        doc_sent = row["sentiment_3label"]
        row_labels = {}
        for a in asps:
            lbl, conf = label_aspect_sentiment(
                row["clean_text"], a, lang, doc_sent)
            if lbl:
                absa_records.append({
                    "review_id": row["review_id"],
                    "platform": row["platform"],
                    "app_name": row["app_name"],
                    "rating": row["rating"],
                    "lang": lang,
                    "aspect": a,
                    "aspect_sentiment": lbl,
                    "confidence": conf,
                    "doc_sentiment": doc_sent,
                    "clean_text": row["clean_text"][:200],
                })
                row_labels[a] = lbl
        # 宽表: 每个方面一列
        for a in ASPECT_KEYWORDS:
            aspect_cols.setdefault(a, []).append(row_labels.get(a, ""))

    df_absa = pd.DataFrame(absa_records)
    # 宽表列加入原df
    for a in ASPECT_KEYWORDS:
        df[f"{a}_sentiment"] = aspect_cols[a]

    # ---- 统计 ----
    log(f"\n【2. 方面级标注统计】")
    log(f"  方面-情感记录总数: {len(df_absa)}")

    # 置信度分布
    log(f"\n【3. 标注置信度分布】")
    conf_vc = df_absa["confidence"].value_counts()
    for c, v in conf_vc.items():
        log(f"    {c}: {v}  ({v/len(df_absa)*100:.1f}%)")

    # 方面 x 方面级情感
    log(f"\n【4. 方面 x 方面级情感 交叉表】")
    ct = pd.crosstab(df_absa["aspect"], df_absa["aspect_sentiment"], margins=True)
    ct = ct.reindex(index=list(ASPECT_KEYWORDS.keys()) + ["All"],
                    columns=["negative","neutral","positive","All"], fill_value=0)
    log(ct.to_string())

    # 各方面情感分布占比
    log(f"\n【5. 各方面情感占比(关键发现)】")
    log(f"  {'方面':<22s} {'负向率':>7s} {'中性率':>7s} {'正向率':>7s}")
    for a in ASPECT_KEYWORDS:
        sub = df_absa[df_absa["aspect"] == a]
        n = len(sub)
        if n == 0:
            continue
        neg = (sub["aspect_sentiment"] == "negative").sum() / n * 100
        neu = (sub["aspect_sentiment"] == "neutral").sum() / n * 100
        pos = (sub["aspect_sentiment"] == "positive").sum() / n * 100
        log(f"  {a:<22s} {neg:>6.1f}% {neu:>6.1f}% {pos:>6.1f}%")

    # 方面级 vs 文档级一致性
    log(f"\n【6. 方面级 vs 文档级情感一致性】")
    agree = (df_absa["aspect_sentiment"] == df_absa["doc_sentiment"]).sum()
    log(f"  一致: {agree} / {len(df_absa)}  ({agree/len(df_absa)*100:.1f}%)")
    log(f"  不一致: {len(df_absa)-agree}  ({(len(df_absa)-agree)/len(df_absa)*100:.1f}%)")
    # 高置信度子集的一致性
    high = df_absa[df_absa["confidence"] == "high"]
    if len(high) > 0:
        agree_h = (high["aspect_sentiment"] == high["doc_sentiment"]).sum()
        log(f"  高置信度子集一致: {agree_h} / {len(high)}  "
            f"({agree_h/len(high)*100:.1f}%)")

    # 各App的方面情感
    log(f"\n【7. 各App各方面负向率(Top 10 App)】")
    top_apps = df_absa["app_name"].value_counts().head(10).index
    for app in top_apps:
        sub = df_absa[df_absa["app_name"] == app]
        neg_rate = (sub["aspect_sentiment"] == "negative").sum() / len(sub) * 100
        log(f"  {app[:35]:<37s} 负向率={neg_rate:.1f}%  (n={len(sub)})")

    # 标注示例
    log(f"\n【8. 标注示例(随机10条)】")
    for _, r in df_absa.sample(min(10, len(df_absa)), random_state=42).iterrows():
        log(f"  [{r['aspect']}/{r['aspect_sentiment']}/{r['confidence']}] "
            f"[{r['lang']}/{r['rating']}星] {r['clean_text'][:80]}")

    # ---- 保存 ----
    # 长表(方面级)
    absa_long_path = os.path.join(BASE, "absa_long.csv")
    df_absa.to_csv(absa_long_path, index=False, encoding="utf-8-sig")
    # 宽表(文档级+方面级)
    out_cols = ["review_id","platform","app_name","rating","review_date",
                "raw_text","clean_text","lang","tokens","token_count",
                "sentiment_3class","sentiment_3label","sentiment_2class",
                "sentiment_2label","aspects","aspect_count"] + \
               [f"{a}_sentiment" for a in ASPECT_KEYWORDS]
    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"\n【9. 输出】")
    log(f"  方面级长表: {absa_long_path}  ({len(df_absa)} 行)")
    log(f"  方面级宽表: {OUT_CSV}  ({len(df)} 行)")

    # ---- 图表 ----
    # 方面级情感堆叠图
    fig, ax = plt.subplots(figsize=(9, 5))
    ct_plot = pd.crosstab(df_absa["aspect"], df_absa["aspect_sentiment"])
    ct_plot = ct_plot.reindex(index=list(ASPECT_KEYWORDS.keys()),
                              columns=["negative","neutral","positive"], fill_value=0)
    ct_plot_pct = ct_plot.div(ct_plot.sum(axis=1), axis=0) * 100
    labels = [f"{a}\n{ASPECT_NAMES_CN[a]}" for a in ASPECT_KEYWORDS]
    ct_plot_pct.plot(kind="barh", stacked=True, ax=ax,
                     color=["#e74c3c","#f39c12","#27ae60"], edgecolor="white")
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Aspect-Level Sentiment Distribution (%)")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_absa_sentiment_pct.png"), dpi=200)
    plt.close()
    log(f"  方面情感图: {FIG_DIR}/07_absa_sentiment_pct.png")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  标注报告: {REPORT}")
    log(f"\n{'='*64}")
    log(f"  第4步(方面级情感标注)完成 ✓")
    log(f"{'='*64}")


if __name__ == "__main__":
    main()
