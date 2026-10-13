# -*- coding: utf-8 -*-
"""
============================================================
 第3步: 方面定义与抽取 (ABSA核心)
 输入: labeled_reviews.csv (19052 行, 含情感标签)
 输出: aspect_extracted.csv + aspect_report.txt + 方面分布图
------------------------------------------------------------
 方面体系(7类, 基于健康App用户体验):
   A1 UI_Design        界面设计: 布局/美观/导航/易用性
   A2 Functionality    功能实用性: 功能丰富度/实用性/准确性
   A3 Cost_Subscription 费用订阅: 价格/订阅/付费/会员/内购
   A4 Privacy_Security  隐私安全: 隐私/数据/权限/安全
   A5 Performance       性能稳定: 崩溃/卡顿/闪退/速度/bug
   A6 Customer_Support  客服支持: 客服/回复/帮助/反馈
   A7 Content_Quality   内容质量: 信息/内容/课程/音频/数据准确
 抽取方法: 基于方面关键词词典的多标签匹配
============================================================
"""
import os
import re
import ast
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE    = r"D:\codex\nlp_thesis"
IN_CSV  = os.path.join(BASE, "labeled_reviews.csv")
OUT_CSV = os.path.join(BASE, "aspect_extracted.csv")
REPORT  = os.path.join(BASE, "aspect_report.txt")
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
# 方面关键词词典 (中英文)
# ==================================================================
ASPECT_DICT = {
    "A1_UI_Design": {
        "en": ["ui", "interface", "design", "layout", "look", "visual",
               "screen", "navigation", "navigate", "menu", "button", "icon",
               "color", "theme", "dark mode", "clean", "cluttered", "ugly",
               "beautiful", "intuitive", "user friendly", "user-friendly",
               "easy to use", "easy to navigate", "aesthetically"],
        "zh": ["界面", "ui", "布局", "美观", "好看", "设计", "排版", "导航",
               "菜单", "按钮", "图标", "颜色", "主题", "简洁", "简洁", "干净",
               "难看", "丑", "直观", "友好", "易用", "操作", "交互", "视觉"]
    },
    "A2_Functionality": {
        "en": ["feature", "function", "tool", "works", "work", "useful",
               "practical", "tracker", "tracking", "remind", "reminder",
               "alarm", "notification", "scan", "search", "sync", "syncing",
               "log", "record", "monitor", "measure", "detect", "calculate",
               "calorie", "step", "heart rate", "sleep track", "meditation",
               "workout", "exercise", "diet", "pill", "medication", "symptom",
               "diagnos", "prescription", "appointment", "doctor", "consult",
               "therapy", "therapist", "counseling"],
        "zh": ["功能", "作用", "实用", "好用", "提醒", "闹钟", "通知", "推送",
               "扫描", "搜索", "同步", "记录", "追踪", "跟踪", "监测", "检测",
               "测量", "计算", "卡路里", "步数", "心率", "睡眠", "冥想",
               "运动", "健身", "饮食", "吃药", "服药", "用药", "症状",
               "诊断", "处方", "预约", "挂号", "医生", "问诊", "咨询",
               "咨询", "治疗", "疗法", "课程", "训练", "计划"]
    },
    "A3_Cost_Subscription": {
        "en": ["price", "pricing", "cost", "costly", "expensive", "cheap",
               "fee", "fees", "subscription", "subscribe", "pay", "payment",
               "paid", "free", "freemium", "premium", "charge", "charged",
               "billing", "bill", "purchase", "buy", "money", "refund",
               "worth", "overpriced", "afford", "affordable", "trial",
               "membership", "member", "in-app", "in app", "plan", "upgrade"],
        "zh": ["价格", "价钱", "贵", "便宜", "费用", "收费", "订阅", "续订",
               "付费", "充值", "免费", "会员", "会员费", "扣费", "退款",
               "退货", "值得", "不值", "性价比", "试用", "套餐", "升级",
               "内购", "花钱", "钱", "白嫖"]
    },
    "A4_Privacy_Security": {
        "en": ["privacy", "private", "data", "personal info", "personal data",
               "permission", "permissions", "security", "secure", "tracking",
               "track me", "spy", "surveillance", "consent", "gdpr",
               "confidential", "hack", "breach", "leak", "collect data",
               "sell data", "third party", "third-party"],
        "zh": ["隐私", "私密", "个人信息", "个人数据", "数据安全", "权限",
               "授权", "安全", "跟踪", "追踪", "监控", "监视", "窃取",
               "泄露", "收集", "贩卖", "第三方", "保密", "黑客", "入侵"]
    },
    "A5_Performance": {
        "en": ["crash", "crashes", "crashing", "freeze", "freezes", "freezing",
               "frozen", "hang", "hanging", "lag", "laggy", "slow", "sluggish",
               "bug", "bugs", "buggy", "glitch", "glitchy", "error", "errors",
               "broken", "won't open", "won't load", "force close", "force stop",
               "restart", "reboot", "unresponsive", "stutter", "loading",
               "load time", "performance", "stable", "stability", "unstable"],
        "zh": ["崩溃", "闪退", "卡顿", "卡死", "死机", "卡", "慢", "延迟",
               "bug", "错误", "故障", "异常", "打不开", "无响应", "重启",
               "黑屏", "白屏", "加载", "转圈", "稳定性", "稳定", "不稳定",
               "掉线", "断开"]
    },
    "A6_Customer_Support": {
        "en": ["support", "customer service", "help", "helpful", "response",
               "respond", "reply", "contact", "email", "chat", "ticket",
               "assistance", "assist", "unhelpful", "ignored", "no response",
               "feedback", "complain", "complaint", "issue", "issues",
               "problem", "problems", "resolve", "resolution", "fix"],
        "zh": ["客服", "服务", "支持", "帮助", "回复", "解答", "联系",
               "邮件", "反馈", "投诉", "售后", "解决", "处理", "态度",
               "问题", "反馈", "没人理", "不回复"]
    },
    "A7_Content_Quality": {
        "en": ["content", "information", "info", "article", "articles",
               "guide", "guides", "resource", "resources", "material",
               "materials", "lesson", "lessons", "course", "courses",
               "audio", "sound", "voice", "music", "video", "quality",
               "accurate", "accuracy", "reliable", "reliability", "helpful",
               "informative", "useful", "detailed", "comprehensive",
               "outdated", "update", "updated", "old data", "wrong"],
        "zh": ["内容", "信息", "资料", "文章", "指南", "资源", "素材",
               "课程", "音频", "声音", "语音", "音乐", "视频", "质量",
               "准确", "可靠", "有用", "详细", "全面", "过时", "更新",
               "陈旧", "错误", "专业"]
    },
}

ASPECT_NAMES_CN = {
    "A1_UI_Design": "界面设计",
    "A2_Functionality": "功能实用性",
    "A3_Cost_Subscription": "费用订阅",
    "A4_Privacy_Security": "隐私安全",
    "A5_Performance": "性能稳定",
    "A6_Customer_Support": "客服支持",
    "A7_Content_Quality": "内容质量",
}

ASPECT_ORDER = list(ASPECT_DICT.keys())


def extract_aspects(text, lang):
    """从文本抽取涉及的方面(多标签). 返回 (aspect_list, matched_dict)."""
    if not text or not isinstance(text, str):
        return [], {}
    text_lower = text.lower()
    matched = {}
    for aspect in ASPECT_ORDER:
        kws = ASPECT_DICT[aspect].get(lang, []) + ASPECT_DICT[aspect].get("en", [])
        hits = [kw for kw in kws if kw.lower() in text_lower]
        if hits:
            matched[aspect] = hits
    return list(matched.keys()), matched


def main():
    lines = []
    def log(s=""):
        print(s); lines.append(str(s))

    df = pd.read_csv(IN_CSV)
    # tokens 列读取后为字符串, 不影响方面抽取(基于 clean_text)
    log("=" * 64)
    log("  第3步: 方面定义与抽取 报告")
    log("=" * 64)
    log(f"\n【输入】{IN_CSV}  ({len(df)} 行)")

    # ---- 方面体系说明 ----
    log(f"\n【1. 方面体系(7类)】")
    for a in ASPECT_ORDER:
        n_en = len(ASPECT_DICT[a]["en"])
        n_zh = len(ASPECT_DICT[a]["zh"])
        log(f"  {a} ({ASPECT_NAMES_CN[a]}): en={n_en}词 zh={n_zh}词")

    # ---- 方面抽取 ----
    log(f"\n【2. 方面抽取(基于关键词词典多标签匹配)】")
    aspect_lists = []
    aspect_details = []
    for _, row in df.iterrows():
        lang = row["lang"] if row["lang"] in ("en", "zh") else "en"
        asps, matched = extract_aspects(row["clean_text"], lang)
        aspect_lists.append(asps)
        # 记录匹配详情(用于审计)
        detail = "; ".join(f"{a}:{','.join(kws[:3])}" for a, kws in matched.items())
        aspect_details.append(detail)
    df["aspects"] = aspect_lists
    df["aspect_count"] = df["aspects"].apply(len)
    df["aspect_detail"] = aspect_details

    # ---- 统计 ----
    log(f"\n【3. 方面抽取统计】")
    # 命中率
    n_hit = int((df["aspect_count"] > 0).sum())
    log(f"  命中至少1个方面: {n_hit} / {len(df)}  ({n_hit/len(df)*100:.1f}%)")
    log(f"  未命中任何方面:   {len(df)-n_hit}  ({(len(df)-n_hit)/len(df)*100:.1f}%)")

    # 每条评论命中方面数分布
    log(f"\n  每条评论命中方面数分布:")
    ac = df["aspect_count"].value_counts().sort_index()
    for k, v in ac.items():
        log(f"    {k}个方面: {v}  ({v/len(df)*100:.1f}%)")

    # 各方面频次
    log(f"\n【4. 各方面频次】")
    all_aspects = []
    for asps in df["aspects"]:
        all_aspects += asps
    aspect_freq = Counter(all_aspects)
    log(f"  {'方面':<22s} {'中文名':<10s} {'频次':>6s} {'占比':>8s}")
    for a in ASPECT_ORDER:
        c = aspect_freq.get(a, 0)
        log(f"  {a:<22s} {ASPECT_NAMES_CN[a]:<10s} {c:>6d} {c/len(df)*100:>7.1f}%")

    # ---- 方面 x 情感 交叉 ----
    log(f"\n【5. 方面 x 情感 交叉表(方面命中时的情感分布)】")
    # 展开为长表: 每条评论的每个方面一行
    rows_long = []
    for _, row in df.iterrows():
        for a in row["aspects"]:
            rows_long.append({
                "review_id": row["review_id"],
                "aspect": a,
                "sentiment": row["sentiment_3label"],
                "rating": row["rating"],
                "lang": row["lang"],
            })
    df_long = pd.DataFrame(rows_long)
    log(f"  方面-情感对总数: {len(df_long)}")
    ct = pd.crosstab(df_long["aspect"], df_long["sentiment"],
                     margins=True)
    ct = ct.reindex(index=ASPECT_ORDER + ["All"],
                    columns=["negative","neutral","positive","All"], fill_value=0)
    log(f"\n{ct.to_string()}")

    # 各方面的负向率(论文重要发现)
    log(f"\n【6. 各方面负向率(关键发现)】")
    log(f"  {'方面':<22s} {'负向数':>6s} {'总数':>6s} {'负向率':>8s}")
    for a in ASPECT_ORDER:
        sub = df_long[df_long["aspect"] == a]
        n_neg = int((sub["sentiment"] == "negative").sum())
        n_total = len(sub)
        neg_rate = n_neg / n_total * 100 if n_total else 0
        log(f"  {a:<22s} {n_neg:>6d} {n_total:>6d} {neg_rate:>7.1f}%")

    # ---- 方面 x 语言 ----
    log(f"\n【7. 方面 x 语言 交叉表】")
    ct2 = pd.crosstab(df_long["aspect"], df_long["lang"], margins=True)
    ct2 = ct2.reindex(index=ASPECT_ORDER + ["All"], fill_value=0)
    log(f"\n{ct2.to_string()}")

    # ---- 未命中样本分析 ----
    log(f"\n【8. 未命中方面样本(随机10条, 用于优化词典)】")
    no_hit = df[df["aspect_count"] == 0]
    for _, row in no_hit.sample(min(10, len(no_hit)), random_state=42).iterrows():
        log(f"  [{row['lang']}/{row['rating']}星] {row['clean_text'][:90]}")

    # ---- 保存 ----
    df["aspects"] = df["aspects"].apply(lambda x: ", ".join(x) if x else "")
    out_cols = ["review_id", "platform", "app_name", "rating", "review_date",
                "raw_text", "clean_text", "lang", "tokens", "token_count",
                "sentiment_3class", "sentiment_3label", "sentiment_2class",
                "sentiment_2label", "aspects", "aspect_count", "aspect_detail"]
    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"\n【9. 输出】")
    log(f"  方面抽取数据: {OUT_CSV}  ({len(df)} 行)")

    # ---- 方面分布图 ----
    colors_a = ["#3498db","#2ecc71","#e74c3c","#9b59b6","#f39c12","#1abc9c","#e67e22"]
    fig, ax = plt.subplots(figsize=(9, 5))
    freqs = [aspect_freq.get(a, 0) for a in ASPECT_ORDER]
    labels = [f"{a}\n{ASPECT_NAMES_CN[a]}" for a in ASPECT_ORDER]
    bars = ax.barh(range(len(ASPECT_ORDER)), freqs, color=colors_a, edgecolor="white")
    ax.set_yticks(range(len(ASPECT_ORDER)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(freqs):
        ax.text(v + 30, i, f"{v} ({v/len(df)*100:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Frequency")
    ax.set_title("Aspect Distribution in Health App Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_aspect_distribution.png"), dpi=200)
    plt.close()

    # 方面 x 情感 堆叠图
    fig, ax = plt.subplots(figsize=(9, 5))
    ct_plot = pd.crosstab(df_long["aspect"], df_long["sentiment"])
    ct_plot = ct_plot.reindex(index=ASPECT_ORDER, columns=["negative","neutral","positive"], fill_value=0)
    ct_plot.plot(kind="barh", stacked=True, ax=ax,
                 color=["#e74c3c","#f39c12","#27ae60"], edgecolor="white")
    ax.set_yticklabels([f"{a}\n{ASPECT_NAMES_CN[a]}" for a in ASPECT_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Aspect- Sentiment Distribution")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_aspect_sentiment.png"), dpi=200)
    plt.close()
    log(f"  方面图表: {FIG_DIR}/  (2张PNG)")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  方面报告: {REPORT}")
    log(f"\n{'='*64}")
    log(f"  第3步(方面定义与抽取)完成 ✓")
    log(f"{'='*64}")


if __name__ == "__main__":
    main()
