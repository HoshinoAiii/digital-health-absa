# -*- coding: utf-8 -*-
"""
============================================================
 文本清洗与分词 —— 健康App用户评论方面级情感分析(ABSA)预处理
 模型方案: BERT -> BiLSTM -> Attention
 数据来源: D:\\codex\\all_health_reviews.csv (19143 条)
 输出:     cleaned_reviews.csv + preprocess_report.txt
============================================================
 本脚本完成数据预处理第一步:
   1. 数据加载与质量探查(缺失/重复/分布)
   2. 文本清洗(URL/HTML/控制符/emoji/重复字符/空白归一/大小写)
   3. 语言检测与中英文分流(中文字符占比启发式)
   4. 分词(中文 jieba / 英文 NLTK, 带正则回退)
============================================================
"""
import os
import re
import sys
import pandas as pd
import jieba
from collections import Counter

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
BASE      = r"D:\codex\nlp_thesis"
INPUT_CSV = r"D:\codex\all_health_reviews.csv"
OUT_CSV   = os.path.join(BASE, "cleaned_reviews.csv")
REPORT    = os.path.join(BASE, "preprocess_report.txt")
os.makedirs(BASE, exist_ok=True)

# ------------------------------------------------------------------
# NLTK 英文分词(带自动下载 + 正则回退)
# ------------------------------------------------------------------
NLTK_OK = False
try:
    import nltk
    from nltk.tokenize import word_tokenize as _wt
    try:
        _wt("a test sentence")
        NLTK_OK = True
    except LookupError:
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            _wt("a test sentence")
            NLTK_OK = True
        except Exception:
            NLTK_OK = False
except Exception:
    NLTK_OK = False

# 正则回退分词(匹配英文单词、数字、标点)
_REGEX_TOK = re.compile(r"\w+|[^\w\s]")
def regex_tokenize(text):
    return _REGEX_TOK.findall(text)


# ==================================================================
# 1. 数据加载
# ==================================================================
def load_data():
    df = pd.read_csv(INPUT_CSV)
    # 统一字段
    df["title"]   = df["title"].fillna("").astype(str).str.strip()
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    # 合并 title + content (Apple 有 title; Google Play title 多为空)
    # 若 title 与 content 完全相同(Apple 部分行), 只保留一份
    def merge_text(row):
        t, c = row["title"], row["content"]
        if t and t == c:
            return c
        return (t + " " + c).strip() if t else c
    df["raw_text"] = df.apply(merge_text, axis=1)
    return df


# ==================================================================
# 2. 文本清洗
# ==================================================================
URL_RE      = re.compile(r"https?://\S+|www\.\S+")
HTML_ENT_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
CONTROL_RE  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# emoji 及大量符号区
EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # 表情
    "\U0001F300-\U0001F5FF"  # 符号象形
    "\U0001F680-\U0001F6FF"  # 交通地图
    "\U0001F1E0-\U0001F1FF"  # 旗帜
    "\U0001F900-\U0001F9FF"  # 补充表情
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF"  # 杂项符号
    "\U0001F000-\U0001F02F"  # 麻将
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "\u2640-\u2642"
    "]+", flags=re.UNICODE)
REPEAT_RE     = re.compile(r"(.)\1{2,}")       # 连续>=3相同字符 -> 压缩为2
WHITESPACE_RE = re.compile(r"\s+")
# 全角标点 -> 半角(中文文本常见)
FULL2HALF = {
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'",
    "【": "[", "】": "]", "《": "<", "》": ">", "、": ",",
}
# 保留: 中文 + ASCII可打印 + 欧元/英镑/人民币 + 常见重音字母
KEEP_RE = re.compile(r"[^\u4e00-\u9fa5\x20-\x7e\u00a3\u20ac\u00a5\u00a2"
                     r"\u00c0-\u024f\u4e00-\u9fff]")


def clean_text(text):
    """文本清洗主函数(中英文兼顾)."""
    if not text:
        return ""
    text = str(text)
    # 1) URL
    text = URL_RE.sub(" ", text)
    # 2) HTML 实体
    text = HTML_ENT_RE.sub(" ", text)
    # 3) 控制字符
    text = CONTROL_RE.sub(" ", text)
    # 4) emoji / 大量符号 -> 移除(降噪; 情感主要由文字承载)
    text = EMOJI_RE.sub(" ", text)
    # 5) 全角标点转半角
    for full, half in FULL2HALF.items():
        text = text.replace(full, half)
    # 6) 移除保留集外的其余字符
    text = KEEP_RE.sub(" ", text)
    # 7) 连续重复字符压缩 (soooo->soo, !!!!->!!)
    text = REPEAT_RE.sub(r"\1\1", text)
    # 8) 空白归一
    text = WHITESPACE_RE.sub(" ", text).strip()
    # 9) 英文小写(BERT-base-uncased 友好; 中文无大小写无影响)
    text = text.lower()
    return text


# ==================================================================
# 3. 语言检测(中文字符占比启发式)
# ==================================================================
CN_CHAR_RE = re.compile(r"[\u4e00-\u9fa5]")
EN_CHAR_RE = re.compile(r"[a-zA-Z]")

def detect_lang(text):
    """中文字符占比 > 0.3 判为 zh, 否则 en."""
    if not text:
        return "unknown"
    cn = len(CN_CHAR_RE.findall(text))
    en = len(EN_CHAR_RE.findall(text))
    total = cn + en
    if total == 0:
        return "unknown"
    return "zh" if cn / total > 0.3 else "en"


# ==================================================================
# 4. 分词
# ==================================================================
# 中文停用词(轻量, 仅用于 EDA 词频; 不影响 BERT 输入)
CN_STOP = set("的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 "
              "着 没有 看 好 自己 这 那 他 她 它 我们 你们 他们 这个 那个 什么 怎么 "
              "这 这个 那个 但 还 而 与 及 等 把 被 让 使 向 从 给 对 为 以 于 之 "
              "可以 因为 所以 如果 虽然 但是 而且 或者 还是 已经 一直 一些 一下 "
              "没 不 很 非常 太 真 就 才 还 再 又 只".split())
EN_STOP = set("the a an and or but if of to in on at for with by is are was were "
              "be been being this that these those it its i my we our you your he "
              "she they them his her their as so not no do does did done have has "
              "had will would can could should shall may might must from into out "
              "up down over under again further then once here there all any both "
              "each few more most other some such only own same than too very just "
              "don t s m re ve ll d".split())


def tokenize(text, lang):
    """按语言分词, 返回 token 列表."""
    if not text:
        return []
    if lang == "zh":
        toks = [t for t in jieba.cut(text, cut_all=False) if t.strip()]
    else:
        if NLTK_OK:
            toks = _wt(text)
        else:
            toks = regex_tokenize(text)
    # 过滤纯空白 token
    toks = [t for t in toks if t.strip()]
    return toks


# ==================================================================
# 主流程
# ==================================================================
def main():
    lines = []  # 报告内容
    def log(s=""):
        print(s)
        lines.append(str(s))

    log("=" * 64)
    log("  健康App评论 — 文本清洗与分词 预处理报告")
    log("=" * 64)

    # ---- 加载 ----
    df = load_data()
    n_raw = len(df)
    log(f"\n【1. 原始数据】")
    log(f"  总记录数: {n_raw}")
    log(f"  字段: {list(df.columns)}")

    # ---- 质量探查 ----
    log(f"\n【2. 数据质量探查】")
    n_empty_raw = int(df["raw_text"].str.len().eq(0).sum())
    log(f"  raw_text 为空: {n_empty_raw}")
    dup_mask = df.duplicated(subset=["review_id"])
    n_dup = int(dup_mask.sum())
    log(f"  review_id 重复: {n_dup}")
    # content 完全重复(不同 review_id 但文本相同)
    n_textdup = int(df.duplicated(subset=["raw_text"]).sum())
    log(f"  raw_text 完全重复: {n_textdup}")

    # 评分分布
    log(f"\n  评分分布(原始):")
    for r, c in df["rating"].value_counts().sort_index().items():
        log(f"    {r}星: {c} ({c / n_raw * 100:.1f}%)")

    # 平台分布
    log(f"\n  平台分布:")
    for p, c in df["platform"].value_counts().items():
        log(f"    {p}: {c} ({c / n_raw * 100:.1f}%)")

    # App 分布
    log(f"\n  App 分布(Top 15):")
    for a, c in df["app_name"].value_counts().head(15).items():
        log(f"    {a}: {c}")

    # ---- 清洗 ----
    log(f"\n【3. 文本清洗】")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    # 清洗后为空
    n_empty_clean = int(df["clean_text"].str.len().eq(0).sum())
    log(f"  清洗后 clean_text 为空: {n_empty_clean}")

    # 清洗前后长度统计
    raw_len = df["raw_text"].str.len()
    clean_len = df["clean_text"].str.len()
    log(f"\n  文本长度(字符)统计:")
    log(f"    {'':12s} {'min':>6s} {'mean':>8s} {'median':>8s} {'max':>6s}")
    log(f"    {'原始':<10s} {raw_len.min():>6d} {raw_len.mean():>8.1f} "
        f"{raw_len.median():>8.1f} {raw_len.max():>6d}")
    log(f"    {'清洗后':<10s} {clean_len.min():>6d} {clean_len.mean():>8.1f} "
        f"{clean_len.median():>8.1f} {clean_len.max():>6d}")

    # ---- 语言检测 ----
    log(f"\n【4. 语言检测】")
    df["lang"] = df["clean_text"].apply(detect_lang)
    log(f"  语言分布:")
    for l, c in df["lang"].value_counts().items():
        log(f"    {l}: {c} ({c / n_raw * 100:.1f}%)")
    # 平台 x 语言 交叉
    log(f"\n  平台 x 语言 交叉表:")
    ct = pd.crosstab(df["platform"], df["lang"])
    log(f"    {ct.to_string()}")

    # ---- 分词 ----
    log(f"\n【5. 分词】")
    log(f"  英文分词器: {'NLTK word_tokenize' if NLTK_OK else '正则回退'}")
    log(f"  中文分词器: jieba (精确模式)")
    jieba.setLogLevel("ERROR")
    df["tokens"] = df.apply(lambda r: tokenize(r["clean_text"], r["lang"]), axis=1)
    df["token_count"] = df["tokens"].apply(len)
    tc = df["token_count"]
    log(f"\n  token 数量统计:")
    log(f"    min={tc.min()}  mean={tc.mean():.1f}  "
        f"median={tc.median():.0f}  max={tc.max()}")

    # ---- 去除无效记录(clean_text 为空) ----
    n_before = len(df)
    df_valid = df[df["clean_text"].str.len() > 0].copy()
    n_after = len(df_valid)
    log(f"\n【6. 有效记录】")
    log(f"  清洗前: {n_before}  ->  去除空文本后: {n_after}  "
        f"(移除 {n_before - n_after})")

    # ---- EDA: 高频词 ----
    log(f"\n【7. 高频词(去停用词, 仅用于探查)】")
    # 英文高频词
    en_tokens = []
    for toks in df_valid[df_valid["lang"] == "en"]["tokens"]:
        en_tokens += [t for t in toks if t.lower() not in EN_STOP
                      and len(t) > 1 and t.isalpha()]
    log(f"\n  英文 Top 30:")
    for w, c in Counter(en_tokens).most_common(30):
        log(f"    {w:<16s} {c}")
    # 中文高频词
    zh_tokens = []
    for toks in df_valid[df_valid["lang"] == "zh"]["tokens"]:
        zh_tokens += [t for t in toks if t not in CN_STOP
                      and len(t.strip()) >= 2
                      and not re.fullmatch(r"[^\u4e00-\u9fa5]+", t)]
    log(f"\n  中文 Top 30:")
    for w, c in Counter(zh_tokens).most_common(30):
        log(f"    {w:<16s} {c}")

    # ---- 评分 x 语言 交叉 ----
    log(f"\n【8. 评分 x 语言 交叉表(有效记录)】")
    ct2 = pd.crosstab(df_valid["rating"], df_valid["lang"], margins=True)
    log(f"    {ct2.to_string()}")

    # ---- 清洗示例 ----
    log(f"\n【9. 清洗示例(随机 8 条)】")
    sample = df_valid.sample(min(8, len(df_valid)), random_state=42)
    for _, row in sample.iterrows():
        raw = row["raw_text"][:80]
        clean = row["clean_text"][:80]
        log(f"  [{row['lang']}/{row['rating']}星] {row['platform']}")
        log(f"    原始: {raw}")
        log(f"    清洗: {clean}")
        log(f"    分词: {' '.join(row['tokens'][:15])}")
        log("")

    # ---- 保存 ----
    log(f"\n【10. 输出文件】")
    out_cols = ["review_id", "platform", "app_name", "rating", "review_date",
                "raw_text", "clean_text", "lang", "tokens", "token_count"]
    df_valid["tokens"] = df_valid["tokens"].apply(lambda x: " ".join(x))
    df_valid[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"  清洗后数据: {OUT_CSV}  ({n_after} 行)")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  清洗报告:   {REPORT}")

    log(f"\n{'=' * 64}")
    log(f"  预处理第一步(文本清洗与分词)完成 ✓")
    log(f"{'=' * 64}")


if __name__ == "__main__":
    main()
