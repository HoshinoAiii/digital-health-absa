# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""数据准备: 读取 absa_long.csv, 构造 (评论, 方面) -> 极性 样本,
分层 8:1:1 划分, 构建词表(供从零训练的 TextCNN/BiLSTM 用),
并标记 high-confidence 金标准子集。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils as u

RAW_LONG = os.path.join(os.path.dirname(_HERE), "user_data", "absa_long.csv")
PROC = os.path.join(u.DATA_DIR, "absa_prepared.pkl")

_token_re = re.compile(r"[a-z0-9']+")


def _tokenize(text):
    return _token_re.findall(str(text).lower())


def load_and_split(seed=42, max_vocab=20000):
    """读取 -> 清洗 -> 分层划分 -> 词表。返回 dict。"""
    df = pd.read_csv(RAW_LONG)
    n0 = len(df)
    # 只保留有效三分类标签
    df = df[df["aspect_sentiment"].isin(u.LABELS)].copy()
    df = df[df["aspect"].isin(u.ASPECTS)].copy()
    df["clean_text"] = df["clean_text"].astype(str).fillna("")
    df = df[df["clean_text"].str.len() > 0].copy()
    df = df.reset_index(drop=True)
    print(f"[data] 原始 {n0} 行 -> 有效 {len(df)} 行")

    df["label"] = df["aspect_sentiment"].map(u.LABEL2ID).astype(int)
    df["aspect_id"] = df["aspect"].map(u.ASPECT2ID).astype(int)
    df["is_gold"] = (df["confidence"] == "high").astype(int)

    # ⛔ 按 review_id 分组划分 8:1:1(防止同一评论文本跨 train/test 泄漏)。
    # 同一评论的所有方面-情感对整体进入同一子集。
    from sklearn.model_selection import GroupShuffleSplit
    groups = df["review_id"].values
    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, tmp = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    tmp_va, tmp_te = next(gss2.split(tmp, groups=groups[tmp]))
    va, te = tmp[tmp_va], tmp[tmp_te]
    # 校验无 review_id 跨集泄漏
    s_tr, s_va, s_te = set(groups[tr]), set(groups[va]), set(groups[te])
    assert not (s_tr & s_te) and not (s_tr & s_va) and not (s_va & s_te), \
        "review_id leakage across splits!"
    split = np.empty(len(df), dtype=object)
    split[tr] = "train"; split[va] = "val"; split[te] = "test"
    df["split"] = split
    print(f"[data] split(group by review_id): train={len(tr)} val={len(va)} test={len(te)}")
    print(f"[data] 无 review_id 跨集泄漏 ✅")
    print(f"[data] gold(high-conf)总数={int(df['is_gold'].sum())}, "
          f"test 内 gold={int(df.loc[df.split=='test','is_gold'].sum())}")

    # 词表(仅用 train 文本, 防泄漏)
    from collections import Counter
    cnt = Counter()
    for t in df.loc[df.split == "train", "clean_text"]:
        cnt.update(_tokenize(t))
    # 0=PAD, 1=UNK
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(max_vocab - 2):
        vocab[w] = len(vocab)
    print(f"[data] vocab size={len(vocab)} (train tokens unique={len(cnt)})")

    os.makedirs(u.DATA_DIR, exist_ok=True)
    df.to_pickle(PROC)
    u.save_json(vocab, os.path.join(u.CACHE_DIR, "vocab.json"))
    return df, vocab


def encode_wordids(texts, aspect_ids, vocab, max_len=40):
    """把文本转成词 id 序列(供 TextCNN/BiLSTM 从零训练)。"""
    pad, unk = 0, 1
    X = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = _tokenize(t)[:max_len]
        for j, w in enumerate(toks):
            X[i, j] = vocab.get(w, unk)
        # 保底: 若文本经英文正则分词后无 token(如纯中文评论),
        # 至少放一个 <unk>, 避免整行全 <pad> 导致方面注意力 softmax 全 -inf -> NaN。
        if len(toks) == 0:
            X[i, 0] = unk
    A = np.asarray(aspect_ids, dtype=np.int64)
    return X, A


if __name__ == "__main__":
    df, vocab = load_and_split()
    # 描述性统计落盘
    desc = {
        "n_pairs": int(len(df)),
        "n_reviews": int(df["review_id"].nunique()),
        "label_dist": {u.ID2LABEL[k]: int(v) for k, v in
                       df["label"].value_counts().sort_index().items()},
        "aspect_dist": {a: int((df.aspect == a).sum()) for a in u.ASPECTS},
        "lang_dist": df["lang"].value_counts().to_dict(),
        "confidence_dist": df["confidence"].value_counts().to_dict(),
        "split_dist": df["split"].value_counts().to_dict(),
        "n_gold": int(df["is_gold"].sum()),
        "aspect_doc_agreement": float(
            (df["aspect_sentiment"] == df["doc_sentiment"]).mean()
        ),
    }
    # 各方面负向率(核心发现)
    negrate = {}
    for a in u.ASPECTS:
        sub = df[df.aspect == a]
        negrate[a] = float((sub.aspect_sentiment == "negative").mean())
    desc["aspect_neg_rate"] = negrate
    u.save_json(desc, os.path.join(u.FIG_DIR, "descriptive_stats.json"))
    print("[data] 描述统计已保存")
