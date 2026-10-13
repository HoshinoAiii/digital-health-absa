# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""探索性/实证分析(全量真实数据):
1. 文档级分布(评分/情感/方面数/文本长度) -> fig_data_dist
2. 各方面情感构成 + 负向率排序 -> fig_aspect_sentiment / fig_aspect_negrate
3. 方面级 vs 文档级一致率 + Cohen's kappa
4. 标注可信度代理: high-confidence 子集上 aspect vs doc 的 kappa/一致率
   (诚实标注: 无人工二次标注, 用窗口高置信标签与文档级标签一致性做代理)
5. 类别不平衡与类别权重
所有结果写入 figures/*.json。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

import utils as u

LABELED = os.path.join(os.path.dirname(_HERE), "user_data", "absa_labeled.csv")
LONG = os.path.join(os.path.dirname(_HERE), "user_data", "absa_long.csv")


def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def main():
    dfl = pd.read_csv(LABELED)
    dfa = pd.read_csv(LONG)
    dfa = dfa[dfa["aspect_sentiment"].isin(u.LABELS)]
    dfa = dfa[dfa["aspect"].isin(u.ASPECTS)].copy()

    out = {}

    # ---- 1. 文档级分布 ----
    out["n_reviews_labeled"] = int(len(dfl))
    out["rating_dist"] = {int(k): int(v) for k, v in
                          dfl["rating"].value_counts().sort_index().items()}
    out["doc_sentiment_dist"] = {
        k: int(v) for k, v in dfl["sentiment_3label"].value_counts().items()}
    out["aspect_count_dist"] = {int(k): int(v) for k, v in
                                dfl["aspect_count"].value_counts().sort_index().items()}
    tc = dfl["token_count"].astype(float)
    out["token_count_stats"] = {
        "mean": float(tc.mean()), "median": float(tc.median()),
        "std": float(tc.std()), "min": int(tc.min()), "max": int(tc.max()),
        "p90": float(tc.quantile(0.90)), "p95": float(tc.quantile(0.95)),
        "pct_le_128": float((tc <= 128).mean()),
        "hist_bins": [int(x) for x in np.histogram(tc.clip(upper=120), bins=24)[0]],
        "hist_edges": [float(x) for x in np.histogram(tc.clip(upper=120), bins=24)[1]],
    }
    hit = dfl["aspect_count"] > 0
    out["aspect_hit_rate"] = float(hit.mean())
    out["mean_aspects_per_hit_review"] = float(
        dfl.loc[hit, "aspect_count"].mean())

    # ---- 2. 各方面情感构成 + 负向率 ----
    aspect_sent = {}
    neg_rate = {}
    freq = {}
    for a in u.ASPECTS:
        sub = dfa[dfa.aspect == a]
        vc = sub["aspect_sentiment"].value_counts()
        d = {l: int(vc.get(l, 0)) for l in u.LABELS}
        aspect_sent[a] = d
        n = sum(d.values())
        freq[a] = n
        neg_rate[a] = float(d["negative"] / n) if n else 0.0
    out["aspect_sentiment_counts"] = aspect_sent
    out["aspect_neg_rate"] = neg_rate
    out["aspect_freq"] = freq
    out["aspect_neg_rate_sorted"] = sorted(
        neg_rate.items(), key=lambda x: -x[1])

    # ---- 3. 方面级 vs 文档级一致性 + Cohen's kappa ----
    both = dfa.dropna(subset=["aspect_sentiment", "doc_sentiment"])
    both = both[both["doc_sentiment"].isin(u.LABELS)]
    a_ids = both["aspect_sentiment"].map(u.LABEL2ID).values
    d_ids = both["doc_sentiment"].map(u.LABEL2ID).values
    out["aspect_doc_agreement"] = float((a_ids == d_ids).mean())
    out["aspect_doc_kappa"] = cohen_kappa(a_ids, d_ids)
    out["n_agreement_pairs"] = int(len(both))

    # ---- 4. 标注可信度代理(high-confidence 子集) ----
    # 诚实说明: 无人工二次标注, 用"窗口高置信"子集与文档级标签的一致性/kappa
    # 作为弱监督标签可信度的代理指标(非人工 IAA)。
    hi = both[both["confidence"] == "high"]
    if len(hi):
        ha = hi["aspect_sentiment"].map(u.LABEL2ID).values
        hd = hi["doc_sentiment"].map(u.LABEL2ID).values
        out["gold_proxy"] = {
            "n_high_conf": int(len(hi)),
            "agreement_vs_doc": float((ha == hd).mean()),
            "kappa_vs_doc": cohen_kappa(ha, hd),
            "note": "代理指标: 高置信(窗口内含情感词)方面标签与文档级标签的一致性; 非人工二次标注",
        }
    out["confidence_dist"] = {
        k: int(v) for k, v in dfa["confidence"].value_counts().items()}

    # ---- 5. 类别不平衡与类别权重 ----
    lab_ids = dfa["aspect_sentiment"].map(u.LABEL2ID).values
    counts = np.bincount(lab_ids, minlength=3)
    out["class_counts"] = {u.ID2LABEL[i]: int(counts[i]) for i in range(3)}
    out["imbalance_ratio"] = float(counts.max() / counts.min())
    cw = u.compute_class_weights(lab_ids)
    out["class_weights"] = {u.ID2LABEL[i]: float(cw[i]) for i in range(3)}

    def _native(o):
        if isinstance(o, dict):
            return {k: _native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_native(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    u.save_json(_native(out), os.path.join(u.FIG_DIR, "eda_stats.json"))
    print(f"[eda] agreement={out['aspect_doc_agreement']:.4f} "
          f"kappa={out['aspect_doc_kappa']:.4f} imbalance={out['imbalance_ratio']:.2f}")
    return dfl, dfa, out


if __name__ == "__main__":
    main()
