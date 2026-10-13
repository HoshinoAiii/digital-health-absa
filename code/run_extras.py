# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""附加实验(依赖 run_experiments 的缓存与划分):
1. 消融: 本文模型去 BiLSTM / 去注意力 (单种子=42, 金标准子集)
2. 加权 vs 不加权交叉熵对 neutral 类 F1 的影响
3. 各方面 F1 (本文 vs 最强基线 BERT-BiLSTM)
4. 混淆矩阵(本文, 金标准子集)
5. 注意力权重样例(本文, 若干 test 样本 token 级)
6. 训练曲线(已由 run_experiments 存 curves)
结果写 figures/*.json。"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import torch

import utils as u
import models as M
import train as T
from run_experiments import load_cache, split_idx
from data_prep import PROC

SEED = 42


def ablation(cache, tr, va, te, gold_te, class_w):
    configs = {
        "Full (BiLSTM+Attention)": dict(use_lstm=True, use_att=True),
        "w/o Attention (mean-pool)": dict(use_lstm=True, use_att=False),
        "w/o BiLSTM (BERT+Attention)": dict(use_lstm=False, use_att=True),
        "w/o both (BERT mean-pool)": dict(use_lstm=False, use_att=False),
    }
    out = {}
    for name, cfg in configs.items():
        u.set_seed(SEED)
        model = M.BertBiLSTMAtt(**cfg)
        model, _ = T.train_bert_head(model, cache, tr, va, te, class_w,
                                     SEED, use_seq=True)
        p = T.predict(model, cache, gold_te, use_seq=True)
        m = u.metrics_full(cache["labels"][gold_te], p)
        out[name] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "weighted_f1": m["weighted_f1"]}
        print(f"[ablation] {name}: macroF1={m['macro_f1']:.4f}", flush=True)
    u.save_json(out, os.path.join(u.FIG_DIR, "ablation_results.json"))
    return out


def weighting(cache, tr, va, te, gold_te, class_w):
    out = {}
    for tag, w in [("weighted", class_w), ("unweighted", np.ones(3))]:
        u.set_seed(SEED)
        model = M.BertBiLSTMAtt(use_lstm=True, use_att=True)
        model, _ = T.train_bert_head(model, cache, tr, va, te, w, SEED,
                                     use_seq=True)
        p = T.predict(model, cache, gold_te, use_seq=True)
        m = u.metrics_full(cache["labels"][gold_te], p)
        out[tag] = {
            "macro_f1": m["macro_f1"],
            "neutral_f1": m["per_class"]["neutral"]["f1"],
            "neutral_recall": m["per_class"]["neutral"]["recall"],
            "per_class": m["per_class"],
        }
        print(f"[weighting] {tag}: macroF1={m['macro_f1']:.4f} "
              f"neutralF1={out[tag]['neutral_f1']:.4f}", flush=True)
    u.save_json(out, os.path.join(u.FIG_DIR, "weighting_results.json"))
    return out


def per_aspect(cache, tr, va, te, gold_te, class_w):
    """训练本文与 BERT-BiLSTM, 在 gold_test 上按方面拆 F1。"""
    from sklearn.metrics import f1_score
    trained = {}
    for name, cfg in [("BERT-BiLSTM-Attention", dict(use_lstm=True, use_att=True)),
                      ("BERT-BiLSTM", dict(use_lstm=True, use_att=False))]:
        u.set_seed(SEED)
        model = M.BertBiLSTMAtt(**cfg)
        model, _ = T.train_bert_head(model, cache, tr, va, te, class_w,
                                     SEED, use_seq=True)
        trained[name] = model
    aids = cache["aspect_ids"]
    y = cache["labels"]
    out = {}
    for name, model in trained.items():
        p = T.predict(model, cache, gold_te, use_seq=True)
        per = {}
        for ai, aname in enumerate(u.ASPECTS):
            m = aids[gold_te] == ai
            if m.sum() >= 5:
                per[aname] = float(f1_score(y[gold_te][m], p[m],
                                            average="macro", zero_division=0))
            else:
                per[aname] = None
        out[name] = per
        print(f"[per_aspect] {name} done", flush=True)
    u.save_json(out, os.path.join(u.FIG_DIR, "per_aspect_f1.json"))
    return out


def confusion_and_attention(cache, tr, va, te, gold_te, class_w):
    u.set_seed(SEED)
    model = M.BertBiLSTMAtt(use_lstm=True, use_att=True)
    model, _ = T.train_bert_head(model, cache, tr, va, te, class_w, SEED,
                                 use_seq=True)
    # 混淆矩阵
    p, alpha = T.predict(model, cache, gold_te, use_seq=True, return_alpha=True)
    cm = u.confusion(cache["labels"][gold_te], p)
    u.save_json({"labels": u.LABELS, "matrix": cm,
                 "subset": "gold_test", "n": int(len(gold_te))},
                os.path.join(u.FIG_DIR, "confusion_matrix.json"))
    print(f"[confusion] saved, n={len(gold_te)}", flush=True)

    # 注意力样例: 取若干 high-conf test 样本, 还原 token 与权重
    df = pd.read_pickle(PROC)
    df_test = df[df.split == "test"].reset_index(drop=True)
    # gold_te 索引是相对 cache 全量的; 需要映射回 df 行。data_prep 与 cache 同序。
    import re
    tokre = re.compile(r"[a-z0-9']+")
    samples = []
    if alpha is not None:
        # 选前几个预测正确且为 negative 的样本做展示
        sel = 0
        for k, gi in enumerate(gold_te):
            if sel >= 6:
                break
            true = cache["labels"][gi]; pred = p[k]
            if true == pred and true in (0, 2):
                text = str(df.iloc[gi]["clean_text"])
                toks = tokre.findall(text.lower())[:30]
                a = alpha[k][:len(toks)]
                if len(toks) < 4:
                    continue
                samples.append({
                    "aspect": u.ASPECTS[int(cache["aspect_ids"][gi])],
                    "true": u.ID2LABEL[int(true)],
                    "pred": u.ID2LABEL[int(pred)],
                    "tokens": toks,
                    "weights": [float(x) for x in a],
                })
                sel += 1
    u.save_json({"samples": samples},
                os.path.join(u.FIG_DIR, "attention_samples.json"))
    print(f"[attention] saved {len(samples)} samples", flush=True)


def main():
    cache = load_cache()
    tr, va, te, gold_te = split_idx(cache)
    class_w = u.compute_class_weights(cache["labels"][tr])
    ablation(cache, tr, va, te, gold_te, class_w)
    weighting(cache, tr, va, te, gold_te, class_w)
    per_aspect(cache, tr, va, te, gold_te, class_w)
    confusion_and_attention(cache, tr, va, te, gold_te, class_w)
    print("[extras] all done")


if __name__ == "__main__":
    main()
