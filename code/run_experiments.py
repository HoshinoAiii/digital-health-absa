# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""主实验编排:
- 载入 BERT 缓存特征 + 词 id 特征
- 训练 6 个模型(TextCNN, BiLSTM-Att, BERT, BERT-BiLSTM, 本文, +LLM 基线)
  多随机种子, 在自动标签 test 与 high-conf 金标准子集双报告
- 消融(去 BiLSTM / 去注意力)
- 加权 vs 不加权(neutral 类分析)
- 各方面 F1、混淆矩阵、注意力权重样例、训练曲线
结果全部落 figures/*.json。"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import torch

import utils as u
import models as M
import train as T
from data_prep import PROC, encode_wordids, load_and_split

SEEDS = [42, 7, 123]   # CPU 约束下 3 种子(支持 mean±std + 配对检验); 显著性另用配对 bootstrap
CACHE = u.CACHE_DIR


def load_cache():
    def L(n):
        return np.load(os.path.join(CACHE, n))
    cache = {
        "seq": L("bert_seq.npy"), "pooled": L("bert_pooled.npy"),
        "mask": L("bert_mask.npy"), "aspect_vec": L("aspect_vec.npy"),
        "labels": L("labels.npy"), "aspect_ids": L("aspect_ids.npy"),
        "is_gold": L("is_gold.npy"), "split": L("split.npy"),
    }
    return cache


def split_idx(cache):
    sp = cache["split"]
    tr = np.where(sp == 0)[0]; va = np.where(sp == 1)[0]; te = np.where(sp == 2)[0]
    gold_te = te[cache["is_gold"][te] == 1]
    return tr, va, te, gold_te


def eval_on(model, cache, idx, use_seq):
    y = cache["labels"][idx]
    p = T.predict(model, cache, idx, use_seq)
    return u.metrics_full(y, p), p


def agg(runs):
    """对多种子结果求 mean/std。runs: list of metric dict。"""
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
        out[k + "_runs"] = [float(v) for v in vals]
    return out


def build_model(name, vocab_size):
    if name == "TextCNN":
        return M.TextCNN(vocab_size), "scratch"
    if name == "BiLSTM-Attention":
        return M.BiLSTMAttentionScratch(vocab_size), "scratch"
    if name == "BERT":
        return M.BertLinear(), "pooled"
    if name == "BERT-BiLSTM":
        return M.BertBiLSTMAtt(use_lstm=True, use_att=False), "seq"
    if name == "BERT-BiLSTM-Attention":
        return M.BertBiLSTMAtt(use_lstm=True, use_att=True), "seq"
    raise ValueError(name)


def main():
    t_start = time.time()
    if not os.path.exists(os.path.join(CACHE, "bert_seq.npy")):
        print("[exp] BERT cache missing - run encode_bert.py first")
        sys.exit(1)
    cache = load_cache()
    tr, va, te, gold_te = split_idx(cache)
    class_w = u.compute_class_weights(cache["labels"][tr])
    u.save_json({u.ID2LABEL[i]: float(class_w[i]) for i in range(3)},
                os.path.join(u.FIG_DIR, "class_weights.json"))
    print(f"[exp] train={len(tr)} val={len(va)} test={len(te)} gold_test={len(gold_te)}")
    print(f"[exp] class_weights={class_w}")

    # 词 id 特征(供 scratch 模型)
    df = pd.read_pickle(PROC)
    vocab = u.load_json(os.path.join(CACHE, "vocab.json"))
    texts = df["clean_text"].values
    aids = df["aspect_id"].values
    Xword, Aword = encode_wordids(texts, aids, vocab, max_len=40)
    yall = df["label"].values

    results = {"seeds": SEEDS, "n_train": int(len(tr)),
               "n_test": int(len(te)), "n_gold_test": int(len(gold_te))}
    main_models = ["TextCNN", "BiLSTM-Attention", "BERT",
                   "BERT-BiLSTM", "BERT-BiLSTM-Attention"]
    per_model = {}
    curves = {}
    confusions = {}
    test_preds_store = {}

    for name in main_models:
        auto_runs, gold_runs = [], []
        t0 = time.time()
        for si, seed in enumerate(SEEDS):
            u.set_seed(seed)
            model, kind = build_model(name, len(vocab))
            if kind == "scratch":
                model, hist = T.train_scratch(
                    model, Xword[tr], Aword[tr], yall[tr],
                    Xword[va], Aword[va], yall[va], class_w, seed)
                p_te = T.predict_scratch(model, Xword[te], Aword[te])
                p_gold = T.predict_scratch(model, Xword[gold_te], Aword[gold_te])
                m_auto = u.metrics_full(yall[te], p_te)
                m_gold = u.metrics_full(yall[gold_te], p_gold)
            else:
                use_seq = (kind == "seq")
                model, hist = T.train_bert_head(
                    model, cache, tr, va, te, class_w, seed, use_seq=use_seq)
                m_auto, p_te = eval_on(model, cache, te, use_seq)
                m_gold, p_gold = eval_on(model, cache, gold_te, use_seq)
            auto_runs.append(m_auto); gold_runs.append(m_gold)
            if si == 0:  # 存首个种子的曲线/混淆/预测
                curves[name] = hist
                confusions[name] = u.confusion(cache["labels"][te] if kind != "scratch"
                                                else yall[te], p_te) if hasattr(u, "confusion") else None
                test_preds_store[name] = p_te.tolist()
            print(f"[exp] {name} seed={seed} auto_f1={m_auto['macro_f1']:.4f} "
                  f"gold_f1={m_gold['macro_f1']:.4f}", flush=True)
        # 存 seed0 的 gold_test 预测(供显著性检验)
        if 'gold_pred_store' not in dir():
            pass
        per_model[name] = {
            "auto_test": agg(auto_runs),
            "gold_test": agg(gold_runs),
            "per_class_gold_seed0": gold_runs[0]["per_class"],
            "gold_pred_seed0": (p_gold.tolist() if isinstance(p_gold, np.ndarray)
                                else list(p_gold)),
        }
        print(f"[exp] {name} DONE in {time.time()-t0:.0f}s "
              f"gold_macroF1={per_model[name]['gold_test']['macro_f1_mean']:.4f}"
              f"±{per_model[name]['gold_test']['macro_f1_std']:.4f}", flush=True)

    # ---- 配对 bootstrap 显著性检验(本文 vs 最强基线, gold_test seed0) ----
    y_gold = cache["labels"][gold_te]
    from sklearn.metrics import f1_score
    proposed = "BERT-BiLSTM-Attention"
    baselines = [m for m in main_models if m != proposed]
    best_base = max(baselines,
                    key=lambda m: per_model[m]["gold_test"]["macro_f1_mean"])
    pa = np.array(per_model[proposed]["gold_pred_seed0"])
    pb = np.array(per_model[best_base]["gold_pred_seed0"])
    rng = np.random.RandomState(12345)
    n = len(y_gold); diffs = []
    B = 2000
    for _ in range(B):
        idx = rng.randint(0, n, n)
        fa = f1_score(y_gold[idx], pa[idx], average="macro", zero_division=0)
        fb = f1_score(y_gold[idx], pb[idx], average="macro", zero_division=0)
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    p_val = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    results["significance"] = {
        "proposed": proposed, "best_baseline": best_base,
        "mean_macroF1_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)),
                 float(np.percentile(diffs, 97.5))],
        "p_value_bootstrap": p_val, "n_bootstrap": B,
        "method": "paired bootstrap on gold_test (seed0 predictions)",
    }
    print(f"[exp] significance {proposed} vs {best_base}: "
          f"dMacroF1={diffs.mean():.4f} p={p_val:.4f}", flush=True)

    results["models"] = per_model
    results["curves"] = curves
    results["confusions"] = confusions
    u.save_json(results, os.path.join(u.FIG_DIR, "main_results.json"))
    print(f"[exp] main experiments done in {time.time()-t_start:.0f}s")
    return results, cache, (tr, va, te, gold_te), class_w, (Xword, Aword, yall, vocab)


if __name__ == "__main__":
    main()
