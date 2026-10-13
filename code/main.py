# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""主程序: 串联数据准备 -> BERT 编码 -> 训练全部模型(5 种子) ->
消融 -> LLM 基线 -> 各方面 F1 -> 混淆矩阵 -> 注意力样例 -> 汇总 JSON。"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import pandas as pd
import torch

import utils as u
from data_prep import PROC, load_and_split, encode_wordids
import models as M
import train as T

SEEDS = [42, 1, 7, 123, 2024]
FIG = u.FIG_DIR


def load_cache():
    c = u.CACHE_DIR
    cache = {
        "seq": np.load(os.path.join(c, "bert_seq.npy"), mmap_mode="r"),
        "pooled": np.load(os.path.join(c, "bert_pooled.npy")),
        "mask": np.load(os.path.join(c, "bert_mask.npy")),
        "aspect_vec": np.load(os.path.join(c, "aspect_vec.npy")),
        "labels": np.load(os.path.join(c, "labels.npy")),
        "aspect_ids": np.load(os.path.join(c, "aspect_ids.npy")),
        "is_gold": np.load(os.path.join(c, "is_gold.npy")),
        "split": np.load(os.path.join(c, "split.npy")),
    }
    return cache


def agg(runs):
    """对多种子 metrics 求 mean/std。"""
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    return out


def main():
    t_start = time.time()
    if not os.path.exists(PROC):
        load_and_split()
    df = pd.read_pickle(PROC)
    cache = load_cache()
    labels = cache["labels"]; split = cache["split"]; gold = cache["is_gold"]
    tr = np.where(split == 0)[0]; va = np.where(split == 1)[0]; te = np.where(split == 2)[0]
    te_gold = te[gold[te] == 1]
    class_w = u.compute_class_weights(labels[tr])
    print(f"[main] train={len(tr)} val={len(va)} test={len(te)} "
          f"test_gold={len(te_gold)} class_w={np.round(class_w,3)}", flush=True)

    results = {"meta": {
        "seeds": SEEDS, "class_weights": class_w.tolist(),
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "n_test": int(len(te)), "n_test_gold": int(len(te_gold)),
        "note": ("BERT 作为冻结特征抽取器(feature-based, CPU 环境无 GPU, "
                 "端到端微调不可行); 下游头训练 5 种子。"),
    }}

    # ---------- 标签可信度: aspect-level vs doc-level 一致性 ----------
    # 说明: 本工作无人工重标注资源, 故报告"方面级弱监督标签"与"文档级标签"
    # 的自动一致性(Cohen's κ)作为标注可信度的可测代理, 并非人工标注者间一致性。
    from sklearn.metrics import cohen_kappa_score
    mm = df["aspect_sentiment"].notna() & df["doc_sentiment"].notna()
    a_lab = df.loc[mm, "aspect_sentiment"].map(u.LABEL2ID).values
    d_lab = df.loc[mm, "doc_sentiment"].map(u.LABEL2ID).values
    kappa = float(cohen_kappa_score(a_lab, d_lab))
    agree = float((a_lab == d_lab).mean())
    hi = df["confidence"] == "high"
    mm_hi = mm & hi
    a_hi = df.loc[mm_hi, "aspect_sentiment"].map(u.LABEL2ID).values
    d_hi = df.loc[mm_hi, "doc_sentiment"].map(u.LABEL2ID).values
    results["label_reliability"] = {
        "definition": ("aspect-level(弱监督) vs doc-level 标签自动一致性; "
                       "非人工 IAA, 无人工重标注资源时的可测代理"),
        "agreement_all": agree,
        "cohen_kappa_all": kappa,
        "agreement_highconf": float((a_hi == d_hi).mean()),
        "cohen_kappa_highconf": float(cohen_kappa_score(a_hi, d_hi)),
        "n_all": int(mm.sum()), "n_highconf": int(mm_hi.sum()),
    }

    # 预备从零模型的词 id
    vocab = u.load_json(os.path.join(u.CACHE_DIR, "vocab.json"))
    texts = df["clean_text"].tolist(); asp_ids = df["aspect_id"].values
    Xall, Aall = encode_wordids(texts, asp_ids, vocab, max_len=32)

    model_specs = {
        "TextCNN": ("scratch", dict(kind="cnn")),
        "BiLSTM-Attention": ("scratch", dict(kind="bilstm")),
        "BERT": ("bert", dict(use_seq=False)),
        "BERT-BiLSTM": ("bert", dict(use_seq=True, use_lstm=True, use_att=False)),
        "BERT-BiLSTM-Attention": ("bert", dict(use_seq=True, use_lstm=True, use_att=True)),
    }

    per_model_runs = {}
    test_metrics = {}
    attn_alpha_store = None
    curve_store = {}
    seed0_pred = {}          # name -> test 预测(seed0), 供各方面 F1
    macro_f1_seeds = {}      # name -> [各种子 gold macro_f1], 供显著性检验

    for name, (typ, cfg) in model_specs.items():
        runs = []; gold_runs = []
        print(f"\n===== {name} ({typ}) =====", flush=True)
        for si, seed in enumerate(SEEDS):
            u.set_seed(seed)
            t0 = time.time()
            if typ == "scratch":
                if cfg["kind"] == "cnn":
                    mdl = M.TextCNN(len(vocab))
                else:
                    mdl = M.BiLSTMAttentionScratch(len(vocab))
                mdl, hist = T.train_scratch(
                    mdl, Xall[tr], Aall[tr], labels[tr],
                    Xall[va], Aall[va], labels[va], class_w, seed)
                pred = T.predict_scratch(mdl, Xall[te], Aall[te])
                pred_gold = T.predict_scratch(mdl, Xall[te_gold], Aall[te_gold])
            else:
                use_seq = cfg["use_seq"]
                if use_seq:
                    mdl = M.BertBiLSTMAtt(use_lstm=cfg["use_lstm"], use_att=cfg["use_att"])
                else:
                    mdl = M.BertLinear()
                mdl, hist = T.train_bert_head(
                    mdl, cache, tr, va, te, class_w, seed, use_seq=use_seq)
                pred = T.predict(mdl, cache, te, use_seq)
                pred_gold = T.predict(mdl, cache, te_gold, use_seq)
            m_auto = u.metrics_full(labels[te], pred)
            m_gold = u.metrics_full(labels[te_gold], pred_gold)
            runs.append(m_auto); gold_runs.append(m_gold)
            print(f"  seed={seed} auto_mF1={m_auto['macro_f1']:.4f} "
                  f"gold_mF1={m_gold['macro_f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            # 保存 seed0 的曲线 / 预测 / 注意力
            if si == 0:
                curve_store[name] = hist
                seed0_pred[name] = pred
                if name == "BERT-BiLSTM-Attention":
                    np.save(os.path.join(u.CACHE_DIR, "test_pred_proposed.npy"), pred)
                    # 注意力样例(前若干 test)
                    sample_idx = te[:60]
                    _, alpha = T.predict(mdl, cache, sample_idx, True, return_alpha=True)
                    attn_alpha_store = (sample_idx, alpha)
        per_model_runs[name] = {"auto": runs, "gold": gold_runs}
        macro_f1_seeds[name] = [r["macro_f1"] for r in gold_runs]
        test_metrics[name] = {
            "auto": agg(runs), "gold": agg(gold_runs),
            "per_class_auto": runs[0]["per_class"],
        }

    results["models"] = test_metrics
    results["training_curves"] = curve_store

    # ---------- 各方面 F1(基于 seed0 预测) ----------
    from sklearn.metrics import f1_score, confusion_matrix
    aspect_ids_te = cache["aspect_ids"][te]
    per_aspect = {}
    for name, pred in seed0_pred.items():
        d = {}
        for ai, aname in enumerate(u.ASPECTS):
            m = aspect_ids_te == ai
            if m.sum() == 0:
                d[aname] = None; continue
            d[aname] = float(f1_score(labels[te][m], pred[m],
                                      average="macro", zero_division=0))
        per_aspect[name] = d
    results["per_aspect_f1"] = per_aspect

    # ---------- 混淆矩阵(所提模型 seed0) ----------
    cm = confusion_matrix(labels[te], seed0_pred["BERT-BiLSTM-Attention"],
                          labels=[0, 1, 2])
    results["confusion_matrix"] = {
        "labels": u.LABELS, "matrix": cm.tolist(),
        "model": "BERT-BiLSTM-Attention",
    }

    # ---------- 显著性检验(所提 vs 最强基线, gold macro-F1, 配对 t) ----------
    from scipy import stats as sst
    proposed = np.array(macro_f1_seeds["BERT-BiLSTM-Attention"])
    baselines = {k: np.array(v) for k, v in macro_f1_seeds.items()
                 if k != "BERT-BiLSTM-Attention"}
    best_base = max(baselines, key=lambda k: baselines[k].mean())
    t_stat, p_val = sst.ttest_rel(proposed, baselines[best_base])
    results["significance"] = {
        "metric": "gold_macro_f1", "proposed": "BERT-BiLSTM-Attention",
        "best_baseline": best_base,
        "proposed_mean": float(proposed.mean()),
        "best_baseline_mean": float(baselines[best_base].mean()),
        "diff": float(proposed.mean() - baselines[best_base].mean()),
        "paired_t": float(t_stat), "p_value": float(p_val),
        "n_seeds": len(SEEDS),
    }
    results["macro_f1_seeds"] = {k: [float(x) for x in v]
                                 for k, v in macro_f1_seeds.items()}

    # ---------- 消融实验(gold macro-F1, 3 种子取均值) ----------
    # full = BERT-BiLSTM-Attention (已有); 追加 -Attention / -BiLSTM / -both
    abl_seeds = SEEDS[:3]
    def run_bert_variant(use_lstm, use_att):
        vals = []
        for seed in abl_seeds:
            u.set_seed(seed)
            mdl = M.BertBiLSTMAtt(use_lstm=use_lstm, use_att=use_att)
            mdl, _ = T.train_bert_head(mdl, cache, tr, va, te, class_w, seed, use_seq=True)
            pg = T.predict(mdl, cache, te_gold, True)
            vals.append(u.metrics_full(labels[te_gold], pg)["macro_f1"])
        return float(np.mean(vals)), float(np.std(vals))
    print("\n===== 消融实验 =====", flush=True)
    full_m = float(np.mean(macro_f1_seeds["BERT-BiLSTM-Attention"][:3]))
    noatt_m, noatt_s = run_bert_variant(True, False)   # 去注意力 = BERT-BiLSTM
    print(f"  -Attention done mF1={noatt_m:.4f}", flush=True)
    nolstm_m, nolstm_s = run_bert_variant(False, True)  # 去 BiLSTM
    print(f"  -BiLSTM done mF1={nolstm_m:.4f}", flush=True)
    noboth_m, noboth_s = run_bert_variant(False, False)  # 都去(=均值池化线性)
    print(f"  -both done mF1={noboth_m:.4f}", flush=True)
    results["ablation"] = {
        "metric": "gold_macro_f1", "n_seeds": len(abl_seeds),
        "full": {"mean": full_m},
        "wo_attention": {"mean": noatt_m, "std": noatt_s,
                         "drop": full_m - noatt_m},
        "wo_bilstm": {"mean": nolstm_m, "std": nolstm_s,
                      "drop": full_m - nolstm_m},
        "wo_both": {"mean": noboth_m, "std": noboth_s,
                    "drop": full_m - noboth_m},
    }

    # ---------- 类别权重消融(加权 vs 不加权对 neutral F1 影响) ----------
    print("\n===== 类别权重消融 =====", flush=True)
    def run_weight_variant(weights):
        u.set_seed(42)
        mdl = M.BertBiLSTMAtt(use_lstm=True, use_att=True)
        mdl, _ = T.train_bert_head(mdl, cache, tr, va, te, weights, 42, use_seq=True)
        pg = T.predict(mdl, cache, te_gold, True)
        return u.metrics_full(labels[te_gold], pg)
    m_weighted = run_weight_variant(class_w)
    m_unweighted = run_weight_variant(np.ones(3))
    results["class_weight_ablation"] = {
        "metric": "gold subset",
        "weighted": {"macro_f1": m_weighted["macro_f1"],
                     "neutral_f1": m_weighted["per_class"]["neutral"]["f1"],
                     "accuracy": m_weighted["accuracy"]},
        "unweighted": {"macro_f1": m_unweighted["macro_f1"],
                       "neutral_f1": m_unweighted["per_class"]["neutral"]["f1"],
                       "accuracy": m_unweighted["accuracy"]},
    }

    # ---------- LLM few-shot 基线(金标准子集) ----------
    print("\n===== LLM few-shot 基线 =====", flush=True)
    try:
        from llm_baseline import run_llm_baseline
        df_gold_test = df.iloc[te_gold].copy()
        llm_res = run_llm_baseline(df_gold_test, max_n=400)
    except Exception as e:
        print("[llm] exception:", repr(e)[:200]); llm_res = None
    results["llm_baseline"] = llm_res if llm_res else {
        "available": False,
        "note": "LLM few-shot 基线不可用(API 未配置或调用失败), 未编造结果。",
    }

    u.save_json(results, os.path.join(FIG, "all_results.json"))
    u.save_json(results, os.path.join(FIG, "all_results.json"))
    # 中间落盘, 供后续脚本
    np.save(os.path.join(u.CACHE_DIR, "test_idx.npy"), te)
    if attn_alpha_store is not None:
        np.save(os.path.join(u.CACHE_DIR, "attn_sample_idx.npy"), attn_alpha_store[0])
        np.save(os.path.join(u.CACHE_DIR, "attn_sample_alpha.npy"), attn_alpha_store[1])
    print(f"\n[main] 主训练完成 {time.time()-t_start:.0f}s", flush=True)
    return results, cache, df, (tr, va, te, te_gold), class_w


if __name__ == "__main__":
    main()
