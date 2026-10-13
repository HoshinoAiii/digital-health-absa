# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""可续跑主实验编排(抗中断): 每个 (模型,种子) 训练完立刻把 test/gold 预测
落盘到 data/cache/preds/, 断线重跑自动跳过已完成项; 全部预测就绪后再从
缓存预测计算所有指标/消融/显著性/各方面F1/混淆矩阵/注意力, 写 figures/*.json。

修复: 从零 BiLSTM-Attention 之前 lr=1e-3 坍缩(全负类), 降到 5e-4 + 15 epoch。"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import torch

import utils as u
import models as M
import train as T
from data_prep import PROC, load_and_split, encode_wordids

SEEDS = [42, 1, 7, 123, 2024]
ABL_SEEDS = [42, 1, 7]
PRED_DIR = os.path.join(u.CACHE_DIR, "preds")
FIG = u.FIG_DIR


def load_cache():
    c = u.CACHE_DIR
    return {
        "seq": np.load(os.path.join(c, "bert_seq.npy"), mmap_mode="r"),
        "pooled": np.load(os.path.join(c, "bert_pooled.npy")),
        "mask": np.load(os.path.join(c, "bert_mask.npy")),
        "aspect_vec": np.load(os.path.join(c, "aspect_vec.npy")),
        "labels": np.load(os.path.join(c, "labels.npy")),
        "aspect_ids": np.load(os.path.join(c, "aspect_ids.npy")),
        "is_gold": np.load(os.path.join(c, "is_gold.npy")),
        "split": np.load(os.path.join(c, "split.npy")),
    }


def _p(tag):
    return os.path.join(PRED_DIR, tag)


def cached(tag):
    return (os.path.exists(_p(tag + "_te.npy")) and
            os.path.exists(_p(tag + "_gold.npy")))


def save_pred(tag, pred_te, pred_gold, hist=None, alpha=None, alpha_idx=None):
    os.makedirs(PRED_DIR, exist_ok=True)
    np.save(_p(tag + "_te.npy"), pred_te)
    np.save(_p(tag + "_gold.npy"), pred_gold)
    if hist is not None:
        with open(_p(tag + "_hist.json"), "w", encoding="utf-8") as f:
            json.dump(hist, f)
    if alpha is not None:
        np.save(_p(tag + "_alpha.npy"), alpha)
        np.save(_p(tag + "_alphaidx.npy"), alpha_idx)


def load_pred(tag):
    return np.load(_p(tag + "_te.npy")), np.load(_p(tag + "_gold.npy"))
def train_one(name, seed, cache, splits, Xword, Aword, yall, class_w, vocab,
              want_alpha=False):
    """训练单个 (模型,种子), 返回 (pred_te, pred_gold, hist, alpha, alpha_idx)。"""
    tr, va, te, te_gold = splits
    u.set_seed(seed)
    alpha = alpha_idx = None
    if name == "TextCNN":
        mdl = M.TextCNN(len(vocab))
        mdl, hist = T.train_scratch(mdl, Xword[tr], Aword[tr], yall[tr],
                                    Xword[va], Aword[va], yall[va], class_w, seed)
        pred_te = T.predict_scratch(mdl, Xword[te], Aword[te])
        pred_gold = T.predict_scratch(mdl, Xword[te_gold], Aword[te_gold])
    elif name == "BiLSTM-Attention":
        mdl = M.BiLSTMAttentionScratch(len(vocab))
        # 修复坍缩: 更低 lr + 更多 epoch + 更大 patience
        mdl, hist = T.train_scratch(mdl, Xword[tr], Aword[tr], yall[tr],
                                    Xword[va], Aword[va], yall[va], class_w, seed,
                                    epochs=15, lr=5e-4, patience=5)
        pred_te = T.predict_scratch(mdl, Xword[te], Aword[te])
        pred_gold = T.predict_scratch(mdl, Xword[te_gold], Aword[te_gold])
    else:
        # BERT 特征头系列
        cfg = {
            "BERT": dict(use_seq=False),
            "BERT-BiLSTM": dict(use_seq=True, use_lstm=True, use_att=False),
            "BERT-BiLSTM-Attention": dict(use_seq=True, use_lstm=True, use_att=True),
            "ABL-wo_bilstm": dict(use_seq=True, use_lstm=False, use_att=True),
            "ABL-wo_both": dict(use_seq=True, use_lstm=False, use_att=False),
            "UNWEIGHTED": dict(use_seq=True, use_lstm=True, use_att=True),
        }[name]
        use_seq = cfg["use_seq"]
        if use_seq:
            mdl = M.BertBiLSTMAtt(use_lstm=cfg["use_lstm"], use_att=cfg["use_att"])
        else:
            mdl = M.BertLinear()
        mdl, hist = T.train_bert_head(mdl, cache, tr, va, te, class_w, seed,
                                      use_seq=use_seq)
        pred_te = T.predict(mdl, cache, te, use_seq)
        pred_gold = T.predict(mdl, cache, te_gold, use_seq)
        if want_alpha and cfg.get("use_att"):
            a_idx = te[:60]
            _, alpha = T.predict(mdl, cache, a_idx, True, return_alpha=True)
            alpha_idx = a_idx
    return pred_te, pred_gold, hist, alpha, alpha_idx


def ensure_all_preds(cache, splits, Xword, Aword, yall, class_w, vocab):
    """确保所有需要的 (模型,种子) 预测已缓存; 未缓存的训练并落盘。"""
    tr = splits[0]
    class_w_unw = np.ones(3)
    jobs = []
    main_models = ["TextCNN", "BiLSTM-Attention", "BERT",
                   "BERT-BiLSTM", "BERT-BiLSTM-Attention"]
    for name in main_models:
        for seed in SEEDS:
            jobs.append((name, seed, class_w))
    for name in ["ABL-wo_bilstm", "ABL-wo_both"]:
        for seed in ABL_SEEDS:
            jobs.append((name, seed, class_w))
    jobs.append(("UNWEIGHTED", 42, class_w_unw))

    for name, seed, cw in jobs:
        tag = f"{name}_s{seed}"
        if cached(tag):
            print(f"[skip] {tag} (cached)", flush=True)
            continue
        t0 = time.time()
        want_alpha = (name == "BERT-BiLSTM-Attention" and seed == 42)
        pte, pgo, hist, alpha, aidx = train_one(
            name, seed, cache, splits, Xword, Aword, yall, cw, vocab, want_alpha)
        m = u.metrics_full(cache["labels"][splits[3]], pgo)
        save_pred(tag, pte, pgo, hist, alpha, aidx)
        print(f"[done] {tag} gold_mF1={m['macro_f1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)


def agg(vals):
    vals = [float(v) for v in vals]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "runs": vals}


def compute_results(cache, splits, df):
    """全部预测就绪后, 从缓存预测计算所有指标并写 JSON。"""
    from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
    from scipy import stats as sst
    tr, va, te, te_gold = splits
    y_te = cache["labels"][te]; y_gold = cache["labels"][te_gold]
    class_w = u.compute_class_weights(cache["labels"][tr])
    results = {"meta": {
        "seeds": SEEDS, "ablation_seeds": ABL_SEEDS,
        "class_weights": {u.ID2LABEL[i]: float(class_w[i]) for i in range(3)},
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "n_test": int(len(te)), "n_test_gold": int(len(te_gold)),
        "note": ("BERT 作为冻结特征抽取器(feature-based; CPU 环境无 GPU, "
                 "端到端微调 5 模型x5 种子不可行), 下游轻量头训练 5 种子。"
                 "gold=high-confidence 子集(窗口内含情感词的方面-情感对)。"),
    }}

    # 标签可信度(方面级弱监督 vs 文档级 自动一致性)
    mm = df["aspect_sentiment"].notna() & df["doc_sentiment"].notna()
    a_lab = df.loc[mm, "aspect_sentiment"].map(u.LABEL2ID).values
    d_lab = df.loc[mm, "doc_sentiment"].map(u.LABEL2ID).values
    hi = df["confidence"] == "high"
    mm_hi = mm & hi
    a_hi = df.loc[mm_hi, "aspect_sentiment"].map(u.LABEL2ID).values
    d_hi = df.loc[mm_hi, "doc_sentiment"].map(u.LABEL2ID).values
    results["label_reliability"] = {
        "definition": ("aspect-level(弱监督) vs doc-level 标签自动一致性(Cohen's κ); "
                       "非人工标注者间一致性, 无人工重标注资源时的可测代理"),
        "agreement_all": float((a_lab == d_lab).mean()),
        "cohen_kappa_all": float(cohen_kappa_score(a_lab, d_lab)),
        "agreement_highconf": float((a_hi == d_hi).mean()),
        "cohen_kappa_highconf": float(cohen_kappa_score(a_hi, d_hi)),
        "n_all": int(mm.sum()), "n_highconf": int(mm_hi.sum()),
    }

    # 主结果: 各模型 auto/gold 的 acc/macroF1/weightedF1 (mean±std over seeds)
    main_models = ["TextCNN", "BiLSTM-Attention", "BERT",
                   "BERT-BiLSTM", "BERT-BiLSTM-Attention"]
    per_model = {}
    macro_gold_seeds = {}
    seed0_pred_te = {}
    for name in main_models:
        auto_acc = []; auto_mf1 = []; auto_wf1 = []
        gold_acc = []; gold_mf1 = []; gold_wf1 = []
        for si, seed in enumerate(SEEDS):
            pte, pgo = load_pred(f"{name}_s{seed}")
            ma = u.metrics_full(y_te, pte); mg = u.metrics_full(y_gold, pgo)
            auto_acc.append(ma["accuracy"]); auto_mf1.append(ma["macro_f1"]); auto_wf1.append(ma["weighted_f1"])
            gold_acc.append(mg["accuracy"]); gold_mf1.append(mg["macro_f1"]); gold_wf1.append(mg["weighted_f1"])
            if si == 0:
                seed0_pred_te[name] = pte
                per_class_gold0 = mg["per_class"]
        macro_gold_seeds[name] = gold_mf1
        per_model[name] = {
            "auto": {"accuracy": agg(auto_acc), "macro_f1": agg(auto_mf1),
                     "weighted_f1": agg(auto_wf1)},
            "gold": {"accuracy": agg(gold_acc), "macro_f1": agg(gold_mf1),
                     "weighted_f1": agg(gold_wf1)},
            "per_class_gold_seed0": per_class_gold0,
        }
    results["models"] = per_model
    results["macro_f1_gold_seeds"] = {k: [float(x) for x in v]
                                      for k, v in macro_gold_seeds.items()}

    # 显著性: 配对 t (跨种子 gold macro-F1) + 配对 bootstrap(seed0 gold 预测)
    proposed = "BERT-BiLSTM-Attention"
    prop = np.array(macro_gold_seeds[proposed])
    baselines = {k: np.array(v) for k, v in macro_gold_seeds.items() if k != proposed}
    best_base = max(baselines, key=lambda k: baselines[k].mean())
    t_stat, p_t = sst.ttest_rel(prop, baselines[best_base])
    pa = load_pred(f"{proposed}_s42")[1]
    pb = load_pred(f"{best_base}_s42")[1]
    rng = np.random.RandomState(12345); n = len(y_gold); diffs = []
    for _ in range(2000):
        bi = rng.randint(0, n, n)
        diffs.append(f1_score(y_gold[bi], pa[bi], average="macro", zero_division=0) -
                     f1_score(y_gold[bi], pb[bi], average="macro", zero_division=0))
    diffs = np.array(diffs)
    results["significance"] = {
        "metric": "gold_macro_f1", "proposed": proposed, "best_baseline": best_base,
        "proposed_mean": float(prop.mean()), "proposed_std": float(prop.std()),
        "best_baseline_mean": float(baselines[best_base].mean()),
        "diff": float(prop.mean() - baselines[best_base].mean()),
        "paired_t": float(t_stat), "p_value_ttest": float(p_t),
        "bootstrap_mean_diff": float(diffs.mean()),
        "bootstrap_ci95": [float(np.percentile(diffs, 2.5)),
                           float(np.percentile(diffs, 97.5))],
        "p_value_bootstrap": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "n_seeds": len(SEEDS), "n_bootstrap": 2000,
    }

    # 各方面 F1 (seed0 test 预测)
    aids_te = cache["aspect_ids"][te]
    per_aspect = {}
    for name in main_models:
        d = {}
        for ai, an in enumerate(u.ASPECTS):
            m = aids_te == ai
            d[an] = (float(f1_score(y_te[m], seed0_pred_te[name][m],
                     average="macro", zero_division=0)) if m.sum() else None)
        per_aspect[name] = d
    results["per_aspect_f1"] = per_aspect

    # 混淆矩阵(所提模型 seed0, test)
    cm = confusion_matrix(y_te, seed0_pred_te[proposed], labels=[0, 1, 2])
    results["confusion_matrix"] = {"labels": u.LABELS, "matrix": cm.tolist(),
                                   "model": proposed, "subset": "auto_test"}

    # 消融(gold macro-F1, ABL_SEEDS 均值): full=proposed, wo_att=BERT-BiLSTM,
    # wo_bilstm=ABL-wo_bilstm, wo_both=ABL-wo_both
    def gold_mf1_mean(name):
        vs = [u.metrics_full(y_gold, load_pred(f"{name}_s{s}")[1])["macro_f1"]
              for s in ABL_SEEDS]
        return float(np.mean(vs)), float(np.std(vs))
    full_m, full_s = gold_mf1_mean(proposed)
    noatt_m, noatt_s = gold_mf1_mean("BERT-BiLSTM")
    nolstm_m, nolstm_s = gold_mf1_mean("ABL-wo_bilstm")
    noboth_m, noboth_s = gold_mf1_mean("ABL-wo_both")
    results["ablation"] = {
        "metric": "gold_macro_f1", "n_seeds": len(ABL_SEEDS),
        "full": {"mean": full_m, "std": full_s},
        "wo_attention": {"mean": noatt_m, "std": noatt_s, "drop": full_m - noatt_m},
        "wo_bilstm": {"mean": nolstm_m, "std": nolstm_s, "drop": full_m - nolstm_m},
        "wo_both": {"mean": noboth_m, "std": noboth_s, "drop": full_m - noboth_m},
    }

    # 类别权重消融(seed42, gold): weighted=proposed_s42, unweighted=UNWEIGHTED_s42
    mw = u.metrics_full(y_gold, load_pred(f"{proposed}_s42")[1])
    mu = u.metrics_full(y_gold, load_pred("UNWEIGHTED_s42")[1])
    results["class_weight_ablation"] = {
        "metric": "gold subset (seed=42)",
        "weighted": {"macro_f1": mw["macro_f1"], "accuracy": mw["accuracy"],
                     "neutral_f1": mw["per_class"]["neutral"]["f1"],
                     "neutral_recall": mw["per_class"]["neutral"]["recall"]},
        "unweighted": {"macro_f1": mu["macro_f1"], "accuracy": mu["accuracy"],
                       "neutral_f1": mu["per_class"]["neutral"]["f1"],
                       "neutral_recall": mu["per_class"]["neutral"]["recall"]},
    }

    # 训练曲线(seed0)
    curves = {}
    for name in main_models:
        hp = _p(f"{name}_s42_hist.json")
        if os.path.exists(hp):
            with open(hp, encoding="utf-8") as f:
                curves[name] = json.load(f)
    results["training_curves"] = curves

    # 注意力样例(proposed seed42)
    ap = _p(f"{proposed}_s42_alpha.npy")
    samples = []
    if os.path.exists(ap):
        import re
        alpha = np.load(ap); a_idx = np.load(_p(f"{proposed}_s42_alphaidx.npy"))
        pte = load_pred(f"{proposed}_s42")[0]
        # a_idx 是相对 cache 全量的 test 前 60; 需其在 te 中的位置取预测
        te_pos = {int(g): k for k, g in enumerate(te)}
        tokre = re.compile(r"[a-z0-9']+")
        sel = 0
        for j, gi in enumerate(a_idx):
            if sel >= 6:
                break
            true = int(cache["labels"][gi])
            pred = int(pte[te_pos[int(gi)]])
            if true == pred and true in (0, 2):
                text = str(df.iloc[gi]["clean_text"])
                toks = tokre.findall(text.lower())[:30]
                if len(toks) < 4:
                    continue
                w = alpha[j][:len(toks)]
                samples.append({
                    "aspect": u.ASPECTS[int(cache["aspect_ids"][gi])],
                    "true": u.ID2LABEL[true], "pred": u.ID2LABEL[pred],
                    "tokens": toks, "weights": [float(x) for x in w]})
                sel += 1
    results["attention_samples"] = {"samples": samples}

    # LLM few-shot 基线(可选)
    try:
        from llm_baseline import run_llm_baseline
        llm = run_llm_baseline(df.iloc[te_gold].copy(), max_n=400)
    except Exception as e:
        print("[llm] exception:", repr(e)[:150]); llm = None
    results["llm_baseline"] = llm if llm else {
        "available": False,
        "note": "LLM few-shot 基线不可用(OPENAI_API_KEY 未配置或调用失败), 未编造结果。"}

    u.save_json(results, os.path.join(FIG, "all_results.json"))
    # 拆分若干独立 JSON, 方便 paper-figure 读取
    u.save_json(results["ablation"], os.path.join(FIG, "ablation_results.json"))
    u.save_json(results["confusion_matrix"], os.path.join(FIG, "confusion_matrix.json"))
    u.save_json(results["per_aspect_f1"], os.path.join(FIG, "per_aspect_f1.json"))
    u.save_json({"models": {k: v["gold"] for k, v in per_model.items()},
                 "significance": results["significance"]},
                os.path.join(FIG, "model_comparison.json"))
    return results


def main():
    t0 = time.time()
    if not os.path.exists(PROC):
        load_and_split()
    df = pd.read_pickle(PROC)
    cache = load_cache()
    sp = cache["split"]
    tr = np.where(sp == 0)[0]; va = np.where(sp == 1)[0]; te = np.where(sp == 2)[0]
    te_gold = te[cache["is_gold"][te] == 1]
    splits = (tr, va, te, te_gold)
    class_w = u.compute_class_weights(cache["labels"][tr])
    print(f"[run_all] train={len(tr)} val={len(va)} test={len(te)} "
          f"gold={len(te_gold)} class_w={np.round(class_w,3)}", flush=True)

    vocab = u.load_json(os.path.join(u.CACHE_DIR, "vocab.json"))
    Xword, Aword = encode_wordids(df["clean_text"].tolist(),
                                  df["aspect_id"].values, vocab, max_len=32)
    yall = df["label"].values

    ensure_all_preds(cache, splits, Xword, Aword, yall, class_w, vocab)
    print(f"[run_all] 所有预测就绪, 计算指标...", flush=True)
    compute_results(cache, splits, df)
    print(f"[run_all] 全部完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

