# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""训练引擎: 支持 BERT 特征头(seq/pooled) 与从零词嵌入模型。
加权交叉熵应对不平衡, Adam + 早停, 记录 train/val 曲线与 test 预测。"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn

import utils as u

DEV = torch.device("cpu")


def _batches(n, bs, shuffle, rng):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def train_bert_head(model, cache, tr, va, te, class_w, seed,
                    epochs=8, bs=256, lr=1e-3, patience=3, use_seq=False):
    """cache: dict of numpy arrays. tr/va/te: index arrays into cache."""
    rng = np.random.RandomState(seed)
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    cw = torch.tensor(class_w, dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=cw)

    labels = cache["labels"]
    aspect_ids = cache["aspect_ids"]
    aspect_vec = torch.tensor(cache["aspect_vec"], dtype=torch.float32)
    seq = cache.get("seq"); pooled = cache.get("pooled"); mask = cache.get("mask")

    def make_batch(bidx):
        y = torch.tensor(labels[bidx], dtype=torch.long, device=DEV)
        aid = torch.tensor(aspect_ids[bidx], dtype=torch.long, device=DEV)
        av = aspect_vec[aspect_ids[bidx]].to(DEV)
        if use_seq:
            X = torch.tensor(seq[bidx].astype(np.float32), device=DEV)
            M = torch.tensor(mask[bidx].astype(np.float32), device=DEV)
            return dict(seq=X, aspect_vec=av, mask=M, aspect_id=aid), y
        else:
            P = torch.tensor(pooled[bidx].astype(np.float32), device=DEV)
            return dict(pooled=P, aspect_id=aid, aspect_vec=av), y

    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for bidx in _batches(len(tr), bs, True, rng):
            b = tr[bidx]
            inp, y = make_batch(b)
            opt.zero_grad()
            logits, _ = model(**inp)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        # val
        vp = predict(model, cache, va, use_seq)
        vm = u.metrics_full(labels[va], vp)["macro_f1"]
        history["train_loss"].append(tot / max(nb, 1))
        history["val_macro_f1"].append(vm)
        if vm > best_f1:
            best_f1 = vm; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, cache, idx, use_seq, return_alpha=False):
    model.eval()
    labels = cache["labels"]; aspect_ids = cache["aspect_ids"]
    aspect_vec = torch.tensor(cache["aspect_vec"], dtype=torch.float32)
    seq = cache.get("seq"); pooled = cache.get("pooled"); mask = cache.get("mask")
    preds = []; alphas = []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            aid = torch.tensor(aspect_ids[b], dtype=torch.long)
            av = aspect_vec[aspect_ids[b]]
            if use_seq:
                X = torch.tensor(seq[b].astype(np.float32))
                M = torch.tensor(mask[b].astype(np.float32))
                logits, al = model(seq=X, aspect_vec=av, mask=M,
                                   aspect_id=aid)
            else:
                P = torch.tensor(pooled[b].astype(np.float32))
                logits, al = model(pooled=P, aspect_id=aid, aspect_vec=av)
            preds.append(logits.argmax(1).numpy())
            if return_alpha and al is not None:
                alphas.append(al.numpy())
    preds = np.concatenate(preds)
    if return_alpha:
        return preds, (np.concatenate(alphas) if alphas else None)
    return preds


# ---------- 从零词嵌入模型 ----------
def train_scratch(model, Xtr, Atr, ytr, Xva, Ava, yva, class_w, seed,
                  epochs=12, bs=256, lr=1e-3, patience=3):
    rng = np.random.RandomState(seed)
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    cw = torch.tensor(class_w, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw)
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_state, bad = -1, None, 0
    Xtr_t = torch.tensor(Xtr); Atr_t = torch.tensor(Atr); ytr_t = torch.tensor(ytr)
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for bidx in _batches(len(Xtr), bs, True, rng):
            opt.zero_grad()
            logits, _ = model(Xtr_t[bidx], Atr_t[bidx])
            loss = crit(logits, ytr_t[bidx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        vp = predict_scratch(model, Xva, Ava)
        vm = u.metrics_full(yva, vp)["macro_f1"]
        history["train_loss"].append(tot / max(nb, 1))
        history["val_macro_f1"].append(vm)
        if vm > best_f1:
            best_f1 = vm; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_scratch(model, X, A):
    model.eval(); preds = []
    Xt = torch.tensor(X); At = torch.tensor(A)
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            logits, _ = model(Xt[s:s + 1024], At[s:s + 1024])
            preds.append(logits.argmax(1).numpy())
    return np.concatenate(preds)
