# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""公共工具: 随机种子、评价指标、类别权重、JSON 落盘。"""
# 自举模块路径(让 sibling import 不依赖调用方式)
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import random
import numpy as np

# 标签体系(三分类)
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# 7 类方面(固定顺序,与 README 一致)
ASPECTS = [
    "A1_UI_Design", "A2_Functionality", "A3_Cost_Subscription",
    "A4_Privacy_Security", "A5_Performance", "A6_Customer_Support",
    "A7_Content_Quality",
]
ASPECT2ID = {a: i for i, a in enumerate(ASPECTS)}
# 方面的自然语言文本(用于 BERT 拼接 / 方面感知注意力 query)
ASPECT_TEXT = {
    "A1_UI_Design": "ui design",
    "A2_Functionality": "functionality",
    "A3_Cost_Subscription": "cost and subscription",
    "A4_Privacy_Security": "privacy and security",
    "A5_Performance": "performance",
    "A6_Customer_Support": "customer support",
    "A7_Content_Quality": "content quality",
}

DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
FIG_DIR = os.path.join(os.path.dirname(_HERE), "figures")


def set_seed(seed):
    """固定随机种子保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compute_class_weights(labels, num_classes=3):
    """balanced 策略类别权重: n_samples / (n_classes * bincount)。"""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(labels) / (num_classes * counts)
    return w


def metrics_full(y_true, y_pred, num_classes=3):
    """返回 accuracy / macro-F1 / weighted-F1 / 每类 P R F1。"""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {}
    for i in range(num_classes):
        per_class[ID2LABEL[i]] = {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(sup[i]),
        }
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[save] {path} ({os.path.getsize(path)} bytes)")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
