# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""检测不合理数值(AI 编造/过拟合/泄漏)。"""
import json, os, sys, math

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
results = {}
for f in sorted(os.listdir(FIG)):
    if f.endswith("_results.json") or f == "all_results.json":
        try:
            with open(os.path.join(FIG, f), "r", encoding="utf-8") as fh:
                results[f] = json.load(fh)
        except Exception:
            pass

errors, warnings, suspicious = [], [], []


def check_unrealistic(name, val):
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return
    key = name.lower()
    if any(w in key for w in ["macro_f1", "weighted_f1", "accuracy", "f1", "auc", "precision", "recall"]):
        if val > 0.99:
            suspicious.append(f"🚩 {name} = {val:.4f}（>0.99），确认是否过拟合/泄漏")
        elif val < 0:
            errors.append(f"❌ {name} = {val} 负值")
    if "p_value" in key or "pvalue" in key:
        if val > 1:
            errors.append(f"❌ {name} = {val} p 值 > 1 不可能")


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if math.isnan(obj):
            errors.append(f"❌ {path} 为 NaN")
        elif math.isinf(obj):
            errors.append(f"❌ {path} 为 Inf")
        check_unrealistic(path, obj)
    elif isinstance(obj, int):
        check_unrealistic(path, obj)


for fname, data in results.items():
    walk(data, fname)

for e in errors:
    print(e)
for s in suspicious:
    print(s)
if errors:
    print(f"\n❌ {len(errors)} 个硬错误 — 必须修复")
    sys.exit(1)
if suspicious:
    print(f"\n🚩 {len(suspicious)} 个可疑'完美'结果 — 逐条确认训练/测试分离、样本量、泄漏")
else:
    print("✅ 所有数值通过 sanity check")
