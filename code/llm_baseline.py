# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""LLM few-shot 基线: 用 OpenAI 兼容接口对金标准子集做方面级情感三分类。
若无 API key 或调用失败, 返回 None(主流程标记为不可用, 绝不编造结果)。"""
import os, sys, json, time, re
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import utils as u

FEWSHOT = """You are an aspect-based sentiment classifier for digital-health app reviews.
Given a REVIEW and an ASPECT, output the sentiment of the review toward that aspect.
Answer with exactly one word: negative, neutral, or positive.

Examples:
REVIEW: "the app keeps crashing every time i open it, totally unusable"
ASPECT: performance -> negative
REVIEW: "clean layout and very easy to navigate, love the design"
ASPECT: ui design -> positive
REVIEW: "it works but the subscription is way too expensive for what it offers"
ASPECT: cost and subscription -> negative
REVIEW: "the customer support team replied quickly and solved my issue"
ASPECT: customer support -> positive
REVIEW: "it's okay, does what it says, nothing special"
ASPECT: functionality -> neutral
"""


def run_llm_baseline(df_gold, max_n=400, model="gpt-4o-mini"):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[llm] no OPENAI_API_KEY -> skip")
        return None
    try:
        from openai import OpenAI
    except Exception:
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "openai"], check=True)
            from openai import OpenAI
        except Exception as e:
            print("[llm] openai sdk unavailable:", repr(e)[:150])
            return None

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)

    sub = df_gold.sample(n=min(max_n, len(df_gold)), random_state=42).reset_index(drop=True)
    y_true, y_pred = [], []
    t0 = time.time()
    for i, row in sub.iterrows():
        asp = u.ASPECT_TEXT[row["aspect"]]
        text = str(row["clean_text"])[:400]
        prompt = f'{FEWSHOT}\nNow classify:\nREVIEW: "{text}"\nASPECT: {asp} ->'
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=5,
            )
            ans = resp.choices[0].message.content.strip().lower()
        except Exception as e:
            if i == 0:
                print("[llm] api call failed:", repr(e)[:200])
                return None
            ans = "neutral"
        m = re.search(r"negative|neutral|positive", ans)
        pred = m.group(0) if m else "neutral"
        y_pred.append(u.LABEL2ID[pred])
        y_true.append(u.LABEL2ID[row["aspect_sentiment"]])
        if i % 50 == 0:
            print(f"[llm] {i}/{len(sub)} elapsed={time.time()-t0:.0f}s", flush=True)
    res = u.metrics_full(np.array(y_true), np.array(y_pred))
    res["n"] = int(len(sub)); res["model"] = model
    print(f"[llm] done n={len(sub)} macroF1={res['macro_f1']:.4f}")
    return res
