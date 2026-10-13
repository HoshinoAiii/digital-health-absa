# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""用冻结的 bert-base-uncased 把 (评论 [SEP] 方面) 编码为特征并缓存。
CPU 环境无法端到端微调 6 模型 x 5 种子, 故采用 BERT 作为固定特征抽取器
(feature-based): 缓存 last_hidden_state 序列(供 BiLSTM/Attention)与
pooled 向量(供线性头), 下游轻量头训练很快, 可跑满 5 种子。
另单独编码 7 个方面文本, 作为方面感知注意力 query。

鲁棒性: 大数组用 np.memmap 落盘(低 RAM), 记录进度可断点续跑,
限制 torch 线程数避免 CPU BLAS 段错误。"""
import os, sys, time, gc
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

import utils as u
from data_prep import PROC, load_and_split

MAX_LEN = 32
BATCH = 32
MODEL_NAME = "bert-base-uncased"
HID = 768


def _paths():
    c = u.CACHE_DIR
    return {
        "seq": os.path.join(c, "bert_seq.npy"),
        "pooled": os.path.join(c, "bert_pooled.npy"),
        "mask": os.path.join(c, "bert_mask.npy"),
        "progress": os.path.join(c, "enc_progress.txt"),
    }


@torch.no_grad()
def encode():
    if not os.path.exists(PROC):
        load_and_split()
    df = pd.read_pickle(PROC)
    n = len(df)
    print(f"[enc] loaded {n} rows", flush=True)

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    texts = df["clean_text"].tolist()
    aspect_texts = [u.ASPECT_TEXT[a] for a in df["aspect"].tolist()]
    os.makedirs(u.CACHE_DIR, exist_ok=True)
    p = _paths()

    # memmap 大数组(断点续跑: 已存在则续用)
    seq = np.lib.format.open_memmap(p["seq"], mode=("r+" if os.path.exists(p["seq"]) else "w+"),
                                    dtype=np.float16, shape=(n, MAX_LEN, HID))
    pooled = np.lib.format.open_memmap(p["pooled"], mode=("r+" if os.path.exists(p["pooled"]) else "w+"),
                                       dtype=np.float16, shape=(n, HID))
    mask_all = np.lib.format.open_memmap(p["mask"], mode=("r+" if os.path.exists(p["mask"]) else "w+"),
                                         dtype=np.float16, shape=(n, MAX_LEN))

    start = 0
    if os.path.exists(p["progress"]):
        try:
            start = int(open(p["progress"]).read().strip())
            print(f"[enc] resume from row {start}", flush=True)
        except Exception:
            start = 0

    t0 = time.time()
    for s in range(start, n, BATCH):
        e = min(s + BATCH, n)
        enc = tok(texts[s:e], aspect_texts[s:e], padding="max_length",
                  truncation=True, max_length=MAX_LEN, return_tensors="pt")
        out = model(**enc)
        lhs = out.last_hidden_state
        seq[s:e] = lhs.numpy().astype(np.float16)
        m = enc["attention_mask"].unsqueeze(-1).float()
        mean_pool = (lhs * m).sum(1) / m.sum(1).clamp(min=1e-6)
        pooled[s:e] = mean_pool.numpy().astype(np.float16)
        mask_all[s:e] = enc["attention_mask"].numpy().astype(np.float16)
        if (s // BATCH) % 40 == 0:
            seq.flush(); pooled.flush(); mask_all.flush()
            with open(p["progress"], "w") as fh:
                fh.write(str(e))
            el = time.time() - t0
            done = e - start
            eta = el / max(done, 1) * (n - e)
            print(f"[enc] {e}/{n} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
            gc.collect()

    seq.flush(); pooled.flush(); mask_all.flush()
    with open(p["progress"], "w") as fh:
        fh.write(str(n))

    # 方面 query 向量(7 个)
    aq = tok([u.ASPECT_TEXT[a] for a in u.ASPECTS], padding=True,
             truncation=True, max_length=8, return_tensors="pt")
    aout = model(**aq)
    am = aq["attention_mask"].unsqueeze(-1).float()
    aspect_vec = ((aout.last_hidden_state * am).sum(1) /
                  am.sum(1).clamp(min=1e-6)).numpy().astype(np.float32)
    np.save(os.path.join(u.CACHE_DIR, "aspect_vec.npy"), aspect_vec)
    np.save(os.path.join(u.CACHE_DIR, "labels.npy"), df["label"].values.astype(np.int64))
    np.save(os.path.join(u.CACHE_DIR, "aspect_ids.npy"), df["aspect_id"].values.astype(np.int64))
    np.save(os.path.join(u.CACHE_DIR, "is_gold.npy"), df["is_gold"].values.astype(np.int64))
    split_map = {"train": 0, "val": 1, "test": 2}
    np.save(os.path.join(u.CACHE_DIR, "split.npy"),
            df["split"].map(split_map).values.astype(np.int64))
    print(f"[enc] DONE in {time.time()-t0:.0f}s seq={seq.shape}", flush=True)


if __name__ == "__main__":
    encode()
