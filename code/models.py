# -*- coding: utf-8 -*-
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

"""模型定义。
从零训练(词嵌入): TextCNN, BiLSTM-Attention (ATAE 风格, 方面嵌入)。
BERT 特征头(冻结 BERT 之上): BERT+Linear, BERT+BiLSTM, BERT+BiLSTM+Attention。
方面感知注意力: 用方面向量作为 query 对序列做打分归一化。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as u

N_CLASS = 3
N_ASPECT = len(u.ASPECTS)


class AspectAttention(nn.Module):
    """方面感知注意力: score_t = v^T tanh(W_h h_t + W_a a)。返回上下文与权重。"""
    def __init__(self, hid, asp_dim):
        super().__init__()
        self.Wh = nn.Linear(hid, hid)
        self.Wa = nn.Linear(asp_dim, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, H, aspect_vec, mask=None):
        # H:(B,L,hid) aspect_vec:(B,asp_dim) mask:(B,L)
        a = self.Wa(aspect_vec).unsqueeze(1)         # (B,1,hid)
        e = self.v(torch.tanh(self.Wh(H) + a)).squeeze(-1)  # (B,L)
        if mask is not None:
            # 保底: 若某行被完全屏蔽(全 pad), 放开首位, 防止 softmax 全 -inf -> NaN。
            m = mask.clone()
            all_masked = (m.sum(dim=1) < 0.5)
            if all_masked.any():
                m[all_masked, 0] = 1.0
            e = e.masked_fill(m < 0.5, float("-inf"))
        alpha = torch.softmax(e, dim=1)              # (B,L)
        ctx = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)   # (B,hid)
        return ctx, alpha


# ---------- 从零训练模型 ----------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=100, ch=100, asp_emb=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.aspect_emb = nn.Embedding(N_ASPECT, asp_emb)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, ch, k, padding=k // 2) for k in (3, 4, 5)])
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(ch * 3 + asp_emb, N_CLASS)

    def forward(self, x, aspect_id):
        z = self.embedding(x).transpose(1, 2)         # (B,emb,L)
        feats = [F.relu(c(z)).max(dim=2).values for c in self.convs]
        h = torch.cat(feats + [self.aspect_emb(aspect_id)], dim=1)
        return self.fc(self.drop(h)), None


class BiLSTMAttentionScratch(nn.Module):
    """ATAE-LSTM 风格: 词嵌入 -> BiLSTM -> 方面感知注意力。"""
    def __init__(self, vocab_size, emb=100, hid=128, asp_emb=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.aspect_emb = nn.Embedding(N_ASPECT, asp_emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.att = AspectAttention(hid * 2, asp_emb)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hid * 2, N_CLASS)

    def forward(self, x, aspect_id):
        mask = (x != 0).float()
        z = self.embedding(x)
        H, _ = self.lstm(z)
        ctx, alpha = self.att(H, self.aspect_emb(aspect_id), mask)
        return self.fc(self.drop(ctx)), alpha


# ---------- BERT 特征头 ----------
class BertLinear(nn.Module):
    """冻结 BERT pooled 向量 + 方面嵌入 -> 线性分类。"""
    def __init__(self, hid=768, asp_emb=64):
        super().__init__()
        self.aspect_emb = nn.Embedding(N_ASPECT, asp_emb)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hid + asp_emb, N_CLASS)

    def forward(self, pooled, aspect_id, **kw):
        h = torch.cat([pooled, self.aspect_emb(aspect_id)], dim=1)
        return self.fc(self.drop(h)), None


class BertBiLSTMAtt(nn.Module):
    """冻结 BERT 序列 -> (可选)BiLSTM -> (可选)方面感知注意力 -> 分类。
    use_lstm / use_att 控制消融。方面 query 用真实 BERT 方面向量。"""
    def __init__(self, hid=768, lstm_hid=128, asp_dim=768,
                 use_lstm=True, use_att=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_att = use_att
        seq_dim = hid
        if use_lstm:
            self.lstm = nn.LSTM(hid, lstm_hid, batch_first=True,
                                bidirectional=True)
            seq_dim = lstm_hid * 2
        if use_att:
            self.att = AspectAttention(seq_dim, asp_dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(seq_dim, N_CLASS)

    def forward(self, seq, aspect_vec, mask=None, **kw):
        H = seq
        if self.use_lstm:
            H, _ = self.lstm(H)
        if self.use_att:
            ctx, alpha = self.att(H, aspect_vec, mask)
        else:
            # 掩码均值池化
            if mask is not None:
                m = mask.unsqueeze(-1)
                ctx = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                ctx = H.mean(1)
            alpha = None
        return self.fc(self.drop(ctx)), alpha
