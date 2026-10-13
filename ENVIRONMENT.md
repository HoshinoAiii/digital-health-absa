# Environment

记录论文实验所用软件环境，供复现者对照。

## Python

| 项目 | 论文实验环境 | 复现验证环境 |
|---|---|---|
| Python | 3.10.11 | 3.13.14 |
| OS | Windows 10/11 | Windows 10/11 |

## 关键依赖

| 库 | 论文环境 | 说明 |
|---|---|---|
| numpy | >= 1.24 | 数值计算 |
| pandas | >= 2.0 | 数据处理 |
| scipy | >= 1.10 | 统计检验（bootstrap / t-test） |
| scikit-learn | >= 1.3 | 划分、评价指标、类别权重 |
| torch | >= 2.0 (CPU) | TextCNN / BiLSTM / BERT-BiLSTM-Attention |
| transformers | >= 4.30 | bert-base-uncased 编码 |
| matplotlib | >= 3.7 | 图 |
| openai | >= 1.0 | LLM few-shot 基线（可选） |
| nltk | 3.10.1 | 英文分词（预处理链） |
| jieba | 0.42 | 中文分词（预处理链） |

完整依赖见 `code/requirements.txt`。一键安装：

```bash
pip install -r code/requirements.txt
```

## 模型

- **预训练语言模型**：`bert-base-uncased`（HuggingFace，vocab_size = 30,522，max_length = 128）
- **池化方式**：掩码平均池化（masked mean pooling）
- **随机种子**：主实验 42；多种子稳健性实验 42 / 1 / 7 / 2023 / 2024

## 硬件

- 实验在 **CPU** 上完成（无需 GPU）。BERT 编码为一次性预处理，结果缓存于
  `data/cache/*.npy`，后续模型训练仅读取缓存特征。

## 可选：LLM 基线

`code/llm_baseline.py` 需要一个 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY=sk-...            # Windows: set OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://...      # 可选，自定义端点
```

若未设置 key，脚本会打印提示并跳过，**不会编造任何结果**（主流程标记该基线为不可用）。
