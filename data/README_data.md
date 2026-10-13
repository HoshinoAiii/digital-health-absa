# data/absa_long.csv — 字段说明

方面级长表（aspect-level long table），**每行一个 (评论, 方面) 对**。

| 字段 | 类型 | 说明 |
|---|---|---|
| `review_id` | str | 评论唯一 ID |
| `platform` | str | 平台：`Google Play` / `Apple App Store` |
| `app_name` | str | App 名称 |
| `rating` | int | 原始评分（1–5 星） |
| `lang` | str | 语言：`en` / `zh` / `unknown` |
| `aspect` | str | 方面类别，取值见下（七类） |
| `aspect_sentiment` | str | 该方面的情感极性：`negative` / `neutral` / `positive` |
| `confidence` | str | 标注置信度：`high`（窗口 ±5 token 内命中情感词）/ `low`（无情感词，回退到文档级标签） |
| `doc_sentiment` | str | 文档级情感（用于对比） |
| `clean_text` | str | 清洗后评论文本（截断） |

## 方面类别（7 类）

| 代码 | 英文 | 中文 |
|---|---|---|
| `A1_UI_Design` | UI Design | 界面设计 |
| `A2_Functionality` | Functionality | 功能实用性 |
| `A3_Cost_Subscription` | Cost / Subscription | 费用订阅 |
| `A4_Privacy_Security` | Privacy / Security | 隐私安全 |
| `A5_Performance` | Performance | 性能稳定 |
| `A6_Customer_Support` | Customer Support | 客服支持 |
| `A7_Content_Quality` | Content Quality | 内容质量 |

## 规模

- 本文件共 **35,717** 行（有效方面-情感对）。
- 经 `code/data_prep.py` 过滤（方面合法 ∧ 标签合法 ∧ 文本非空）后得 **31,349** 行，
  即论文全文所用样本量。
- 按 `review_id` 分组 8:1:1 划分：**train 25,066 / val 3,182 / test 3,101**。
- 七类方面分布（31,349）：A2 8,297 / A3 6,217 / A7 4,878 / A6 4,358 /
  A1 3,517 / A5 2,636 / A4 1,446。

## 标签映射

| rating | 三分类 | |
|---|---|---|
| 1–2 星 | 0 | negative |
| 3 星 | 1 | neutral |
| 4–5 星 | 2 | positive |

> 情感标签为**方面级规则标注**结果（词典 + 窗口），非人工逐一标注；
> 人工核验见论文第 5 章。
