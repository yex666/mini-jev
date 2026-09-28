# MiniJEV: 0.6B System One 决策模型微调实战

## 1. 项目概述

### 1.1 什么是 JEV？

**JEV** 是 TypeSafe AI 于 2026年9月发布的 **"System One"决策模型**，核心理念：

- **不生成文本**，只做判断（选择题/评分/布尔值）
- 输入 `state + questions` → 输出 `类型化答案 + 概率 + 置信度`
- 单次前向传播，延迟 70~500ms，零 token 解码
- 适用于 Agent 决策、路由、审核等毫秒级结构化决策场景

### 1.2 本项目目标

基于开源 Qwen3-0.6B 骨干网络，通过 LoRA 微调，训练一个类 JEV 的轻量级决策模型：

```
输入: 上下文状态(state) + 决策问题(questions)
输出: 结构化答案（Choice/Score/Bool）+ 概率分布
```

### 1.3 参考资源

| 资源 | 说明 |
|------|------|
| [AgentJev](https://www.cnblogs.com/itech/p/23126319) | 0.6B 开源决策模型，Apache-2.0 |
| [Open-Jev/SemIf](https://baijiahao.baidu.com/s?id=1876986889617071976) | 家用 3090 可跑的开源复刻版 |
| [NanoJev](https://juejin.cn/post/7687513063491239974) | 0.6B 并行决策模型 |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 基座模型 |
| [PEFT/LoRA](https://github.com/huggingface/peft) | 参数高效微调框架 |

---

## 2. 技术原理

### 2.1 System One vs System Two

```
┌─────────────────────────────────────────────────────────────┐
│                    System Two (传统LLM)                      │
│  输入 → Token-by-Token 生成 → 完整文本 → 解析提取答案         │
│  延迟: 秒级    成本: 高    适用: 开放式生成任务               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    System One (JEV式)                        │
│  输入 → 单次前向 → 直接读取 logits → 概率分布 → 结构化答案   │
│  延迟: 毫秒级  成本: 低    适用: 分类/路由/决策任务           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心机制：Logit 提取与概率校准

JEV 的关键创新：**不做自回归解码，直接读取模型对候选选项的 logit 值**。

```python
# 伪代码示意
inputs = tokenizer(state + question, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits

# 提取候选选项的 logit
option_a_logit = logits[0, tokenizer("A").input_ids[0]]
option_b_logit = logits[0, tokenizer("B").input_ids[0]]

# Softmax 转概率
probs = softmax([option_a_logit, option_b_logit])
# probs = [0.73, 0.27]  ← 这就是模型的选择概率
```

### 2.3 三种问题类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **Choice** | 从预定义选项中选一个 | `{"type": "choice", "options": ["紧急", "普通", "低优"]}` |
| **Score** | 1-10 分评分 | `{"type": "score", "min": 1, "max": 10}` |
| **Bool** | 是/否判断 | `{"type": "bool"}` |

### 2.4 LoRA 微调原理

```
原始模型权重 W (冻结)
     ↓
     +  ΔW = A × B  (可训练，低秩分解)
     ↓
输出

A: d × r 矩阵 (r << d, 如 r=8)
B: r × d 矩阵

可训练参数量: 2 × d × r ≈ 原模型的 0.1%~1%
```

---

## 3. 数据集设计

### 3.1 数据格式

采用 JSONL 格式，每行一个训练样本：

```json
{
  "id": "sample_001",
  "state": {
    "user_message": "这个bug修复后测试不通过，报错信息是NullPointerException",
    "context": {
      "issue_type": "bug",
      "priority": "high",
      "assignee": "developer_A"
    }
  },
  "questions": [
    {
      "id": "q1",
      "type": "choice",
      "text": "这个问题的根因最可能是什么？",
      "options": ["空指针未检查", "对象已释放", "并发竞争", "配置错误"]
    },
    {
      "id": "q2",
      "type": "bool",
      "text": "需要立即回滚吗？"
    },
    {
      "id": "q3",
      "type": "score",
      "text": "这个问题的严重程度评分(1-10)？"
    }
  ],
  "answers": [
    {"question_id": "q1", "answer": "空指针未检查", "confidence": 0.85},
    {"question_id": "q2", "answer": true, "confidence": 0.72},
    {"question_id": "q3", "answer": 7, "confidence": 0.68}
  ]
}
```

### 3.2 Prompt 模板

将结构化数据转换为模型可读的 prompt：

```
[System]
你是一个决策助手。根据给定的状态信息，回答决策问题。
对于每个问题，只输出你的选择，不要解释。

[State]
用户消息: 这个bug修复后测试不通过，报错信息是NullPointerException
上下文: {"issue_type": "bug", "priority": "high", "assignee": "developer_A"}

[Questions]
Q1 (选择): 这个问题的根因最可能是什么？
  A. 空指针未检查
  B. 对象已释放
  C. 并发竞争
  D. 配置错误

Q2 (是否): 需要立即回滚吗？
  A. 是
  B. 否

Q3 (评分): 这个问题的严重程度评分(1-10)？
  请输入1-10的数字

[Expected Answers]
Q1: A
Q2: A
Q3: 7
```

### 3.3 数据来源策略

| 来源 | 方法 | 预估数据量 |
|------|------|------------|
| **规则生成** | 用大模型(GPT-4/Claude)批量生成 | 1000~5000 条 |
| **人工标注** | 真实业务场景标注 | 100~500 条(高质量) |
| **数据增强** | 对现有样本做同义替换、扰动 | 扩充 3~5 倍 |

---

## 4. 硬件需求评估

### 4.1 训练显存估算

```
0.6B 模型 LoRA 微调:
├── 模型参数(FP16):  0.6B × 2 bytes = 1.2 GB
├── LoRA 参数:       ~5M × 2 bytes = 10 MB
├── 梯度:            ~5M × 2 bytes = 10 MB
├── 优化器状态:      ~5M × 8 bytes = 40 MB
├── 激活值:          batch_size × seq_len × ... ≈ 1~3 GB
└── ─────────────────────────────────────────────
    总计:            ~3~5 GB (batch_size=4, seq_len=512)
```

### 4.2 硬件推荐

| 环境 | GPU | 显存 | 可行性 |
|------|-----|------|--------|
| **本机(Intel UHD 770)** | 无独显 | — | ❌ 不支持 CUDA，只能 CPU 推理 |
| **Google Colab 免费版** | T4 | 15 GB | ✅ 完全够用 |
| **RTX 3060/4060** | NVIDIA | 8~12 GB | ✅ 可以训练 |
| **RTX 3090/4090** | NVIDIA | 24 GB | ✅ 非常充裕 |
| **云GPU租赁** | A100/V100 | 40~80 GB | ✅ 专业级 |

### 4.3 你的环境方案

由于本机只有 Intel 核显，建议：

1. **开发阶段**: 本机写代码、调试逻辑
2. **训练阶段**: 上传到 Google Colab 或租用云 GPU
3. **推理阶段**: 导出模型后用 CPU 在本机运行

---

## 5. 训练流程

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         训练流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 数据准备                                                     │
│     ├── 定义数据 schema                                          │
│     ├── 生成/收集训练样本                                        │
│     └── 转换为 JSONL 格式                                       │
│                                                                 │
│  ② 模型加载                                                     │
│     ├── 加载 Qwen3-0.6B 基座                                    │
│     └── 应用 LoRA 配置                                          │
│                                                                 │
│  ③ 训练循环                                                     │
│     ├── 前向传播计算 loss                                        │
│     ├── 反向传播更新 LoRA 参数                                   │
│     └── 验证集评估                                               │
│                                                                 │
│  ④ 模型导出                                                     │
│     ├── 保存 LoRA adapter                                        │
│     └── 可选: 合并为完整模型                                     │
│                                                                 │
│  ⑤ 推理部署                                                     │
│     ├── 加载基座 + LoRA adapter                                  │
│     ├── 输入 state + questions                                  │
│     └── 提取 logits → 概率 → 结构化答案                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 关键超参数

```yaml
# 训练配置
model:
  base_model: "Qwen/Qwen3-0.6B"
  lora_r: 8              # LoRA 秩
  lora_alpha: 32         # LoRA 缩放因子
  lora_dropout: 0.1

training:
  batch_size: 4
  gradient_accumulation_steps: 4  # 等效 batch=16
  learning_rate: 2e-4
  num_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_length: 512

evaluation:
  eval_strategy: "steps"
  eval_steps: 100
  save_steps: 100
```

### 5.3 Loss 函数设计

```python
# 标准交叉熵损失
loss = CrossEntropyLoss(model_output, target_token_ids)

# 其中 target 是我们期望的答案 token
# 例如: Q1 答案是 "A" → target = tokenizer("A").input_ids
```

---

## 6. 项目结构

```
D:/jev/
├── DESIGN.md                 # 本设计文档
├── README.md                 # 使用说明
├── requirements.txt          # Python 依赖
│
├── configs/                  # 配置文件
│   ├── train_config.yaml     # 训练配置
│   └── lora_config.yaml      # LoRA 配置
│
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的训练数据
│   │   ├── train.jsonl       # 训练集
│   │   ├── val.jsonl         # 验证集
│   │   └── test.jsonl        # 测试集
│   └── generate_data.py      # 数据生成脚本
│
├── src/                      # 源代码
│   ├── __init__.py
│   ├── model.py              # 模型加载与 LoRA 配置
│   ├── dataset.py            # 数据集处理
│   ├── trainer.py            # 训练逻辑
│   ├── inference.py          # 推理逻辑
│   └── utils.py              # 工具函数
│
├── scripts/                  # 运行脚本
│   ├── train.py              # 训练入口
│   ├── evaluate.py           # 评估脚本
│   └── demo.py               # Demo 推理
│
├── notebooks/                # Jupyter 笔记本
│   └── tutorial.ipynb        # 教程笔记本
│
├── outputs/                  # 输出目录
│   ├── models/               # 保存的模型
│   └── logs/                 # 训练日志
│
└── tests/                    # 测试
    └── test_dataset.py
```

---

## 7. 快速开始

### 7.1 环境安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 7.2 准备数据

```bash
# 生成示例数据
python data/generate_data.py
```

### 7.3 训练模型

```bash
# 本地训练(需要 GPU)
python scripts/train.py --config configs/train_config.yaml

# 或使用 Colab 上传训练
```

### 7.4 推理测试

```bash
# 运行 demo
python scripts/demo.py
```

---

## 8. 预期效果

### 8.1 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Accuracy** | 答案正确率 | > 70% |
| **Calibration Error** | 概率校准误差 | < 0.1 |
| **Latency** | 单次推理延迟 | < 100ms (GPU) |

### 8.2 示例输出

```python
# 输入
state = {"user_message": "系统响应很慢，CPU占用100%", "priority": "high"}
questions = [
    {"type": "choice", "text": "最可能的原因？", "options": ["死循环", "内存泄漏", "DDoS攻击"]},
    {"type": "bool", "text": "需要紧急处理？"}
]

# 输出
result = model.predict(state, questions)
print(result)
# {
#   "answers": [
#     {"question": "最可能的原因？", "answer": "死循环", "probability": 0.68},
#     {"question": "需要紧急处理？", "answer": True, "probability": 0.92}
#   ],
#   "latency_ms": 45
# }
```

---

## 9. 常见问题

### Q1: 为什么选择 Qwen3-0.6B 作为基座？

- 参数规模小，训练成本低
- 中文能力强
- 社区支持好，工具链完善

### Q2: 没有 GPU 能训练吗？

- 纯 CPU 训练极慢，不建议
- 推荐 Google Colab (免费 T4) 或云 GPU 租赁

### Q3: 数据量需要多少？

- 最少 100~500 条高质量标注数据
- 理想情况 1000~5000 条
- 可用大模型辅助生成

### Q4: 训练需要多久？

- Colab T4: 1000 条数据约 30 分钟~2 小时
- RTX 3090: 约 10~30 分钟

---

## 10. 后续扩展

1. **量化部署**: 使用 GGUF/AWQ 量化，进一步降低推理成本
2. **增量学习**: 支持在线学习新场景
3. **多任务联合训练**: 同时训练多个决策任务
4. **蒸馏**: 用大模型(JEV API/GPT-4)蒸馏知识到小模型

---

## 参考资料

1. [TypeSafe AI - Jev 官方介绍](https://www.langchain.com/blog/building-a-harness-with-jev)
2. [AgentJev 开源实现](https://www.cnblogs.com/itech/p/23126319)
3. [HuggingFace PEFT 文档](https://github.com/huggingface/peft)
4. [Qwen3 模型文档](https://huggingface.co/Qwen/Qwen3-0.6B)
5. [LoRA 论文](https://arxiv.org/abs/2106.09685)
