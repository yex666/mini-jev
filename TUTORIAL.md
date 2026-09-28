# MiniJEV 完整教程：从零理解小模型微调全流程

## 目录
1. [RTX 4060 可行性分析](#1-rtx-4060-可行性分析)
2. [数据集从哪来？](#2-数据集从哪来)
3. [代码逐行详解](#3-代码逐行详解)
4. [Claude Opus 国内订阅指南](#4-claude-opus-国内订阅指南)

---

## 1. RTX 4060 可行性分析

### 结论：完全没问题！

| 项目 | 数值 |
|------|------|
| RTX 4060 显存 | **8 GB** |
| 0.6B 模型 LoRA 训练显存需求 | **~3-5 GB** |
| 剩余空间 | **3-5 GB（可加大 batch_size）** |

### 详细计算

```
0.6B 模型 LoRA 微调显存占用:

┌─────────────────────────────────────────────────┐
│ 组件              │ 计算方式            │ 占用    │
├─────────────────────────────────────────────────┤
│ 模型权重(FP16)    │ 0.6B × 2 bytes      │ 1.2 GB  │
│ LoRA参数(r=8)     │ ~5M × 2 bytes       │ 10 MB   │
│ 梯度              │ ~5M × 2 bytes       │ 10 MB   │
│ AdamW优化器状态   │ ~5M × 8 bytes       │ 40 MB   │
│ 激活值(batch=4)   │ 动态分配            │ 1-2 GB  │
│ CUDA上下文        │ 固定开销            │ 0.5 GB  │
├─────────────────────────────────────────────────┤
│ 总计              │                     │ ~3 GB   │
└─────────────────────────────────────────────────┘

RTX 4060 有 8GB，还剩 5GB 余量 → 非常轻松！
```

### 对比参考

根据实测数据：
- [RTX 4060 8GB 微调 Qwen3.5-9B](https://blog.csdn.net/excellent_fish_c/article/details/161584856) — 9B 都能跑，0.6B 更轻松
- [消费级显卡 LoRA 微调极限测试](https://blog.csdn.net/gedonshen/article/details/162519630) — RTX 4060 可用 QLoRA 跑 8B

---

## 2. 数据集从哪来？

### 2.1 JEV 生态的数据集方案

我调研了 GitHub 上的开源项目，主要有以下几种方式：

#### 方法 A：蒸馏法（最常用）

用大模型（GPT-4/Claude/JEV API）生成高质量标注数据，小模型模仿学习。

```python
# 步骤：
# 1. 准备原始场景数据（如 bug 报告、代码审查记录）
# 2. 调用大模型生成 "场景 + 问题 + 答案" 三元组
# 3. 人工抽检或自动过滤低质量样本
# 4. 导出为 JSONL 格式训练
```

**参考项目：**
- [typed-decisions 数据集](https://huggingface.co/datasets/com-kotobalabs/typed-decisions-repo-governance) — Typed Decisions 官方测试集
- [agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection) — Agent 交互轨迹数据

#### 方法 B：Logit 直接读取法（SemIf/Open-Jev）

**不需要训练！** 直接用冻结的基座模型，读取候选 token 的 logit 值做 softmax。

```python
# SemIf 的做法：不训练，直接推理
logits = frozen_model(input).logits[:, -1, :]
prob_A = softmax(logits[tokenizer("A").input_ids[0]])
```

**参考项目：**
- [Open-Jev/SemIf](https://baijiahao.baidu.com/s?id=1876986889617071976) — 家用 3090 可跑
- [awesome-jev](https://github.com/yibie/awesome-jev) — JEV 生态聚合列表

#### 方法 C：规则+人工标注

我们 `generate_data.py` 目前用的就是这种，适合入门学习。

### 2.2 推荐的数据获取流程

```
Step 1: 收集原始业务数据
        ├── Bug 工单系统导出
        ├── 代码审查记录
        └── 任务分配历史

Step 2: 用大模型批量生成标注
        prompt = """
        给定以下场景，生成决策问题和答案：
        {原始数据}
        
        输出格式：
        - questions: 决策问题列表
        - answers: 每个问题的正确答案和置信度
        """
        
        # 调用 Claude/GPT-4 批量生成
        for raw_data in raw_dataset:
            result = call_llm(prompt.format(raw_data))
            save_to_jsonl(result)

Step 3: 质量过滤
        ├── 去除明显错误的答案
        ├── 检查选项是否合理
        └── 人工抽检 10% 样本

Step 4: 数据增强（可选）
        ├── 同义词替换
        ├── 句式改写
        └── 添加噪声
```

### 2.3 实际数据量建议

| 阶段 | 数据量 | 用途 |
|------|--------|------|
| 原型验证 | 100-500 条 | 跑通流程，验证可行性 |
| 正式训练 | 1000-5000 条 | 获得可用模型 |
| 生产部署 | 10000+ 条 | 高精度需求 |

---

## 3. 代码逐行详解

### 3.1 整体架构回顾

```
┌─────────────────────────────────────────────────────────────┐
│                    完整训练流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① 数据准备 (generate_data.py)                              │
│     └── 生成 {state, questions, answers} 格式的 JSONL       │
│                                                             │
│  ② 数据集加载 (dataset.py)                                  │
│     └── JSONL → Prompt 拼接 → Token IDs → PyTorch Dataset  │
│                                                             │
│  ③ 模型初始化 (model.py)                                    │
│     └── 加载 Qwen3-0.6B + 应用 LoRA                        │
│                                                             │
│  ④ 训练循环 (trainer.py)                                    │
│     └── 前向传播 → 计算 Loss → 反向传播 → 更新 LoRA 参数    │
│                                                             │
│  ⑤ 推理预测 (model.py / demo.py)                            │
│     └── 输入 state+question → 提取 logits → 概率 → 答案     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 `src/model.py` 逐行解析

这个文件是核心，负责：
1. 加载基座模型
2. 应用 LoRA
3. 实现 JEV 式 logit 提取

#### 第一部分：导入依赖

```python
import torch
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
```

**解释：**
- `torch`: PyTorch 深度学习框架
- `transformers`: HuggingFace 的模型库，提供预训练模型加载
- `peft`: Parameter-Efficient Fine-Tuning，LoRA 的实现库

---

#### 第二部分：类初始化

```python
class MiniJEVModel:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B",  # 基座模型名称
        lora_config: Optional[LoraConfig] = None,   # LoRA配置，None表示不用LoRA
        device: Optional[str] = None,               # 设备：cuda/cpu
        torch_dtype: torch.dtype = torch.float16,   # 数据类型：FP16省显存
    ):
```

**关键参数说明：**
- `base_model_name`: 从 HuggingFace 下载的模型名，会自动缓存到本地
- `lora_config`: 如果传入配置，则启用 LoRA 微调模式
- `device`: 自动检测 GPU，没有就用 CPU
- `torch_dtype`: FP16 比 FP32 省一半显存，精度损失很小

---

#### 第三部分：加载 Tokenizer

```python
# 加载 tokenizer
print(f"Loading tokenizer from {base_model_name}...")
self.tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,  # Qwen 需要此参数
)

# 设置 pad token
if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
```

**什么是 Tokenizer？**

Tokenizer 是把文本转换成数字的工具。模型只接受数字输入。

```
输入: "你好"
Tokenizer 处理后: [1234, 5678]  ← 这是 token IDs
```

**为什么要设置 pad_token？**

训练时需要把不同长度的序列对齐（padding），需要一个特殊符号表示"填充位"。

```
序列1: [你, 好, <pad>, <pad>]
序列2: [今, 天, 天, 气, 好]
```

---

#### 第四部分：加载基座模型

```python
# 加载基座模型
print(f"Loading base model from {base_model_name}...")
self.base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch_dtype,           # FP16 省显存
    device_map=self.device if self.device == "cuda" else None,
)

if self.device != "cuda":
    self.base_model = self.base_model.to(self.device)
```

**AutoModelForCausalLM 是什么？**

这是因果语言模型（Causal Language Model），也就是 GPT 类的自回归模型。它会预测下一个 token。

**device_map 参数：**
- `"cuda"`: 自动分配到 GPU
- `None`: 先加载到 CPU，再手动移动

---

#### 第五部分：应用 LoRA

```python
def apply_lora(self, lora_config: LoraConfig):
    """应用 LoRA 配置"""
    print(f"Applying LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    self.base_model = get_peft_model(self.base_model, lora_config)
    self.lora_applied = True
    self.print_trainable_parameters()
```

**LoRA 的核心原理：**

```
原始权重 W (d × d 矩阵，冻结不动)
         ↓
输出 = W × x + ΔW × x
              ↑
         ΔW = A × B  (低秩分解，只训练 A 和 B)

A: d × r 矩阵 (r 很小，比如 8)
B: r × d 矩阵

原来要训练: d × d = 4096 × 4096 = 1600万参数
现在只需训练: 2 × d × r = 2 × 4096 × 8 = 6.5万参数 ← 减少 99.6%!
```

---

#### 第六部分：前向传播（核心！）

```python
def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """前向传播，返回 logits"""
    outputs = self.model(
        input_ids=input_ids,          # 输入 token IDs
        attention_mask=attention_mask, # 哪些是真实token，哪些是padding
    )
    return outputs.logits  # [batch_size, seq_len, vocab_size]
```

**什么是 logits？**

Logits 是模型对词表中每个 token 的"未归一化分数"。

```
假设词表只有 5 个词: ["你", "好", "世", "界", "<unk>"]

模型最后一个位置的 logits:
[2.3, 0.5, -1.2, 0.8, -0.5]
  ↑    ↑     ↑     ↑     ↑
 "你" "好"  "世"  "界"  unk

经过 Softmax 变成概率:
[0.65, 0.11, 0.02, 0.15, 0.07]
  ↑
 最大 → 模型预测下一个词是 "你"
```

---

#### 第七部分：获取最后一个 token 的 logits（JEV 核心！）

```python
def get_last_token_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """获取最后一个 token 的 logits"""
    logits = self.forward(input_ids, attention_mask)
    
    # 找到每个样本中最后一个非 padding token 的位置
    batch_size = input_ids.shape[0]
    last_positions = attention_mask.sum(dim=1) - 1  # [batch_size]
    
    # 提取对应位置的 logits
    last_logits = logits[torch.arange(batch_size), last_positions]
    return last_logits  # [batch_size, vocab_size]
```

**为什么取最后一个 token？**

因为 JEV 模型不做自回归解码，而是看模型在"读完所有输入后"的状态。

```
输入: "[State] bug: CPU 100%  [Question] 根因? Answer:"
                                                       ↑
                                                   最后一个 token
                                                   
这个位置汇聚了前面所有信息 → 用它来做决策
```

---

#### 第八部分：计算选项概率（JEV 核心中的核心！）

```python
def get_option_probabilities(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    options: List[str],  # 例如 ["A", "B", "C", "D"]
) -> Dict[str, float]:
    """计算各选项的概率分布"""
    
    # Step 1: 获取最后一个 token 的 logits
    last_logits = self.get_last_token_logits(input_ids, attention_mask)
    # last_logits shape: [1, vocab_size]
    
    # Step 2: 提取各选项对应 token 的 logit
    option_logits = []
    for option in options:
        # 把选项文本转成 token ID
        token_id = self.tokenizer.encode(option, add_special_tokens=False)[0]
        # 例如: "A" → token_id = 10032
        
        # 从 logits 中取出该位置的分数
        logit_value = last_logits[0, token_id].item()
        option_logits.append(logit_value)
    
    # Step 3: Softmax 转概率
    option_logits_tensor = torch.tensor(option_logits)
    probs = torch.softmax(option_logits_tensor, dim=0)
    
    # 返回: {"A": 0.45, "B": 0.30, "C": 0.15, "D": 0.10}
    return {option: prob.item() for option, prob in zip(options, probs)}
```

**图解这个过程：**

```
输入 Prompt: "...Answer:"
              ↓
         Tokenizer
              ↓
     [1234, 5678, ..., 9999]  ← token IDs
              ↓
          模型前向传播
              ↓
     logits: [-2.1, 0.5, 3.2, ..., 1.8]  ← 词表大小的向量
              ↓
     提取选项对应位置的 logit:
     "A" 的 token_id = 10032 → logit_A = logits[10032] = 2.1
     "B" 的 token_id = 10033 → logit_B = logits[10033] = 0.8
     "C" 的 token_id = 10034 → logit_C = logits[10034] = -0.5
     "D" 的 token_id = 10035 → logit_D = logits[10035] = 0.3
              ↓
         Softmax 归一化
              ↓
     probs = [0.55, 0.18, 0.07, 0.20]
              ↓
     最终答案: "A" (概率 55%)
```

**这就是 JEV 的核心机制：不生成文本，直接读概率！**

---

#### 第九部分：构建 Prompt

```python
def _build_choice_prompt(self, state: Dict[str, Any], question: str, options: List[str]) -> str:
    """构建选择题 prompt"""
    option_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    options_text = "\n".join([f"  {label}. {opt}" for label, opt in zip(option_labels, options)])

    prompt = f"""[System]
你是一个决策助手。根据给定的状态信息，回答决策问题。
只输出你的选择字母（A/B/C/D等），不要解释。

[State]
{self._format_state(state)}

[Question]
{question}
{options_text}

Answer:"""
    return prompt
```

**为什么这样设计 Prompt？**

- `[System]`: 告诉模型角色和约束
- `[State]`: 提供上下文信息
- `[Question]`: 明确问题和选项
- `Answer:`: 引导模型输出答案位置

---

### 3.3 `src/dataset.py` 逐行解析

这个文件负责把 JSONL 数据转成模型能训练的格式。

#### 核心：__getitem__ 方法

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """获取单个样本"""
    sample = self.samples[idx]

    # Step 1: 构建 prompt 和 target
    prompt = self._build_prompt(sample)   # 输入部分
    target = self._build_target(sample)   # 期望答案

    # Step 2: Tokenize prompt
    prompt_inputs = self.tokenizer(
        prompt,
        truncation=True,
        max_length=self.max_seq_length - 10,
        return_tensors=None,
    )
    prompt_ids = prompt_inputs["input_ids"]  # 例如: [1234, 5678, ...]

    # Step 3: Tokenize target
    target_inputs = self.tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=10,
        return_tensors=None,
    )
    target_ids = target_inputs["input_ids"]  # 例如: [10032] (代表"A")

    # Step 4: 拼接 input_ids
    input_ids = prompt_ids + target_ids
    # 完整输入: [prompt tokens...] + [answer tokens...]

    # Step 5: 构建 labels
    # 关键! prompt 部分设为 -100 (不计算 loss)，只有 answer 部分计算 loss
    labels = [-100] * len(prompt_ids) + target_ids
    
    # 为什么要这样设计？
    # 因为我们只想让模型学习"回答问题"，不想让它学习"复述问题"
    # -100 是 PyTorch 的特殊标记，表示"忽略这个位置"

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
```

**图解数据转换过程：**

```
原始 JSONL:
{
  "state": {"bug": "CPU 100%"},
  "questions": [{"text": "根因?", "options": ["性能", "配置"]}],
  "answers": [{"answer": "性能"}]
}
          ↓
构建 Prompt:
"[System] 你是决策助手...\n[State] bug: CPU 100%\n[Question] 根因?\n  A. 性能\n  B. 配置\n\nAnswer:"
          ↓
构建 Target:
"A"
          ↓
Tokenize:
prompt_ids: [1234, 5678, ..., 9999]  (假设 50 个 token)
target_ids: [10032]                  ("A" 的 token ID)
          ↓
拼接:
input_ids: [1234, 5678, ..., 9999, 10032]  (51 个 token)
          ↓
构建 Labels:
labels: [-100, -100, ..., -100, 10032]
         ↑ prompt部分忽略     ↑ 只在答案位置计算loss
          ↓
送入模型训练!
```

---

### 3.4 `src/trainer.py` 逐行解析

这个文件封装了 HuggingFace Trainer，处理训练循环。

#### 核心：train 方法

```python
def train(self):
    """执行训练"""
    # Step 1: 创建训练参数
    training_args = self._create_training_args()
    
    # Step 2: 创建 Trainer
    trainer = Trainer(
        model=self.model_wrapper.model,      # 我们的模型
        args=training_args,                  # 训练超参数
        train_dataset=self.train_dataset,    # 训练数据
        eval_dataset=self.eval_dataset,      # 验证数据
        data_collator=collate_fn,            # 批处理函数
        tokenizer=self.model_wrapper.tokenizer,
    )

    # Step 3: 开始训练！
    train_result = trainer.train()
    
    # Step 4: 保存最终模型
    final_output_dir = os.path.join(self.config["training"]["output_dir"], "final_model")
    trainer.save_model(final_output_dir)
    self.model_wrapper.tokenizer.save_pretrained(final_output_dir)
    
    return trainer
```

**Trainer 内部做了什么？**

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播
        outputs = model(batch["input_ids"], batch["attention_mask"])
        logits = outputs.logits
        
        # 2. 计算 Loss (CrossEntropy)
        # 只计算 labels != -100 的位置
        loss = CrossEntropyLoss(logits, batch["labels"])
        
        # 3. 反向传播
        loss.backward()
        
        # 4. 更新参数（只更新 LoRA 的 A, B 矩阵）
        optimizer.step()
        optimizer.zero_grad()
        
        # 5. 记录日志
        if step % logging_steps == 0:
            log(loss.item())
```

---

### 3.5 `data/generate_data.py` 解析

这个文件生成示例训练数据。

#### 核心数据结构

```python
sample = {
    "id": "sample_001",
    "state": {                          # 上下文状态
        "user_message": "系统响应很慢",
        "priority": "high"
    },
    "questions": [                      # 决策问题列表
        {
            "type": "choice",           # 问题类型: choice/score/bool
            "text": "根因是什么？",
            "options": ["性能问题", "配置错误", "代码缺陷"]
        }
    ],
    "answers": [                        # 期望答案
        {
            "question_id": "q1",
            "answer": "性能问题",       # 正确选项
            "confidence": 0.85          # 置信度
        }
    ]
}
```

---

## 4. Claude Opus 国内订阅指南

关于你问的 Claude Opus 5.5（目前最新版本是 Opus 4.7/4.8），国内订阅有以下几种方式：

### 4.1 订阅档位一览

| 档位 | 价格 | 核心权益 |
|------|------|----------|
| Free | 免费 | 基础对话，Sonnet 限额使用 |
| Pro | $20/月 | 完整功能，Opus 可用 |
| Max 5x | $100/月 | Pro + 5倍用量 |
| Max 20x | $200/月 | Pro + 20倍用量 |

### 4.2 国内订阅方式对比

| 方式 | 难度 | 成本 | 风险 |
|------|------|------|------|
| **App Store 美区订阅** | ⭐⭐ | $20/月 | ✅ 最安全，正规渠道 |
| **代充平台** | ⭐ | $20/月 + 服务费 | ⚠️ 需选靠谱平台 |
| **虚拟信用卡 VCC** | ⭐⭐⭐ | $20/月 | ❌ 易被风控封号 |
| **API 中转服务** | ⭐⭐ | 按量付费 | ⚠️ 第三方稳定性不一 |

### 4.3 推荐方式：App Store 美区订阅

**步骤：**

```
Step 1: 注册美区 Apple ID
        ├── 访问 appleid.apple.com
        ├── 选择"其他国家/地区" → 美国
        └── 使用任意邮箱注册

Step 2: 购买美区礼品卡
        ├── 淘宝/拼多多搜索"美区苹果礼品卡"
        ├── 面额选择 $25 或 $50
        └── 购买后获得兑换码

Step 3: 兑换并订阅
        ├── App Store 登录美区账号
        ├── 兑换礼品卡余额
        ├── 搜索 Claude 下载
        └── 在 App 内订阅 Pro ($20/月)
```

**优点：**
- 完全正规，不会被封号
- 支持支付宝/微信购买礼品卡
- 一次设置，长期可用

**参考教程：**
- [Claude Pro 充值全攻略](https://www.programmercarl.com/qita/0021.claudeprodaichong.html)
- [Claude 国内订阅指南](https://github.com/Brooke587/claude-pro-china-zh)

### 4.4 替代方案：API 中转

如果只是想按量付费调用 Claude API：

```
国内 API 中转平台（仅供参考，自行评估风险）：
- 支持支付宝充值
- 按 token 计费
- 无需科学上网

注意：选择运营时间长、口碑好的平台
```

**参考：**
- [国内低成本调用 Claude API](https://devpress.csdn.net/v1/article/detail/159043686)

---

## 总结

### 你现在应该清楚了：

1. ✅ **RTX 4060 完全可以训练 0.6B** — 甚至绰绰有余
2. ✅ **数据集来源** — 蒸馏法（大模型生成）是最主流的方式
3. ✅ **代码每一步的原理** — Logit 提取是 JEV 的核心
4. ✅ **Claude 订阅方式** — App Store 美区是最稳妥的选择

### 下一步行动：

```bash
# 1. 安装依赖
cd D:/jev
pip install -r requirements.txt

# 2. 生成示例数据
python data/generate_data.py --num_samples 100

# 3. 把你的项目上传到 Colab 或在本机 RTX 4060 上训练
python scripts/train.py --config configs/train_config.yaml

# 4. 运行 Demo 验证效果
python scripts/demo.py
```

有任何问题随时问我！
