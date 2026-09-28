"""
数据集处理模块

本模块负责:
1. 加载 JSONL 格式的训练数据
2. 将结构化数据转换为 prompt 格式
3. 提供 PyTorch Dataset 接口
"""

import json
import torch
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MiniJEVDataset(Dataset):
    """MiniJEV 决策模型数据集"""

    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ):
        """
        初始化数据集

        Args:
            data_file: JSONL 数据文件路径
            tokenizer: 分词器
            max_seq_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # 加载数据
        print(f"Loading data from {data_file}...")
        self.samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]

        # 构建 prompt 和 target
        prompt = self._build_prompt(sample)
        target = self._build_target(sample)

        # Tokenize prompt (输入部分)
        prompt_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length - 10,  # 预留 target 空间
            return_tensors=None,
        )
        prompt_ids = prompt_inputs["input_ids"]

        # Tokenize target (答案部分)
        target_inputs = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=10,
            return_tensors=None,
        )
        target_ids = target_inputs["input_ids"]

        # 拼接 input_ids
        input_ids = prompt_ids + target_ids

        # 截断
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            target_ids = target_ids[:self.max_seq_length - len(prompt_ids)]

        # 构建 labels: prompt 部分设为 -100 (不计算 loss)，target 部分为真实 token
        labels = [-100] * len(prompt_ids) + target_ids

        # 截断 labels
        if len(labels) > self.max_seq_length:
            labels = labels[:self.max_seq_length]

        # 确保长度一致
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """构建 prompt"""
        state = sample.get("state", {})
        questions = sample.get("questions", [])

        # 格式化状态
        state_text = self._format_state(state)

        # 格式化问题
        questions_text = self._format_questions(questions)

        prompt = f"""[System]
你是一个决策助手。根据给定的状态信息，回答决策问题。
对于每个问题，只输出你的选择，不要解释。

[State]
{state_text}

[Questions]
{questions_text}

[Answers]
"""
        return prompt

    def _build_target(self, sample: Dict[str, Any]) -> str:
        """构建 target (期望答案)"""
        answers = sample.get("answers", [])
        questions = sample.get("questions", [])

        target_parts = []
        option_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

        for i, answer in enumerate(answers):
            q_type = questions[i]["type"] if i < len(questions) else "choice"

            if q_type == "choice":
                # 选择题: 输出选项字母
                options = questions[i].get("options", [])
                answer_text = answer.get("answer", "")
                if answer_text in options:
                    idx = options.index(answer_text)
                    target_parts.append(option_labels[idx] if idx < len(option_labels) else "A")
                else:
                    target_parts.append("A")  # 默认
            elif q_type == "bool":
                # 布尔题: 输出是/否
                target_parts.append("A" if answer.get("answer") else "B")
            elif q_type == "score":
                # 评分题: 输出数字
                target_parts.append(str(answer.get("answer", 5)))

        return "\n".join(target_parts)

    def _format_state(self, state: Dict[str, Any]) -> str:
        """格式化状态信息"""
        lines = []
        for key, value in state.items():
            if isinstance(value, dict):
                # 嵌套字典展开
                for k, v in value.items():
                    lines.append(f"{k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) if lines else "无额外上下文"

    def _format_questions(self, questions: List[Dict[str, Any]]) -> str:
        """格式化问题列表"""
        lines = []
        option_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

        for i, q in enumerate(questions):
            q_type = q.get("type", "choice")
            q_text = q.get("text", "")

            if q_type == "choice":
                options = q.get("options", [])
                options_text = "\n".join([f"  {label}. {opt}" for label, opt in zip(option_labels, options)])
                lines.append(f"Q{i+1} (选择): {q_text}\n{options_text}")
            elif q_type == "bool":
                lines.append(f"Q{i+1} (是否): {q_text}\n  A. 是\n  B. 否")
            elif q_type == "score":
                min_val = q.get("min", 1)
                max_val = q.get("max", 10)
                lines.append(f"Q{i+1} (评分): {q_text}\n  请输入{min_val}-{max_val}的数字")

        return "\n\n".join(lines)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    批处理函数，处理变长序列

    Args:
        batch: 样本列表

    Returns:
        批处理后的字典
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Padding
        padded_input = torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids.append(padded_input)
        attention_mask.append(padded_mask)
        labels.append(padded_labels)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer

    print("Testing MiniJEVDataset...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据集（需要先准备数据）
    try:
        dataset = MiniJEVDataset("./data/processed/train.jsonl", tokenizer)
        print(f"Dataset size: {len(dataset)}")

        # 获取单个样本
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")

        # 解码查看
        input_text = tokenizer.decode(sample["input_ids"])
        print(f"\nInput text preview:\n{input_text[:500]}...")
    except FileNotFoundError:
        print("Data file not found. Please generate data first.")
