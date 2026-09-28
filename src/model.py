"""
模型加载与 LoRA 配置模块

本模块负责:
1. 加载基座模型 (Qwen3-0.6B)
2. 应用 LoRA 配置
3. 提供模型推理接口
"""

import torch
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class MiniJEVModel:
    """MiniJEV 决策模型封装类"""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-0.6B",
        lora_config: Optional[LoraConfig] = None,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        初始化模型

        Args:
            base_model_name: 基座模型名称或路径
            lora_config: LoRA 配置，为 None 时不应用 LoRA
            device: 设备 ("cuda", "cpu", 或具体 GPU)
            torch_dtype: 数据类型
        """
        self.base_model_name = base_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        # 加载 tokenizer
        print(f"Loading tokenizer from {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )

        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        print(f"Loading base model from {base_model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device == "cuda" else None,
        )

        if self.device != "cuda":
            self.base_model = self.base_model.to(self.device)

        # 应用 LoRA
        self.lora_applied = False
        if lora_config is not None:
            self.apply_lora(lora_config)

        self.model = self.base_model
        print(f"Model loaded on {self.device}")

    def apply_lora(self, lora_config: LoraConfig):
        """应用 LoRA 配置"""
        print(f"Applying LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.lora_applied = True
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        if hasattr(self.base_model, "print_trainable_parameters"):
            self.base_model.print_trainable_parameters()
        else:
            total_params = sum(p.numel() for p in self.base_model.parameters())
            trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                  f"({100 * trainable_params / total_params:.2f}%)")

    def load_lora_adapter(self, adapter_path: str):
        """加载已训练的 LoRA adapter"""
        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.lora_applied = True
        print("LoRA adapter loaded successfully")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回 logits

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def get_last_token_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取最后一个 token 的 logits

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            last_logits: [batch_size, vocab_size]
        """
        logits = self.forward(input_ids, attention_mask)
        # 找到每个样本中最后一个非 padding token 的位置
        batch_size = input_ids.shape[0]
        last_positions = attention_mask.sum(dim=1) - 1  # [batch_size]
        last_logits = logits[torch.arange(batch_size), last_positions]
        return last_logits

    def get_option_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        options: List[str],
    ) -> Dict[str, float]:
        """
        计算各选项的概率分布

        Args:
            input_ids: 输入 token IDs [1, seq_len]
            attention_mask: 注意力掩码 [1, seq_len]
            options: 候选选项列表，如 ["A", "B", "C", "D"]

        Returns:
            各选项的概率字典，如 {"A": 0.45, "B": 0.30, "C": 0.15, "D": 0.10}
        """
        # 获取最后一个 token 的 logits
        last_logits = self.get_last_token_logits(input_ids, attention_mask)  # [1, vocab_size]

        # 提取各选项对应 token 的 logit
        option_logits = []
        option_tokens = []
        for option in options:
            # 获取选项第一个 token 的 ID
            token_id = self.tokenizer.encode(option, add_special_tokens=False)[0]
            option_tokens.append(token_id)
            option_logits.append(last_logits[0, token_id].item())

        # Softmax 转概率
        option_logits_tensor = torch.tensor(option_logits)
        probs = torch.softmax(option_logits_tensor, dim=0)

        return {option: prob.item() for option, prob in zip(options, probs)}

    def predict_choice(
        self,
        state: Dict[str, Any],
        question: str,
        options: List[str],
    ) -> Dict[str, Any]:
        """
        预测选择题答案

        Args:
            state: 状态信息字典
            question: 问题文本
            options: 候选选项列表

        Returns:
            预测结果字典，包含 answer 和 probabilities
        """
        # 构建 prompt
        prompt = self._build_choice_prompt(state, question, options)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # 计算概率
        with torch.no_grad():
            probs = self.get_option_probabilities(input_ids, attention_mask, options)

        # 选择概率最高的选项
        best_option = max(probs, key=probs.get)

        return {
            "answer": best_option,
            "probabilities": probs,
            "confidence": probs[best_option],
        }

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

    def _format_state(self, state: Dict[str, Any]) -> str:
        """格式化状态信息"""
        lines = []
        for key, value in state.items():
            if isinstance(value, dict):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


def create_default_lora_config() -> LoraConfig:
    """创建默认的 LoRA 配置"""
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )


if __name__ == "__main__":
    # 测试代码
    print("Testing MiniJEVModel...")

    # 加载模型（不应用 LoRA）
    model = MiniJEVModel(base_model_name="Qwen/Qwen3-0.6B")

    # 测试选择题预测
    state = {"user_message": "系统响应很慢", "priority": "high"}
    question = "最可能的原因是什么？"
    options = ["死循环", "内存泄漏", "DDoS攻击", "配置错误"]

    result = model.predict_choice(state, question, options)
    print(f"\nPrediction result: {result}")
