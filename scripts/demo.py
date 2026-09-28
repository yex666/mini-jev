"""
推理 Demo 脚本

用法:
    python scripts/demo.py [--model_path outputs/models/final_model]
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import MiniJEVModel


def demo_choice():
    """演示选择题预测"""
    print("\n" + "=" * 60)
    print("Demo: Choice Question Prediction")
    print("=" * 60)

    # 加载模型（使用基座模型，不应用 LoRA）
    model = MiniJEVModel(base_model_name="Qwen/Qwen3-0.6B")

    # 测试用例
    test_cases = [
        {
            "state": {"user_message": "系统响应很慢，CPU占用100%", "priority": "high"},
            "question": "这个问题的根因最可能是什么？",
            "options": ["性能问题", "配置错误", "代码缺陷", "网络问题"]
        },
        {
            "state": {"file_type": "java", "change_lines": 150, "has_tests": True},
            "question": "这个PR的风险等级？",
            "options": ["低风险", "中风险", "高风险"]
        },
        {
            "state": {"issue_type": "bug", "severity": "critical", "affected_users": 1000},
            "question": "应该分配给哪个团队？",
            "options": ["开发团队", "运维团队", "安全团队", "产品团队"]
        }
    ]

    for i, tc in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"State: {tc['state']}")
        print(f"Question: {tc['question']}")
        print(f"Options: {tc['options']}")

        result = model.predict_choice(tc["state"], tc["question"], tc["options"])

        print(f"\nPrediction:")
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for opt, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {opt}: {prob:.2%} {bar}")


def demo_with_lora():
    """演示加载 LoRA adapter 进行预测"""
    print("\n" + "=" * 60)
    print("Demo: Prediction with LoRA Adapter")
    print("=" * 60)

    lora_path = "outputs/models/final_model"

    if not os.path.exists(lora_path):
        print(f"\n[Warning] LoRA adapter not found at {lora_path}")
        print("Please train the model first or specify a valid path.")
        return

    # 加载基座模型
    model = MiniJEVModel(base_model_name="Qwen/Qwen3-0.6B")

    # 加载 LoRA adapter
    try:
        model.load_lora_adapter(lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    except Exception as e:
        print(f"Failed to load LoRA adapter: {e}")
        return

    # 测试
    state = {"user_message": "登录页面报500错误", "priority": "critical"}
    question = "需要立即处理吗？"
    options = ["是", "否"]

    print(f"\nState: {state}")
    print(f"Question: {question}")
    print(f"Options: {options}")

    result = model.predict_choice(state, question, options)

    print(f"\nPrediction:")
    print(f"  Answer: {result['answer']}")
    print(f"  Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniJEV Demo Script")
    parser.add_argument(
        "--with_lora",
        action="store_true",
        help="Use LoRA adapter for prediction"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MiniJEV - System One Decision Model Demo")
    print("=" * 60)

    if args.with_lora:
        demo_with_lora()
    else:
        demo_choice()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
