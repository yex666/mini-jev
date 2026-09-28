"""
数据生成脚本

本脚本用于生成 MiniJEV 训练所需的示例数据。
实际使用时，可以用大模型（如 GPT-4、Claude）批量生成更多高质量数据。
"""

import json
import random
import os
from typing import Dict, Any, List


def generate_sample_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    生成示例训练数据

    Args:
        num_samples: 生成样本数量

    Returns:
        样本列表
    """
    samples = []

    # 定义一些模板场景
    scenarios = [
        {
            "scene": "bug_report",
            "states": [
                {"user_message": "系统响应很慢，CPU占用100%", "priority": "high", "module": "performance"},
                {"user_message": "登录页面报500错误", "priority": "critical", "module": "auth"},
                {"user_message": "按钮点击没反应", "priority": "medium", "module": "ui"},
                {"user_message": "数据导出失败，提示超时", "priority": "high", "module": "export"},
                {"user_message": "用户无法上传图片", "priority": "medium", "module": "upload"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "type": "choice",
                    "text": "这个问题的根因最可能是什么？",
                    "options": ["性能问题", "配置错误", "代码缺陷", "网络问题"]
                },
                {
                    "id": "q2",
                    "type": "bool",
                    "text": "需要立即处理吗？"
                },
                {
                    "id": "q3",
                    "type": "score",
                    "text": "这个问题的严重程度评分(1-10)？"
                }
            ]
        },
        {
            "scene": "code_review",
            "states": [
                {"file_type": "java", "change_lines": 150, "has_tests": True, "complexity": "medium"},
                {"file_type": "python", "change_lines": 50, "has_tests": False, "complexity": "low"},
                {"file_type": "javascript", "change_lines": 300, "has_tests": True, "complexity": "high"},
                {"file_type": "go", "change_lines": 80, "has_tests": True, "complexity": "medium"},
            ],
            "questions": [
                {
                    "id": "q1",
                    "type": "choice",
                    "text": "这个PR的风险等级？",
                    "options": ["低风险", "中风险", "高风险"]
                },
                {
                    "id": "q2",
                    "type": "bool",
                    "text": "建议合并吗？"
                },
                {
                    "id": "q3",
                    "type": "score",
                    "text": "代码质量评分(1-10)？"
                }
            ]
        },
        {
            "scene": "task_routing",
            "states": [
                {"issue_type": "bug", "severity": "critical", "affected_users": 1000},
                {"issue_type": "feature_request", "severity": "low", "affected_users": 10},
                {"issue_type": "question", "severity": "medium", "affected_users": 100},
                {"issue_type": "security", "severity": "critical", "affected_users": 10000},
            ],
            "questions": [
                {
                    "id": "q1",
                    "type": "choice",
                    "text": "应该分配给哪个团队？",
                    "options": ["开发团队", "运维团队", "安全团队", "产品团队"]
                },
                {
                    "id": "q2",
                    "type": "bool",
                    "text": "需要升级处理吗？"
                },
                {
                    "id": "q3",
                    "type": "score",
                    "text": "优先级评分(1-10)？"
                }
            ]
        }
    ]

    # 为每个场景生成答案
    answers_map = {
        "bug_report": {
            "high_performance": [
                {"question_id": "q1", "answer": "性能问题", "confidence": 0.85},
                {"question_id": "q2", "answer": True, "confidence": 0.9},
                {"question_id": "q3", "answer": 8, "confidence": 0.7}
            ],
            "critical_auth": [
                {"question_id": "q1", "answer": "代码缺陷", "confidence": 0.9},
                {"question_id": "q2", "answer": True, "confidence": 0.95},
                {"question_id": "q3", "answer": 9, "confidence": 0.8}
            ],
            "medium_ui": [
                {"question_id": "q1", "answer": "代码缺陷", "confidence": 0.7},
                {"question_id": "q2", "answer": False, "confidence": 0.6},
                {"question_id": "q3", "answer": 5, "confidence": 0.6}
            ],
            "high_export": [
                {"question_id": "q1", "answer": "性能问题", "confidence": 0.75},
                {"question_id": "q2", "answer": True, "confidence": 0.8},
                {"question_id": "q3", "answer": 7, "confidence": 0.7}
            ],
            "medium_upload": [
                {"question_id": "q1", "answer": "网络问题", "confidence": 0.6},
                {"question_id": "q2", "answer": False, "confidence": 0.5},
                {"question_id": "q3", "answer": 6, "confidence": 0.6}
            ]
        },
        "code_review": {
            "java_medium_tested": [
                {"question_id": "q1", "answer": "中风险", "confidence": 0.7},
                {"question_id": "q2", "answer": True, "confidence": 0.8},
                {"question_id": "q3", "answer": 7, "confidence": 0.7}
            ],
            "python_low_untested": [
                {"question_id": "q1", "answer": "低风险", "confidence": 0.8},
                {"question_id": "q2", "answer": True, "confidence": 0.9},
                {"question_id": "q3", "answer": 6, "confidence": 0.6}
            ],
            "js_high_tested": [
                {"question_id": "q1", "answer": "高风险", "confidence": 0.85},
                {"question_id": "q2", "answer": False, "confidence": 0.7},
                {"question_id": "q3", "answer": 5, "confidence": 0.7}
            ],
            "go_medium_tested": [
                {"question_id": "q1", "answer": "中风险", "confidence": 0.7},
                {"question_id": "q2", "answer": True, "confidence": 0.8},
                {"question_id": "q3", "answer": 7, "confidence": 0.7}
            ]
        },
        "task_routing": {
            "bug_critical_many_users": [
                {"question_id": "q1", "answer": "开发团队", "confidence": 0.9},
                {"question_id": "q2", "answer": True, "confidence": 0.95},
                {"question_id": "q3", "answer": 9, "confidence": 0.9}
            ],
            "feature_low_few_users": [
                {"question_id": "q1", "answer": "产品团队", "confidence": 0.85},
                {"question_id": "q2", "answer": False, "confidence": 0.9},
                {"question_id": "q3", "answer": 3, "confidence": 0.8}
            ],
            "question_medium": [
                {"question_id": "q1", "answer": "开发团队", "confidence": 0.7},
                {"question_id": "q2", "answer": False, "confidence": 0.7},
                {"question_id": "q3", "answer": 5, "confidence": 0.6}
            ],
            "security_critical": [
                {"question_id": "q1", "answer": "安全团队", "confidence": 0.95},
                {"question_id": "q2", "answer": True, "confidence": 0.98},
                {"question_id": "q3", "answer": 10, "confidence": 0.95}
            ]
        }
    }

    # 答案映射键
    answer_keys = {
        "bug_report": ["high_performance", "critical_auth", "medium_ui", "high_export", "medium_upload"],
        "code_review": ["java_medium_tested", "python_low_untested", "js_high_tested", "go_medium_tested"],
        "task_routing": ["bug_critical_many_users", "feature_low_few_users", "question_medium", "security_critical"]
    }

    sample_id = 0
    for scene_data in scenarios:
        scene = scene_data["scene"]
        states = scene_data["states"]
        questions = scene_data["questions"]
        keys = answer_keys[scene]

        for i, state in enumerate(states):
            if sample_id >= num_samples:
                break

            key = keys[i % len(keys)]
            answers = answers_map[scene][key]

            sample = {
                "id": f"sample_{sample_id:04d}",
                "state": state,
                "questions": questions,
                "answers": answers
            }
            samples.append(sample)
            sample_id += 1

    # 如果还需要更多样本，复制并添加噪声
    while len(samples) < num_samples:
        base_sample = random.choice(samples[:len(scenarios) * 4])  # 从基础样本中选择
        new_sample = augment_sample(base_sample, len(samples))
        samples.append(new_sample)

    return samples[:num_samples]


def augment_sample(sample: Dict[str, Any], new_id: int) -> Dict[str, Any]:
    """对样本进行简单增强（这里只是复制，实际可以加噪声）"""
    new_sample = {
        "id": f"sample_{new_id:04d}",
        "state": sample["state"].copy(),
        "questions": sample["questions"],
        "answers": sample["answers"]
    }
    return new_sample


def save_dataset(samples: List[Dict[str, Any]], output_dir: str):
    """保存数据集到 JSONL 文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 划分训练集、验证集、测试集 (8:1:1)
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }

    for split_name, split_samples in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_samples)} samples to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MiniJEV training data")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Output directory")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples...")
    samples = generate_sample_data(args.num_samples)
    save_dataset(samples, args.output_dir)
    print("Done!")
