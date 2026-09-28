"""
训练入口脚本

用法:
    python scripts/train.py --config configs/train_config.yaml
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import train_from_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MiniJEV Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MiniJEV Training Script")
    print("=" * 60)

    # 检查数据文件是否存在
    if not os.path.exists("data/processed/train.jsonl"):
        print("\n[Warning] Training data not found!")
        print("Please run 'python data/generate_data.py' first to generate data.")
        return

    # 开始训练
    trainer = train_from_config(args.config)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
