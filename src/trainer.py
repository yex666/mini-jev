"""
训练逻辑模块

本模块负责:
1. 配置 Trainer
2. 执行训练循环
3. 保存模型和 checkpoint
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from .model import MiniJEVModel, create_default_lora_config
from .dataset import MiniJEVDataset, collate_fn


class MiniJEVTrainer:
    """MiniJEV 训练器"""

    def __init__(self, config_path: str = "configs/train_config.yaml"):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        print(f"Loaded config from {config_path}")
        self._print_config()

        # 初始化模型
        lora_config = self._create_lora_config()
        self.model_wrapper = MiniJEVModel(
            base_model_name=self.config["model"]["base_model_name"],
            lora_config=lora_config,
            torch_dtype=getattr(torch, self.config["model"].get("torch_dtype", "float16")),
        )

        # 创建输出目录
        os.makedirs(self.config["training"]["output_dir"], exist_ok=True)
        os.makedirs(self.config["training"]["logging_dir"], exist_ok=True)

    def _print_config(self):
        """打印关键配置"""
        print("\n=== Training Configuration ===")
        print(f"Base model: {self.config['model']['base_model_name']}")
        print(f"LoRA r: {self.config['lora']['r']}")
        print(f"Epochs: {self.config['training']['num_train_epochs']}")
        print(f"Batch size: {self.config['training']['per_device_train_batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print("=" * 30 + "\n")

    def _create_lora_config(self) -> LoraConfig:
        """创建 LoRA 配置"""
        lora_cfg = self.config["lora"]
        return LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )

    def _create_training_args(self) -> TrainingArguments:
        """创建训练参数"""
        train_cfg = self.config["training"]
        return TrainingArguments(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            save_strategy=train_cfg.get("save_strategy", "steps"),
            save_steps=train_cfg.get("save_steps", 100),
            save_total_limit=train_cfg.get("save_total_limit", 3),
            eval_strategy=train_cfg.get("eval_strategy", "steps"),
            eval_steps=train_cfg.get("eval_steps", 100),
            logging_dir=train_cfg["logging_dir"],
            logging_steps=train_cfg.get("logging_steps", 10),
            report_to=train_cfg.get("report_to", "tensorboard"),
            fp16=train_cfg.get("fp16", True),
            optim=train_cfg.get("optim", "adamw_torch"),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            seed=train_cfg.get("seed", 42),
            remove_unused_columns=False,
        )

    def load_datasets(self):
        """加载训练和验证数据集"""
        data_cfg = self.config["data"]

        print("Loading training dataset...")
        self.train_dataset = MiniJEVDataset(
            data_file=data_cfg["train_file"],
            tokenizer=self.model_wrapper.tokenizer,
            max_seq_length=self.config["training"]["max_seq_length"],
        )

        if os.path.exists(data_cfg["eval_file"]):
            print("Loading evaluation dataset...")
            self.eval_dataset = MiniJEVDataset(
                data_file=data_cfg["eval_file"],
                tokenizer=self.model_wrapper.tokenizer,
                max_seq_length=self.config["training"]["max_seq_length"],
            )
        else:
            print("No evaluation dataset found, skipping eval")
            self.eval_dataset = None

    def train(self):
        """执行训练"""
        print("\n=== Starting Training ===\n")

        # 创建训练参数
        training_args = self._create_training_args()

        # 创建 Trainer
        trainer = Trainer(
            model=self.model_wrapper.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collate_fn,
            tokenizer=self.model_wrapper.tokenizer,
        )

        # 开始训练
        train_result = trainer.train()

        # 保存最终模型
        final_output_dir = os.path.join(
            self.config["training"]["output_dir"],
            "final_model"
        )
        print(f"\nSaving final model to {final_output_dir}...")
        trainer.save_model(final_output_dir)
        self.model_wrapper.tokenizer.save_pretrained(final_output_dir)

        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print("\n=== Training Completed ===")
        print(f"Final model saved to: {final_output_dir}")

        return trainer

    def evaluate(self, trainer: Optional[Trainer] = None) -> Dict[str, float]:
        """执行评估"""
        if self.eval_dataset is None:
            print("No evaluation dataset available")
            return {}

        if trainer is None:
            training_args = self._create_training_args()
            trainer = Trainer(
                model=self.model_wrapper.model,
                args=training_args,
                eval_dataset=self.eval_dataset,
                data_collator=collate_fn,
                tokenizer=self.model_wrapper.tokenizer,
            )

        print("\n=== Starting Evaluation ===\n")
        metrics = trainer.evaluate()

        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return metrics


def train_from_config(config_path: str = "configs/train_config.yaml"):
    """从配置文件启动训练的便捷函数"""
    trainer = MiniJEVTrainer(config_path)
    trainer.load_datasets()
    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniJEV Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    train_from_config(args.config)
