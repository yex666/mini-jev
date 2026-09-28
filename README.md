# MiniJEV: 0.6B System One 决策模型微调实战

一个轻量级的 JEV 风格决策模型训练框架，基于 Qwen3-0.6B + LoRA 微调。

## 项目简介

MiniJEV 实现了类似 [Jev](https://www.langchain.com/blog/building-a-harness-with-jev) 的 "System One" 决策模型：

- **不生成文本**，只做判断（选择题/评分/布尔值）
- **单次前向传播**，直接读取 logits 计算概率
- **毫秒级响应**，适合 Agent 决策场景

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
python data/generate_data.py --num_samples 100
```

### 3. 训练模型

```bash
python scripts/train.py --config configs/train_config.yaml
```

### 4. 运行 Demo

```bash
# 使用基座模型推理
python scripts/demo.py

# 使用训练后的 LoRA adapter
python scripts/demo.py --with_lora
```

## 项目结构

```
D:/jev/
├── DESIGN.md              # 详细设计文档
├── README.md              # 本文件
├── requirements.txt       # Python 依赖
├── configs/               # 配置文件
│   └── train_config.yaml  # 训练配置
├── data/                  # 数据目录
│   ├── generate_data.py   # 数据生成脚本
│   └── processed/         # 处理后的数据
├── src/                   # 源代码
│   ├── model.py           # 模型定义
│   ├── dataset.py         # 数据集处理
│   └── trainer.py         # 训练逻辑
├── scripts/               # 运行脚本
│   ├── train.py           # 训练入口
│   └── demo.py            # 推理 Demo
└── outputs/               # 输出目录
    ├── models/            # 保存的模型
    └── logs/              # 训练日志
```

## 硬件要求

| 环境 | GPU | 显存 | 可行性 |
|------|-----|------|--------|
| Google Colab | T4 | 15 GB | ✅ 推荐 |
| RTX 3060+ | NVIDIA | 8+ GB | ✅ 可以 |
| RTX 3090/4090 | NVIDIA | 24 GB | ✅ 充裕 |
| Intel UHD 770 | 无独显 | — | ❌ 不支持 |

## 参考资源

- [AgentJev](https://www.cnblogs.com/itech/p/23126319) - 0.6B 开源决策模型
- [Open-Jev/SemIf](https://baijiahao.baidu.com/s?id=1876986889617071976) - 家用 3090 可跑版本
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) - 基座模型
- [PEFT/LoRA](https://github.com/huggingface/peft) - 参数高效微调框架

## License

MIT
