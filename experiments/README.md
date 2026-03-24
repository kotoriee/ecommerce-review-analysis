# 实验总览

本目录包含基于最新学术成果的启发性探索实验，**独立于主项目流程**，不影响原有训练和评估代码。

---

## 实验目录

| 目录 | 内容 | 优先级 | 预计工作量 |
|------|------|--------|-----------|
| [adaptive_temperature](adaptive_temperature/) | 动态温度缩放 | 高 | 1-2天 |
| [calibration](calibration/) | 置信度校准 | 高 | 1-2天 |
| [curriculum_learning](curriculum_learning/) | 课程学习优化 | 高 | 3-5天 |
| [contrastive_learning](contrastive_learning/) | 对比学习增强 | 中 | 1周 |
| [attention_residual](attention_residual/) | Attention Residual | 低 | 2-4周 |

---

## 背景

### 当前基线（2026-03-23）

| 指标 | 数值 |
|------|------|
| 模型 | Qwen3-4B + QLoRA |
| 准确率 | **86.50%** |
| 训练数据 | 8,500条（课程学习S1-S5）|
| 软标签来源 | DeepSeek-R1 蒸馏（T=2.0）|

### 探索目标

将最新学术成果（EMNLP 2024, arXiv 2025）应用到情感分析训练，验证是否能进一步提升：
- 准确率（目标：> 88%）
- 中性类识别（目标：> 85%）
- 置信度校准（ECE < 0.05）

---

## 快速开始

```bash
# 1. 动态温度实验
cd adaptive_temperature
python adaptive_temperature.py

# 2. 温度校准
cd calibration
python temperature_calibration.py

# 3. 课程学习优化
cd curriculum_learning
python curriculum_enhanced.py
```

---

## 实验规范

1. **独立性**：所有实验代码独立运行，不修改 `../code/` 下原有文件
2. **数据复用**：软标签数据统一从 `../data/curriculum/` 读取
3. **结果记录**：每个实验输出到各自目录的 `results/` 文件夹
4. **优先级**：按文档标注优先级执行，高优先级先尝试

---

## 参考论文

1. **EMNLP 2024** - "Calibrating Long-form Generations From Large Language Models"
2. **arXiv 2024** - "A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation"
3. **EMNLP 2024** - "Enhancing Language Model Factuality via Activation-Based Confidence Calibration"
4. **arXiv 2025** - "Direct Confidence Alignment"
5. **Frontiers AI 2025** - "Uncertainty and Variability in LLM-based Sentiment Analysis"

---

## 联系

实验问题记录：`experiments/issues.md`
