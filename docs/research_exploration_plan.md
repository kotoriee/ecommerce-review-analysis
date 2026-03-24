# 启发性探索：将最新学术成果应用到情感分析训练

## 当前项目状态

- **数据**：8616条软标注数据（NEG/NEU/POS三分类）
- **标注模型**：DeepSeek-V3.2（高质量软标签）
- **已有基础**：`train_sentiment_soft.py` 已实现基础软标签蒸馏（QLoRA + KL Divergence）
- **技术栈**：Unsloth、Qwen3-4B、Temperature=2.0、Alpha=0.5

## 最新学术成果探索方向

### 方向1：高级软标签蒸馏技术

#### 1.1 温度缩放优化（Temperature Scaling）
**来源**：EMNLP 2024 "Calibrating Long-form Generations From Large Language Models"

**当前问题**：固定温度 T=2.0 可能不是最优

**探索方案**：
- **动态温度**：根据样本置信度自适应调整温度
  - 高置信度样本（conf > 0.9）：T=1.5（保留锐利分布）
  - 中置信度样本（0.6 < conf < 0.9）：T=2.0（当前设置）
  - 低置信度样本（conf < 0.6）：T=2.5-3.0（更平滑分布）
- **可学习温度**：将温度作为可训练参数

**实验设计**：
\`\`\`python
# 动态温度示例
def adaptive_temperature(confidence):
    if confidence > 0.9:
        return 1.5
    elif confidence > 0.6:
        return 2.0
    else:
        return 2.5 + (0.6 - confidence) * 2  # 最高到3.0
\`\`\`

#### 1.2 混合损失函数（Hybrid Loss）
**来源**：arXiv 2024 "A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation"

**当前问题**：仅使用 KL + SFT 混合

**探索方案**：
- **三阶段损失**：
  1. 硬标签CE（保留基础能力）
  2. KL散度（学习软分布）
  3. **对比损失**（学习类间关系）

\`\`\`python
# 对比损失增强
contrastive_loss = InfoNCE(student_logits, teacher_logits)
total_loss = alpha * kl_loss + beta * ce_loss + gamma * contrastive_loss
\`\`\`

#### 1.3 标签平滑（Label Smoothing）
**来源**：2024 Calibration Surveys

**探索方案**：
- 对硬标签应用标签平滑（epsilon=0.1）
- 与软标签结合使用

---

### 方向2：置信度校准技术

#### 2.1 后处理校准（Post-hoc Calibration）
**来源**：EMNLP 2024 "Calibrating the Confidence of Large Language Models"

**问题**：模型输出概率可能与真实置信度不一致

**探索方案**：
1. **温度缩放（Temperature Scaling）**
   - 在验证集上优化T
   - 应用到推理阶段

2. **直方图校准（Histogram Binning）**
   - 将置信度分成多个bin
   - 每个bin学习校准映射

**实验代码**：
\`\`\`python
# 温度缩放校准
class TemperatureScaler:
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))

    def fit(self, logits, labels):
        # 在验证集上优化温度
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
        # ... 优化NLL

    def forward(self, logits):
        return logits / self.temperature
\`\`\`

#### 2.2 置信度与性能对齐
**来源**：arXiv 2025 "Direct Confidence Alignment"

**探索方案**：
- 训练时引入置信度正则化项
- 鼓励模型输出与准确率匹配的置信度

---

### 方向3：数据增强与课程学习

#### 3.1 基于置信度的课程学习
**来源**：Confidence-based Curriculum Learning

**探索方案**：
- 按置信度分层训练
  1. 第1阶段：仅高置信度样本（conf > 0.85）
  2. 第2阶段：加入中置信度样本（conf > 0.6）
  3. 第3阶段：全部数据

**优势**：先学习"容易"的样本，逐步增加难度

#### 3.2 软标签数据增强
**来源**：Label-Consistent Data Generation (2025)

**探索方案**：
- 对低置信度样本进行回译（back-translation）增强
- 保持软标签分布一致

---

### 方向4：模型架构优化

#### 4.1 Attention Residual机制
**来源**：Kimi-Attention-Residual (wyf3/llm_related)

**探索方案**：
- 修改残差连接为attention-weighted aggregation
- 可能需要自定义模型架构

**适用性**：⭐⭐⭐（高复杂度，需要大量修改）

#### 4.2 Multi-Task Learning
**来源**：LLM Distillation Survey 2025

**探索方案**：
- 同时预测：情感标签 + 置信度 + 推理过程
- 增强模型的解释能力

---

### 方向5：评估与验证创新

#### 5.1 ECE（Expected Calibration Error）评估
**来源**：EMNLP 2024 "Enhancing Language Model Factuality via Activation-Based Confidence Calibration"

**需要实现**：
\`\`\`python
def compute_ece(confidences, accuracies, n_bins=10):
    """计算期望校准误差"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return ece
\`\`\`

#### 5.2 不确定性量化
**来源**：Frontiers in AI 2025 "Uncertainty and Variability in LLM-based Sentiment Analysis"

**探索方案**：
- MC Dropout评估不确定性
- 集成方法（训练多个模型）

---

## 推荐实验优先级

### 高优先级（快速收益）
1. **动态温度缩放** - 实现简单，潜在收益大
2. **温度校准（Post-hoc）** - 无需重新训练
3. **基于置信度的课程学习** - 数据已准备好

### 中优先级（中等投入）
4. **混合损失函数**（加入对比学习）
5. **ECE评估指标** - 更好地理解模型校准

### 低优先级（长期研究）
6. **Attention Residual** - 高复杂度
7. **多任务学习** - 需要额外标注

---

## 下一步行动建议

### 立即尝试（1-2天）
\`\`\`bash
# 实验1：动态温度 vs 固定温度
python train_sentiment_soft.py --data train_3cls.json --adaptive-temp

# 实验2：温度校准
python calibrate_temperature.py --model checkpoint --val-data val_3cls.json
\`\`\`

### 短期探索（1周）
1. 实现ECE评估脚本
2. 对比不同温度策略
3. 尝试课程学习训练

### 中期研究（2-4周）
1. 对比学习损失集成
2. 数据增强实验
3. 多模型集成

---

## 需要创建的新文件

| 文件 | 用途 |
|------|------|
| `code/experiments/adaptive_temperature.py` | 动态温度训练 |
| `code/experiments/temperature_calibration.py` | 后处理校准 |
| `code/experiments/curriculum_learning.py` | 课程学习训练 |
| `code/evaluation/calibration_metrics.py` | ECE等指标 |
| `code/experiments/contrastive_loss.py` | 对比学习增强 |

---

## 参考论文清单

1. **Resource-Limited Multimodal Sentiment** (arXiv 2508.05234) - 软标签蒸馏
2. **LoRA+KD Hybrid** (arXiv 2410.20777) - 混合微调
3. **Calibrating LLM Confidence** (EMNLP 2024) - 置信度校准
4. **Direct Confidence Alignment** (arXiv 2512.11998) - 置信度对齐
5. **Uncertainty in LLM Sentiment** (Frontiers 2025) - 不确定性量化
