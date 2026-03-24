# 置信度校准实验

## 背景

模型输出的概率（softmax）往往不能真实反映预测置信度。通过校准技术，可以使模型"更诚实"地表达不确定性。

## 核心思想

**来源**: EMNLP 2024 "Calibrating the Confidence of Large Language Models"

校准目标：如果模型说"我有80%的信心"，那么应该有80%的预测是正确的。

## 实验方案

### 方案1：后处理温度缩放

在验证集上学习最优温度参数，推理时应用：

```python
class TemperatureScaler(nn.Module):
    """
    温度缩放校准器
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr=0.01, max_iter=50):
        """
        在验证集上优化温度参数

        Args:
            logits: 模型输出 (N, C)
            labels: 真实标签 (N,)
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """应用温度缩放"""
        return logits / self.temperature

    def calibrate_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """返回校准后的概率"""
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=-1)
```

### 方案2：直方图分箱校准

```python
class HistogramBinningCalibrator:
    """
    直方图分箱校准
    """
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_accuracies = np.zeros(n_bins)
        self.bin_counts = np.zeros(n_bins)

    def fit(self, confidences: np.ndarray, accuracies: np.ndarray):
        """
        学习每个bin的校准映射

        Args:
            confidences: 模型预测置信度 (N,)
            accuracies: 二元正确数组 (N,) - 1表示预测正确
        """
        for i in range(self.n_bins):
            mask = (confidences > self.bin_boundaries[i]) & \
                   (confidences <= self.bin_boundaries[i+1])
            if mask.sum() > 0:
                self.bin_accuracies[i] = accuracies[mask].mean()
                self.bin_counts[i] = mask.sum()

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """应用校准"""
        calibrated = np.copy(confidences)
        for i in range(self.n_bins):
            mask = (confidences > self.bin_boundaries[i]) & \
                   (confidences <= self.bin_boundaries[i+1])
            if mask.sum() > 0:
                calibrated[mask] = self.bin_accuracies[i]
        return calibrated
```

### 方案3：置信度对齐损失

**来源**: arXiv 2025 "Direct Confidence Alignment"

在训练时引入置信度正则化：

```python
class ConfidenceAlignmentLoss(nn.Module):
    """
    置信度对齐损失
    """
    def __init__(self, lambda_align=0.1):
        super().__init__()
        self.lambda_align = lambda_align
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # 标准交叉熵
        ce = self.ce_loss(logits, labels)

        # 置信度对齐项
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0]
        accuracy = (logits.argmax(dim=-1) == labels).float()

        # 鼓励置信度接近实际准确率
        alignment_loss = F.mse_loss(confidence, accuracy)

        return ce + self.lambda_align * alignment_loss
```

## 评估指标：ECE

```python
def compute_ece(confidences: np.ndarray,
                accuracies: np.ndarray,
                n_bins: int = 10) -> float:
    """
    计算期望校准误差 (Expected Calibration Error)

    Args:
        confidences: 模型预测置信度 (N,)
        accuracies: 二元正确数组 (N,)
        n_bins: 分箱数量

    Returns:
        ece: 期望校准误差
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & \
               (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * abs(bin_acc - bin_conf)

    return ece
```

## 文件结构

```
calibration/
├── README.md                  # 本文件
├── temperature_scaling.py    # 温度缩放校准
├── histogram_binning.py      # 直方图分箱
├── confidence_alignment.py   # 置信度对齐损失
├── calibration_metrics.py    # ECE等评估指标
├── run_calibration.py        # 主运行脚本
└── results/
    ├── reliability_diagram.png  # 可靠性图
    ├── before_calibration.json
    └── after_calibration.json
```

## 快速开始

```bash
# 1. 评估当前模型校准度
python calibration_metrics.py \
    --model ../../models/qwen3-4b-sentiment-soft-full \
    --data ../../data/curriculum/val_fixed.json \
    --output ./results/before_calibration.json

# 2. 应用温度缩放校准
python temperature_scaling.py \
    --model ../../models/qwen3-4b-sentiment-soft-full \
    --val_data ../../data/curriculum/val_fixed.json \
    --test_data ../../data/curriculum/test.json \
    --output ./results/temperature_scaling.json

# 3. 对比校准效果
python compare_calibration.py \
    --before ./results/before_calibration.json \
    --after ./results/temperature_scaling.json
```

## 预期效果

| 指标 | 校准前 | 校准后 (目标) |
|------|--------|---------------|
| ECE | ~0.08-0.12 | < 0.05 |
| 准确率 | 86.50% | 不变（或±0.5%）|
| 置信度-准确率相关性 | 低 | 高 |

## 可视化

生成可靠性图（Reliability Diagram）展示校准效果：
- X轴：置信度区间
- Y轴：实际准确率
- 对角线：完美校准
