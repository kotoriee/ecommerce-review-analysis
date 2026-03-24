# 课程学习优化

## 背景

当前已实现基础课程学习（S1-S5 分阶段训练）。本实验探索更精细的课程策略，进一步提升模型性能。

## 核心思想

**来源**: Confidence-based Curriculum Learning

学习过程应该像人类学习一样：
1. 先从"简单"的样本开始
2. 逐步增加难度
3. 最终学习全部数据

## 当前实现回顾

```python
# 当前S1-S5分箱策略
S1: confidence > 0.95  (高置信度)
S2: confidence > 0.90
S3: confidence > 0.85
S4: confidence > 0.70
S5: 全部数据
```

## 实验方案

### 方案1：动态课程调度

不仅按置信度分箱，还考虑训练进度动态调整：

```python
class DynamicCurriculumScheduler:
    """
    动态课程调度器
    """
    def __init__(self,
                 total_epochs: int = 3,
                 warmup_epochs: float = 0.5):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_active_stages(self, epoch: float) -> List[str]:
        """
        根据当前epoch决定激活哪些阶段

        策略:
        - Epoch 0-0.5: 仅 S1 (最高置信度)
        - Epoch 0.5-1.0: S1 + S2
        - Epoch 1.0-1.5: S1 + S2 + S3
        - Epoch 1.5-2.0: S1 + S2 + S3 + S4
        - Epoch 2.0+: 全部 S1-S5
        """
        if epoch < 0.5:
            return ['S1']
        elif epoch < 1.0:
            return ['S1', 'S2']
        elif epoch < 1.5:
            return ['S1', 'S2', 'S3']
        elif epoch < 2.0:
            return ['S1', 'S2', 'S3', 'S4']
        else:
            return ['S1', 'S2', 'S3', 'S4', 'S5']

    def get_stage_weight(self, stage: str, epoch: float) -> float:
        """
        为不同阶段分配采样权重

        早期阶段给予更高权重，确保模型充分学习简单模式
        """
        weights = {
            'S1': 1.0 - epoch * 0.1,  # 随训练递减
            'S2': 1.0,
            'S3': 1.0 + epoch * 0.05,
            'S4': 1.0 + epoch * 0.1,
            'S5': 1.0 + epoch * 0.15,  # 随训练递增
        }
        return weights.get(stage, 1.0)
```

### 方案2：难度度量融合

不仅用置信度，还结合其他难度指标：

```python
class MultiFactorDifficultyScorer:
    """
    多因素难度评分器
    """
    def __init__(self):
        self.weights = {
            'confidence': 0.4,
            'text_length': 0.2,
            'class_entropy': 0.2,
            'sentiment_polarity': 0.2
        }

    def compute_difficulty(self, sample: Dict) -> float:
        """
        综合多因素计算样本难度

        难度越高 = 越难学习
        """
        scores = []

        # 1. 置信度 (低置信度 = 高难度)
        conf_score = 1.0 - sample['confidence']
        scores.append(('confidence', conf_score))

        # 2. 文本长度 (过长或过短都可能增加难度)
        text_len = len(sample['text'].split())
        len_score = abs(text_len - 50) / 100  # 偏离50词的归一化距离
        scores.append(('text_length', len_score))

        # 3. 软标签熵 (高熵 = 模糊 = 高难度)
        probs = sample['soft_label']
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        entropy_score = entropy / np.log(3)  # 归一化到[0,1]
        scores.append(('class_entropy', entropy_score))

        # 4. 情感极性强度 (中性情感 = 高难度)
        neg, neu, pos = probs
        polarity = abs(pos - neg)
        polarity_score = 1.0 - polarity  # 低极性 = 高难度
        scores.append(('sentiment_polarity', polarity_score))

        # 加权求和
        total_difficulty = sum(
            self.weights[name] * score
            for name, score in scores
        )

        return total_difficulty

    def create_curriculum_stages(self,
                                  samples: List[Dict],
                                  n_stages: int = 5) -> Dict[str, List[Dict]]:
        """
        根据难度分数创建课程阶段
        """
        # 计算所有样本难度
        for sample in samples:
            sample['difficulty'] = self.compute_difficulty(sample)

        # 按难度排序
        samples_sorted = sorted(samples, key=lambda x: x['difficulty'])

        # 分箱
        stage_size = len(samples_sorted) // n_stages
        stages = {}
        for i in range(n_stages):
            start = i * stage_size
            end = start + stage_size if i < n_stages - 1 else len(samples_sorted)
            stages[f'S{i+1}'] = samples_sorted[start:end]

        return stages
```

### 方案3：遗忘感知重采样

监控模型在每个样本上的表现，对"容易遗忘"的样本增加采样概率：

```python
class ForgettingAwareSampler:
    """
    遗忘感知采样器

    跟踪每个样本的历史预测一致性，
    对频繁改变预测的样本给予更多关注
    """
    def __init__(self, n_samples: int):
        self.forget_counts = np.zeros(n_samples)
        self.last_predictions = {}
        self.sample_indices = np.arange(n_samples)

    def update(self, indices: np.ndarray, predictions: np.ndarray):
        """
        更新遗忘计数

        Args:
            indices: 样本索引
            predictions: 当前预测结果
        """
        for idx, pred in zip(indices, predictions):
            if idx in self.last_predictions:
                if self.last_predictions[idx] != pred:
                    self.forget_counts[idx] += 1
            self.last_predictions[idx] = pred

    def get_sampling_weights(self) -> np.ndarray:
        """
        获取采样权重

        遗忘次数越多的样本，权重越高
        """
        # 遗忘次数 + 1 的平方根作为权重
        weights = np.sqrt(self.forget_counts + 1)
        return weights / weights.sum()

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """
        根据遗忘权重采样一批样本
        """
        weights = self.get_sampling_weights()
        return np.random.choice(
            self.sample_indices,
            size=batch_size,
            p=weights
        )
```

## 实验对比

| 实验 | 课程策略 | 预期提升 |
|------|----------|----------|
| baseline | 固定S1-S5 | 86.50% |
| exp1 | 动态调度 | +0.5-1% |
| exp2 | 多因素难度 | +1-2% |
| exp3 | 遗忘感知 | +1-1.5% |

## 文件结构

```
curriculum_learning/
├── README.md                   # 本文件
├── dynamic_scheduler.py       # 动态课程调度
├── difficulty_scorer.py       # 多因素难度评分
├── forgetting_sampler.py      # 遗忘感知采样
├── curriculum_trainer.py      # 课程学习训练器
├── compare_strategies.py      # 策略对比
└── results/
    ├── baseline.json
    ├── dynamic_schedule.json
    ├── multi_factor.json
    └── comparison.png
```

## 快速开始

```bash
# 运行基线（复现当前最佳）
python curriculum_trainer.py \
    --strategy baseline \
    --data_dir ../../data/curriculum \
    --output ./results/baseline.json

# 动态调度实验
python curriculum_trainer.py \
    --strategy dynamic \
    --warmup_epochs 0.5 \
    --output ./results/dynamic.json

# 多因素难度实验
python curriculum_trainer.py \
    --strategy multi_factor \
    --difficulty_weights "confidence:0.4,text_length:0.2,entropy:0.2,polarity:0.2" \
    --output ./results/multi_factor.json

# 对比结果
python compare_strategies.py \
    --results_dir ./results \
    --output ./results/comparison.png
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| warmup_epochs | 0.5 | 快速进入下一阶段 |
| stage_weights | 动态 | 各阶段采样权重 |
| difficulty_factors | 4 | 难度评分维度 |

## 预期挑战

1. **动态调度**：需要修改数据加载器支持动态采样
2. **多因素难度**：需要预计算所有样本的特征
3. **遗忘感知**：需要维护额外的状态跟踪
