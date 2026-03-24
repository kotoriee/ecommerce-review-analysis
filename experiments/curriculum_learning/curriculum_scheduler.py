"""
课程学习优化 - 动态调度器实现

基于置信度和多因素难度的动态课程学习训练
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Iterator
import random


class CurriculumStageDataset(Dataset):
    """
    课程学习阶段数据集

    支持按难度分层的样本加载
    """

    def __init__(self, samples: List[Dict], tokenizer, max_seq_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 编码文本
        text = sample['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label']),
            'soft_label': torch.tensor(sample.get('soft_label', [0.33, 0.33, 0.34])),
            'difficulty': sample.get('difficulty', 0.5)
        }


class DynamicCurriculumScheduler:
    """
    动态课程调度器

    根据训练进度动态决定哪些阶段的样本参与训练
    """

    def __init__(self,
                 stages: Dict[str, List[Dict]],
                 total_epochs: float = 3.0,
                 warmup_ratio: float = 0.3):
        """
        Args:
            stages: 各阶段样本字典 {'S1': [...], 'S2': [...], ...}
            total_epochs: 总训练轮数
            warmup_ratio: 快速升温比例（前x%epoch快速解锁所有阶段）
        """
        self.stages = stages
        self.total_epochs = total_epochs
        self.warmup_ratio = warmup_ratio
        self.stage_names = sorted(stages.keys())  # S1, S2, ...

        print(f"课程阶段: {self.stage_names}")
        for name in self.stage_names:
            print(f"  {name}: {len(stages[name])} 样本")

    def get_active_stages(self, epoch: float) -> List[str]:
        """
        获取当前epoch激活的阶段

        策略：渐进式解锁，warmup阶段快速解锁，之后全部可用
        """
        progress = epoch / self.total_epochs

        if progress < self.warmup_ratio:
            # warmup阶段：按进度逐步解锁
            num_active = max(1, int(progress / self.warmup_ratio * len(self.stages)))
            return self.stage_names[:num_active]
        else:
            # 之后全部可用
            return self.stage_names

    def get_sampling_weights(self, epoch: float) -> Dict[str, float]:
        """
        获取各阶段采样权重

        早期：简单样本权重高
        后期：困难样本权重高
        """
        progress = epoch / self.total_epochs

        weights = {}
        for i, name in enumerate(self.stage_names):
            # 越简单的阶段（索引越小），早期权重越高
            base_weight = 1.0 - progress * 0.3  # 基础权重递减
            stage_factor = 1.0 + i * progress * 0.2  # 困难阶段权重递增
            weights[name] = base_weight * stage_factor

        return weights

    def get_stage_progressive_weights(self, epoch: float) -> Dict[str, float]:
        """
        更激进的渐进策略：当前最高阶段权重最高
        """
        active_stages = self.get_active_stages(epoch)

        if len(active_stages) == 1:
            return {active_stages[0]: 1.0}

        # 最近解锁的阶段权重最高
        weights = {}
        for i, stage in enumerate(active_stages):
            # 指数增长：越靠后的阶段权重越高
            weights[stage] = 2.0 ** i

        # 归一化
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class CurriculumSampler(Sampler):
    """
    课程学习采样器

    根据调度器的权重从各阶段采样
    """

    def __init__(self,
                 stages: Dict[str, List[int]],  # 阶段名 -> 样本索引列表
                 scheduler: DynamicCurriculumScheduler,
                 epoch: float,
                 num_samples: int,
                 strategy: str = 'progressive'):
        """
        Args:
            stages: 各阶段包含的样本索引
            scheduler: 调度器
            epoch: 当前epoch（可以是小数，如1.5）
            num_samples: 每轮采样的样本数
            strategy: 采样策略
        """
        self.stages = stages
        self.scheduler = scheduler
        self.epoch = epoch
        self.num_samples = num_samples
        self.strategy = strategy

    def __iter__(self) -> Iterator[int]:
        """生成样本索引序列"""
        # 获取当前激活的阶段
        active_stages = self.scheduler.get_active_stages(self.epoch)

        # 获取采样权重
        if self.strategy == 'progressive':
            stage_weights = self.scheduler.get_stage_progressive_weights(self.epoch)
        else:
            stage_weights = self.scheduler.get_sampling_weights(self.epoch)

        # 按权重决定从各阶段采多少样本
        samples_per_stage = {}
        remaining = self.num_samples

        for stage in active_stages:
            weight = stage_weights.get(stage, 1.0)
            # 按比例分配
            n_samples = int(self.num_samples * weight)
            n_samples = min(n_samples, len(self.stages[stage]))
            samples_per_stage[stage] = n_samples
            remaining -= n_samples

        # 剩余样本分配给最后一个阶段
        if remaining > 0 and active_stages:
            last_stage = active_stages[-1]
            samples_per_stage[last_stage] = samples_per_stage.get(last_stage, 0) + remaining

        # 从各阶段采样
        all_indices = []
        for stage, n in samples_per_stage.items():
            indices = self.stages[stage]
            # 有放回或无放回采样
            if n <= len(indices):
                sampled = random.sample(indices, n)
            else:
                sampled = random.choices(indices, k=n)
            all_indices.extend(sampled)

        # 打乱顺序
        random.shuffle(all_indices)

        return iter(all_indices[:self.num_samples])

    def __len__(self):
        return self.num_samples


class MultiFactorDifficultyScorer:
    """
    多因素难度评分器

    综合考虑置信度、文本长度、情感模糊度等因素
    """

    def __init__(self,
                 confidence_weight: float = 0.4,
                 length_weight: float = 0.2,
                 entropy_weight: float = 0.2,
                 polarity_weight: float = 0.2):
        self.weights = {
            'confidence': confidence_weight,
            'length': length_weight,
            'entropy': entropy_weight,
            'polarity': polarity_weight
        }

    def compute_difficulty(self, sample: Dict) -> float:
        """
        计算样本难度分数（越高越难）

        Returns:
            difficulty: 0-1之间的难度分数
        """
        scores = {}

        # 1. 置信度分数（低置信度 = 高难度）
        soft_label = sample.get('soft_label', sample.get('probabilities', [0.33, 0.33, 0.34]))
        confidence = max(soft_label)
        scores['confidence'] = 1.0 - confidence

        # 2. 文本长度分数（过长或过短）
        text = sample.get('text', sample.get('input', ''))
        word_count = len(text.split())
        # 理想长度30-100词，偏离则增加难度
        if word_count < 30:
            scores['length'] = (30 - word_count) / 30 * 0.5
        elif word_count > 100:
            scores['length'] = min((word_count - 100) / 100, 1.0)
        else:
            scores['length'] = 0.0

        # 3. 标签熵分数（高熵 = 模糊 = 高难度）
        probs = np.array(soft_label)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(3)  # 三分类最大熵
        scores['entropy'] = entropy / max_entropy

        # 4. 情感极性分数（中性情感 = 高难度）
        neg, neu, pos = soft_label
        polarity = abs(pos - neg)  # 正负情感差异
        scores['polarity'] = 1.0 - polarity  # 低极性 = 高难度

        # 加权求和
        total_difficulty = sum(
            self.weights[name] * score
            for name, score in scores.items()
        )

        # 裁剪到[0,1]
        return min(max(total_difficulty, 0.0), 1.0)

    def create_curriculum_stages(self,
                                  samples: List[Dict],
                                  n_stages: int = 5) -> Dict[str, List[Dict]]:
        """
        根据难度分数创建课程阶段
        """
        # 计算难度
        for sample in samples:
            sample['difficulty'] = self.compute_difficulty(sample)

        # 按难度排序（简单 -> 困难）
        samples_sorted = sorted(samples, key=lambda x: x['difficulty'])

        # 分箱
        stage_size = len(samples_sorted) // n_stages
        stages = {}

        for i in range(n_stages):
            start = i * stage_size
            end = start + stage_size if i < n_stages - 1 else len(samples_sorted)
            stage_name = f'S{i+1}'
            stages[stage_name] = samples_sorted[start:end]

            # 打印统计
            difficulties = [s['difficulty'] for s in stages[stage_name]]
            print(f"{stage_name}: {len(stages[stage_name])} samples, "
                  f"difficulty [{min(difficulties):.3f}, {max(difficulties):.3f}]")

        return stages


def load_and_split_data(data_path: str,
                        scorer: MultiFactorDifficultyScorer,
                        n_stages: int = 5) -> Dict[str, List[Dict]]:
    """
    加载数据并按难度分阶段
    """
    # 加载数据
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"\n加载 {len(samples)} 条样本")

    # 分阶段
    stages = scorer.create_curriculum_stages(samples, n_stages)

    return stages


def main():
    """示例用法"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='数据路径')
    parser.add_argument('--n_stages', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    print("="*60)
    print("课程学习 - 数据准备")
    print("="*60)

    # 创建难度评分器
    scorer = MultiFactorDifficultyScorer(
        confidence_weight=0.4,
        length_weight=0.2,
        entropy_weight=0.2,
        polarity_weight=0.2
    )

    # 加载并分阶段
    stages = load_and_split_data(args.data, scorer, args.n_stages)

    # 创建调度器
    scheduler = DynamicCurriculumScheduler(
        stages=stages,
        total_epochs=args.epochs,
        warmup_ratio=0.3
    )

    # 模拟不同epoch的采样
    print("\n" + "="*60)
    print("采样策略演示")
    print("="*60)

    for epoch in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        active = scheduler.get_active_stages(epoch)
        weights = scheduler.get_stage_progressive_weights(epoch)

        print(f"\nEpoch {epoch:.1f}:")
        print(f"  激活阶段: {active}")
        print(f"  采样权重: {weights}")


if __name__ == '__main__':
    main()
