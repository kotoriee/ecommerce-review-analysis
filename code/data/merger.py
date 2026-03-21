"""
数据集合并与划分模块 (merger.py)

实现数据集的划分、多语言合并、JSONL 落盘功能。
遵循 CLAUDE.md 约束：原子化设计、TDD。
"""

import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import logging

from .schema import RawRecord, ProcessedRecord

logger = logging.getLogger(__name__)


# ==================== 配置定义 ====================

@dataclass
class SplitConfig:
    """数据集划分配置"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_by_label: bool = True
    shuffle: bool = True
    random_seed: int = 42

    def __post_init__(self):
        """验证比例有效性"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        if any(r < 0 for r in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("Split ratios must be non-negative")


# ==================== 核心划分函数 ====================

def _shuffle_records(records: List[RawRecord], seed: int) -> List[RawRecord]:
    """随机打乱记录顺序"""
    shuffled = records.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled


def _stratified_split(
    records: List[RawRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[RawRecord], List[RawRecord], List[RawRecord]]:
    """分层划分数据集，保持标签分布"""
    # 按标签分组
    by_label: Dict[int, List[RawRecord]] = defaultdict(list)
    for r in records:
        by_label[r.sentiment_label].append(r)

    train, val, test = [], [], []

    for label, group in by_label.items():
        n = len(group)
        # 计算各集大小，使用四舍五入减少累积误差
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # 测试集取剩余
        n_test = n - n_train - n_val

        # 确保非负
        if n_test < 0:
            # 调整训练集
            n_train = max(0, n_train + n_test)
            n_test = n - n_train - n_val

        # 随机打乱该组
        group = _shuffle_records(group, seed + label)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:n_train + n_val + n_test])

    return train, val, test


def _random_split(
    records: List[RawRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[RawRecord], List[RawRecord], List[RawRecord]]:
    """随机划分数据集"""
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    shuffled = _shuffle_records(records, seed)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]

    return train, val, test


def split_dataset(
    records: List[RawRecord],
    config: Optional[SplitConfig] = None
) -> Tuple[List[RawRecord], List[RawRecord], List[RawRecord]]:
    """
    划分数据集为训练/验证/测试集

    Args:
        records: 原始记录列表
        config: 划分配置，默认 70/15/15 分层划分

    Returns:
        (train_records, val_records, test_records)
    """
    if config is None:
        config = SplitConfig()

    if not records:
        return [], [], []

    if config.stratify_by_label:
        train, val, test = _stratified_split(
            records,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
            config.random_seed
        )
    else:
        train, val, test = _random_split(
            records,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
            config.random_seed
        )

    # 再次打乱各集合（可选）
    if config.shuffle:
        train = _shuffle_records(train, config.random_seed)
        val = _shuffle_records(val, config.random_seed + 1)
        test = _shuffle_records(test, config.random_seed + 2)

    logger.info(
        f"Dataset split: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(total={len(records)})"
    )

    return train, val, test


# ==================== JSONL 落盘 ====================

def _record_to_dict(record: Union[RawRecord, ProcessedRecord]) -> dict:
    """将记录转换为可 JSON 序列化的字典"""
    # 使用 mode='json' 将 datetime 等对象转换为字符串
    return record.model_dump(mode='json')


def save_to_jsonl(
    records: List[Union[RawRecord, ProcessedRecord]],
    filepath: str,
    ensure_ascii: bool = False
) -> int:
    """
    将记录保存为 JSONL 格式

    Args:
        records: 记录列表
        filepath: 输出文件路径
        ensure_ascii: 是否将非 ASCII 字符转义

    Returns:
        写入的记录数
    """
    count = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            data = _record_to_dict(record)
            json_line = json.dumps(data, ensure_ascii=ensure_ascii)
            f.write(json_line + '\n')
            count += 1

    logger.info(f"Saved {count} records to {filepath}")
    return count


def load_from_jsonl(
    filepath: str,
    record_type: str = "raw"
) -> List[Union[RawRecord, ProcessedRecord]]:
    """
    从 JSONL 文件加载记录

    Args:
        filepath: JSONL 文件路径
        record_type: "raw" 或 "processed"

    Returns:
        记录列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: record_type 无效
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if record_type not in ("raw", "processed"):
        raise ValueError(f"record_type must be 'raw' or 'processed', got {record_type}")

    records = []
    record_class = RawRecord if record_type == "raw" else ProcessedRecord

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                record = record_class(**data)
                records.append(record)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error parsing line {line_num} in {filepath}: {e}")
                continue

    logger.info(f"Loaded {len(records)} {record_type} records from {filepath}")
    return records


# ==================== 多语言数据合并 ====================

def merge_multilingual_data(
    language_data: Dict[str, List[RawRecord]],
    shuffle: bool = True,
    random_seed: int = 42
) -> List[RawRecord]:
    """
    合并多语言数据集

    Args:
        language_data: 语言代码到记录列表的映射，如 {"zh": [...], "en": [...]}
        shuffle: 是否随机打乱合并后的顺序
        random_seed: 随机种子

    Returns:
        合并后的记录列表
    """
    merged = []

    for lang, records in language_data.items():
        if records:
            merged.extend(records)
            logger.debug(f"Added {len(records)} records for language '{lang}'")

    if shuffle and merged:
        merged = _shuffle_records(merged, random_seed)

    logger.info(f"Merged {len(language_data)} languages, total {len(merged)} records")
    return merged


# ==================== 数据集验证 ====================

def _count_labels(records: List[RawRecord]) -> Dict[int, int]:
    """统计各标签数量"""
    counts = defaultdict(int)
    for r in records:
        counts[r.sentiment_label] += 1
    return dict(counts)


def _check_no_overlap(
    train: List[RawRecord],
    val: List[RawRecord],
    test: List[RawRecord]
) -> Tuple[bool, List[str]]:
    """检查数据集之间是否有重叠"""
    train_ids = {r.id for r in train}
    val_ids = {r.id for r in val}
    test_ids = {r.id for r in test}

    errors = []

    overlap_train_val = train_ids & val_ids
    if overlap_train_val:
        errors.append(f"Overlap between train and val: {len(overlap_train_val)} records")

    overlap_train_test = train_ids & test_ids
    if overlap_train_test:
        errors.append(f"Overlap between train and test: {len(overlap_train_test)} records")

    overlap_val_test = val_ids & test_ids
    if overlap_val_test:
        errors.append(f"Overlap between val and test: {len(overlap_val_test)} records")

    return len(errors) == 0, errors


def _calculate_distribution_diff(
    dist1: Dict[int, int],
    dist2: Dict[int, int]
) -> float:
    """计算两个分布之间的差异（L1距离）"""
    all_labels = set(dist1.keys()) | set(dist2.keys())

    total1 = sum(dist1.values())
    total2 = sum(dist2.values())

    if total1 == 0 or total2 == 0:
        return float('inf')

    diff = 0.0
    for label in all_labels:
        p1 = dist1.get(label, 0) / total1
        p2 = dist2.get(label, 0) / total2
        diff += abs(p1 - p2)

    return diff / 2  # 归一化到 [0, 1]


def validate_dataset_split(
    train: List[RawRecord],
    val: List[RawRecord],
    test: List[RawRecord],
    min_samples_per_split: int = 1,
    check_distribution: bool = True,
    distribution_threshold: float = 0.3
) -> Dict:
    """
    验证数据集划分的有效性

    Args:
        train: 训练集
        val: 验证集
        test: 测试集
        min_samples_per_split: 每个划分最少样本数
        check_distribution: 是否检查标签分布一致性
        distribution_threshold: 分布差异阈值

    Returns:
        验证结果字典，包含 is_valid, errors, warnings, statistics
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {
            "train": {"count": len(train), "labels": _count_labels(train)},
            "val": {"count": len(val), "labels": _count_labels(val)},
            "test": {"count": len(test), "labels": _count_labels(test)}
        }
    }

    # 检查最小样本数
    for name, records in [("train", train), ("val", val), ("test", test)]:
        if len(records) < min_samples_per_split:
            result["errors"].append(
                f"{name} set has {len(records)} samples, minimum required: {min_samples_per_split}"
            )
            result["is_valid"] = False

    # 检查无重叠
    no_overlap, overlap_errors = _check_no_overlap(train, val, test)
    if not no_overlap:
        result["errors"].extend(overlap_errors)
        result["is_valid"] = False

    # 检查标签分布
    if check_distribution and train and val and test:
        train_labels = result["statistics"]["train"]["labels"]
        val_labels = result["statistics"]["val"]["labels"]
        test_labels = result["statistics"]["test"]["labels"]

        val_diff = _calculate_distribution_diff(train_labels, val_labels)
        test_diff = _calculate_distribution_diff(train_labels, test_labels)

        if val_diff > distribution_threshold:
            result["warnings"].append(
                f"Validation label distribution differs from train: {val_diff:.2f}"
            )
        if test_diff > distribution_threshold:
            result["warnings"].append(
                f"Test label distribution differs from train: {test_diff:.2f}"
            )

    return result


# ==================== 便捷函数 ====================

def save_split_dataset(
    train: List[RawRecord],
    val: List[RawRecord],
    test: List[RawRecord],
    output_dir: str,
    prefix: str = "",
    record_type: str = "raw"
) -> Dict[str, str]:
    """
    保存划分后的数据集到 JSONL 文件

    Args:
        train: 训练集
        val: 验证集
        test: 测试集
        output_dir: 输出目录
        prefix: 文件名前缀
        record_type: 记录类型

    Returns:
        文件路径字典 {"train": ..., "val": ..., "test": ...}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    splits = [("train", train), ("val", val), ("test", test)]

    for name, records in splits:
        filename = f"{prefix}{name}.jsonl" if prefix else f"{name}.jsonl"
        filepath = os.path.join(output_dir, filename)
        save_to_jsonl(records, filepath)
        paths[name] = filepath

    return paths


# 延迟导入 os（避免循环导入问题）
import os


# ==================== 测试入口 ====================

if __name__ == "__main__":
    print("Testing merger module...\n")

    # 创建测试数据
    test_records = [
        RawRecord(
            language="zh" if i % 2 == 0 else "en",
            source="test",
            original_text=f"text_{i}",
            sentiment_label=i % 3
        )
        for i in range(100)
    ]

    # 测试划分
    print("Test 1: Dataset split")
    train, val, test = split_dataset(test_records)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 测试验证
    print("\nTest 2: Validate split")
    result = validate_dataset_split(train, val, test)
    print(f"  Valid: {result['is_valid']}")
    print(f"  Stats: {result['statistics']}")

    # 测试多语言合并
    print("\nTest 3: Merge multilingual")
    merged = merge_multilingual_data({
        "zh": [r for r in test_records if r.language == "zh"],
        "en": [r for r in test_records if r.language == "en"]
    })
    print(f"  Merged: {len(merged)} records")

    print("\n✓ All merger tests completed!")
