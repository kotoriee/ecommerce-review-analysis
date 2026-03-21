"""
测试数据集合并与划分模块 (merger.py)

使用 unittest 验证数据划分、JSONL 落盘功能的正确性。
"""

import unittest
import tempfile
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.merger import (
    split_dataset,
    save_to_jsonl,
    load_from_jsonl,
    merge_multilingual_data,
    validate_dataset_split,
    SplitConfig
)
from data.schema import RawRecord, ProcessedRecord


class TestSplitConfig(unittest.TestCase):
    """测试划分配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SplitConfig()
        self.assertEqual(config.train_ratio, 0.7)
        self.assertEqual(config.val_ratio, 0.15)
        self.assertEqual(config.test_ratio, 0.15)
        self.assertTrue(config.stratify_by_label)
        self.assertTrue(config.shuffle)

    def test_custom_config(self):
        """测试自定义配置"""
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        self.assertEqual(config.train_ratio, 0.8)
        self.assertEqual(config.val_ratio, 0.1)
        self.assertEqual(config.test_ratio, 0.1)

    def test_invalid_ratio(self):
        """测试无效比例应抛出异常"""
        with self.assertRaises(ValueError):
            SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestSplitDataset(unittest.TestCase):
    """测试数据集划分功能"""

    def setUp(self):
        """创建测试数据"""
        self.records = [
            RawRecord(
                language="zh",
                source="amazon_hf",
                original_text=f"测试文本{i}",
                sentiment_label=i % 3
            )
            for i in range(100)
        ]

    def test_split_sizes(self):
        """测试划分后的数据集大小"""
        config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        train, val, test = split_dataset(self.records, config)

        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)

    def test_no_overlap(self):
        """测试划分后的数据集无重叠"""
        config = SplitConfig()
        train, val, test = split_dataset(self.records, config)

        train_ids = {r.id for r in train}
        val_ids = {r.id for r in val}
        test_ids = {r.id for r in test}

        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)

    def test_empty_dataset(self):
        """测试空数据集"""
        train, val, test = split_dataset([])
        self.assertEqual(len(train), 0)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)

    def test_small_dataset(self):
        """测试小数据集处理"""
        small_records = self.records[:5]
        train, val, test = split_dataset(small_records)

        total = len(train) + len(val) + len(test)
        self.assertEqual(total, 5)


class TestSaveAndLoadJsonl(unittest.TestCase):
    """测试 JSONL 保存和加载"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.records = [
            RawRecord(
                language="en",
                source="amazon_hf",
                original_text="Great product",
                sentiment_label=2,
                rating=5
            ),
            RawRecord(
                language="zh",
                source="ozon_local",
                original_text="测试",
                sentiment_label=1
            )
        ]

    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_raw_records(self):
        """测试保存 RawRecord"""
        filepath = os.path.join(self.temp_dir, "test.jsonl")
        save_to_jsonl(self.records, filepath)

        self.assertTrue(os.path.exists(filepath))

        # 验证内容
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

            data = json.loads(lines[0])
            self.assertEqual(data["language"], "en")
            self.assertEqual(data["sentiment_label"], 2)

    def test_save_processed_records(self):
        """测试保存 ProcessedRecord"""
        processed = [
            ProcessedRecord(
                **self.records[0].model_dump(),
                text_for_nlp="great product",
                text_for_llm="Great product",
                word_count=2,
                char_count=13,
                length_category="short"
            )
        ]

        filepath = os.path.join(self.temp_dir, "processed.jsonl")
        save_to_jsonl(processed, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            self.assertEqual(data["text_for_nlp"], "great product")
            self.assertEqual(data["word_count"], 2)

    def test_load_from_jsonl(self):
        """测试从 JSONL 加载"""
        filepath = os.path.join(self.temp_dir, "test.jsonl")
        save_to_jsonl(self.records, filepath)

        loaded = load_from_jsonl(filepath, record_type="raw")
        self.assertEqual(len(loaded), 2)
        self.assertIsInstance(loaded[0], RawRecord)
        self.assertEqual(loaded[0].language, "en")

    def test_load_processed_records(self):
        """测试加载 ProcessedRecord"""
        processed = [
            ProcessedRecord(
                **self.records[0].model_dump(),
                text_for_nlp="great product",
                text_for_llm="Great product",
                word_count=2,
                char_count=13,
                length_category="short"
            )
        ]

        filepath = os.path.join(self.temp_dir, "processed.jsonl")
        save_to_jsonl(processed, filepath)

        loaded = load_from_jsonl(filepath, record_type="processed")
        self.assertEqual(len(loaded), 1)
        self.assertIsInstance(loaded[0], ProcessedRecord)
        self.assertEqual(loaded[0].text_for_nlp, "great product")

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            load_from_jsonl("/nonexistent/path.jsonl")

    def test_round_trip(self):
        """测试保存后加载的一致性"""
        filepath = os.path.join(self.temp_dir, "roundtrip.jsonl")
        save_to_jsonl(self.records, filepath)
        loaded = load_from_jsonl(filepath, record_type="raw")

        self.assertEqual(len(loaded), len(self.records))
        for orig, load in zip(self.records, loaded):
            self.assertEqual(orig.id, load.id)
            self.assertEqual(orig.language, load.language)
            self.assertEqual(orig.original_text, load.original_text)
            self.assertEqual(orig.sentiment_label, load.sentiment_label)


class TestMergeMultilingualData(unittest.TestCase):
    """测试多语言数据合并"""

    def setUp(self):
        """创建多语言测试数据"""
        self.zh_records = [
            RawRecord(language="zh", source="amazon_hf", original_text="好", sentiment_label=2)
            for _ in range(10)
        ]
        self.en_records = [
            RawRecord(language="en", source="amazon_hf", original_text="good", sentiment_label=2)
            for _ in range(10)
        ]
        self.ru_records = [
            RawRecord(language="ru", source="ozon_local", original_text="хорошо", sentiment_label=2)
            for _ in range(10)
        ]

    def test_merge_all(self):
        """测试合并所有语言"""
        merged = merge_multilingual_data({
            "zh": self.zh_records,
            "en": self.en_records,
            "ru": self.ru_records
        })

        self.assertEqual(len(merged), 30)
        languages = {r.language for r in merged}
        self.assertEqual(languages, {"zh", "en", "ru"})

    def test_merge_partial(self):
        """测试部分语言合并"""
        merged = merge_multilingual_data({
            "zh": self.zh_records,
            "en": self.en_records
        })

        self.assertEqual(len(merged), 20)

    def test_empty_language(self):
        """测试包含空列表的语言"""
        merged = merge_multilingual_data({
            "zh": self.zh_records,
            "en": []
        })

        self.assertEqual(len(merged), 10)

    def test_empty_dict(self):
        """测试空字典"""
        merged = merge_multilingual_data({})
        self.assertEqual(len(merged), 0)

    def test_shuffle_preserves_all(self):
        """测试 shuffle 后保留所有数据"""
        merged = merge_multilingual_data({
            "zh": self.zh_records,
            "en": self.en_records
        }, shuffle=True)

        self.assertEqual(len(merged), 20)
        zh_count = sum(1 for r in merged if r.language == "zh")
        en_count = sum(1 for r in merged if r.language == "en")
        self.assertEqual(zh_count, 10)
        self.assertEqual(en_count, 10)


class TestValidateDatasetSplit(unittest.TestCase):
    """测试数据集划分验证"""

    def test_valid_split(self):
        """测试有效划分"""
        train = [RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=i % 3) for i in range(60)]
        val = [RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=i % 3) for i in range(20)]
        test = [RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=i % 3) for i in range(20)]

        result = validate_dataset_split(train, val, test)
        self.assertTrue(result["is_valid"])

    def test_empty_split(self):
        """测试空划分"""
        result = validate_dataset_split([], [], [], min_samples_per_split=1)
        self.assertFalse(result["is_valid"])

    def test_label_distribution(self):
        """测试标签分布检查"""
        # 创建不平衡的数据集
        train = [RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=2) for _ in range(100)]
        val = [RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=0) for _ in range(10)]
        test = [RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=1) for _ in range(10)]

        result = validate_dataset_split(train, val, test, check_distribution=True)
        # 分布差异大时应发出警告
        self.assertIn("warnings", result)

    def test_no_overlap_validation(self):
        """测试无重叠验证"""
        record = RawRecord(language="zh", source="amazon_hf", original_text="t", sentiment_label=1)
        train = [record]
        val = [record]  # 相同记录
        test = []

        result = validate_dataset_split(train, val, test)
        self.assertFalse(result["is_valid"])


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.records = [
            RawRecord(
                language=lang,
                source="amazon_hf",
                original_text=f"text_{i}",
                sentiment_label=i % 3
            )
            for lang in ["zh", "en", "ru"]
            for i in range(30)
        ]

    def tearDown(self):
        """清理"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline(self):
        """测试完整流程：划分 -> 保存 -> 加载 -> 验证"""
        # 1. 划分
        config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        train, val, test = split_dataset(self.records, config)

        # 2. 保存
        train_path = os.path.join(self.temp_dir, "train.jsonl")
        val_path = os.path.join(self.temp_dir, "val.jsonl")
        test_path = os.path.join(self.temp_dir, "test.jsonl")

        save_to_jsonl(train, train_path)
        save_to_jsonl(val, val_path)
        save_to_jsonl(test, test_path)

        # 3. 加载
        loaded_train = load_from_jsonl(train_path, record_type="raw")
        loaded_val = load_from_jsonl(val_path, record_type="raw")
        loaded_test = load_from_jsonl(test_path, record_type="raw")

        # 4. 验证
        self.assertEqual(len(loaded_train), len(train))
        self.assertEqual(len(loaded_val), len(val))
        self.assertEqual(len(loaded_test), len(test))

        # 5. 数据集验证
        result = validate_dataset_split(loaded_train, loaded_val, loaded_test)
        self.assertTrue(result["is_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
