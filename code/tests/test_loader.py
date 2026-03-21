"""
测试数据加载器 (loader.py)

使用 unittest 和 mock 验证数据加载功能的正确性。
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (
    map_rating_to_label,
    load_local_csv,
    load_local_jsonl,
    load_multilingual_dataset
)
from data.schema import RawRecord


class TestMapRatingToLabel(unittest.TestCase):
    """测试星级到标签的映射函数"""

    def test_negative_ratings(self):
        """测试负面评分映射"""
        self.assertEqual(map_rating_to_label(1), 0)  # 1星 -> 负面
        self.assertEqual(map_rating_to_label(2), 0)  # 2星 -> 负面

    def test_neutral_rating(self):
        """测试中性评分映射"""
        self.assertEqual(map_rating_to_label(3), 1)  # 3星 -> 中性

    def test_positive_ratings(self):
        """测试正面评分映射"""
        self.assertEqual(map_rating_to_label(4), 2)  # 4星 -> 正面
        self.assertEqual(map_rating_to_label(5), 2)  # 5星 -> 正面

    def test_invalid_ratings(self):
        """测试无效评分抛出异常"""
        with self.assertRaises(ValueError):
            map_rating_to_label(0)   # 0星无效
        with self.assertRaises(ValueError):
            map_rating_to_label(6)   # 6星无效
        with self.assertRaises(ValueError):
            map_rating_to_label(-1)  # 负数无效


class TestLoadLocalCSV(unittest.TestCase):
    """测试本地 CSV 加载功能"""

    def setUp(self):
        """创建临时 CSV 文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, "test_data.csv")

        # 创建测试 CSV
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'rating', 'label'])
            writer.writeheader()
            writer.writerow({
                'text': 'Great product!',
                'rating': '5',
                'label': '2'
            })
            writer.writerow({
                'text': 'Bad quality',
                'rating': '1',
                'label': '0'
            })
            writer.writerow({
                'text': 'Average item',
                'rating': '3',
                'label': '1'
            })

    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_csv_with_label_column(self):
        """测试使用 label 列加载"""
        records = load_local_csv(
            self.csv_file,
            language="en",
            text_column="text",
            label_column="label"
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].original_text, "Great product!")
        self.assertEqual(records[0].sentiment_label, 2)
        self.assertEqual(records[0].language, "en")
        self.assertEqual(records[0].source, "ozon_local")

    def test_load_csv_with_rating_column(self):
        """测试使用 rating 列计算标签"""
        records = load_local_csv(
            self.csv_file,
            language="en",
            text_column="text",
            rating_column="rating"
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].sentiment_label, 2)  # 5星 -> 正面
        self.assertEqual(records[1].sentiment_label, 0)  # 1星 -> 负面
        self.assertEqual(records[2].sentiment_label, 1)  # 3星 -> 中性

    def test_load_csv_with_limit(self):
        """测试加载数量限制"""
        records = load_local_csv(
            self.csv_file,
            language="en",
            text_column="text",
            label_column="label",
            n_samples=2
        )

        self.assertEqual(len(records), 2)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件抛出异常"""
        with self.assertRaises(FileNotFoundError):
            load_local_csv("/nonexistent/path.csv", language="en")

    def test_skip_empty_text(self):
        """测试跳过空文本行"""
        # 创建包含空文本的 CSV
        csv_with_empty = os.path.join(self.temp_dir, "with_empty.csv")
        with open(csv_with_empty, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label'])
            writer.writeheader()
            writer.writerow({'text': 'Valid text', 'label': '2'})
            writer.writerow({'text': '   ', 'label': '1'})  # 空文本
            writer.writerow({'text': '', 'label': '2'})    # 空文本

        records = load_local_csv(csv_with_empty, language="en", text_column="text", label_column="label")
        self.assertEqual(len(records), 1)


class TestLoadLocalJSONL(unittest.TestCase):
    """测试本地 JSONL 加载功能"""

    def setUp(self):
        """创建临时 JSONL 文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.jsonl_file = os.path.join(self.temp_dir, "test_data.jsonl")

        # 创建测试 JSONL
        with open(self.jsonl_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "id": "uuid-1",
                "language": "zh",
                "source": "amazon_hf",
                "original_text": "很好",
                "sentiment_label": 2
            }) + "\n")
            f.write(json.dumps({
                "id": "uuid-2",
                "language": "en",
                "source": "amazon_hf",
                "original_text": "Bad",
                "sentiment_label": 0
            }) + "\n")

    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_jsonl(self):
        """测试加载 JSONL 文件"""
        records = load_local_jsonl(self.jsonl_file)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].language, "zh")
        self.assertEqual(records[0].sentiment_label, 2)
        self.assertEqual(records[1].language, "en")
        self.assertEqual(records[1].sentiment_label, 0)

    def test_load_jsonl_with_limit(self):
        """测试加载数量限制"""
        records = load_local_jsonl(self.jsonl_file, n_samples=1)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].language, "zh")

    def test_skip_invalid_lines(self):
        """测试跳过无效行"""
        # 创建包含无效行的 JSONL
        invalid_jsonl = os.path.join(self.temp_dir, "invalid.jsonl")
        with open(invalid_jsonl, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"language": "zh", "source": "amazon_hf", "original_text": "ok", "sentiment_label": 2}) + "\n")
            f.write("invalid json line\n")  # 无效行
            f.write(json.dumps({"language": "en", "source": "amazon_hf", "original_text": "ok", "sentiment_label": 0}) + "\n")

        records = load_local_jsonl(invalid_jsonl)
        self.assertEqual(len(records), 2)  # 跳过1行无效行


class TestLoadMultilingualDataset(unittest.TestCase):
    """测试多语言数据集加载"""

    @patch('data.loader.fetch_hf_dataset')
    def test_load_zh_and_en(self, mock_fetch_hf):
        """测试加载中英文数据"""
        # Mock HF 数据返回
        mock_fetch_hf.return_value = [
            RawRecord(language="zh", source="amazon_hf", original_text="好", sentiment_label=2),
            RawRecord(language="zh", source="amazon_hf", original_text="差", sentiment_label=0),
        ]

        records = load_multilingual_dataset(
            languages=["zh", "en"],
            samples_per_lang=100
        )

        # 验证 fetch_hf_dataset 被调用两次
        self.assertEqual(mock_fetch_hf.call_count, 2)
        mock_fetch_hf.assert_any_call("zh", n_samples=100)
        mock_fetch_hf.assert_any_call("en", n_samples=100)

    @patch('data.loader.load_local_csv')
    def test_load_russian_from_local(self, mock_load_csv):
        """测试从本地加载俄文数据"""
        mock_load_csv.return_value = [
            RawRecord(language="ru", source="ozon_local", original_text="хорошо", sentiment_label=2),
        ]

        records = load_multilingual_dataset(
            languages=["ru"],
            samples_per_lang=50,
            local_ru_path="/path/to/ru.csv"
        )

        mock_load_csv.assert_called_once_with("/path/to/ru.csv", language="ru", n_samples=50)
        self.assertEqual(len(records), 1)

    @patch('data.loader.fetch_hf_dataset')
    def test_handle_hf_errors(self, mock_fetch_hf):
        """测试处理 HF 加载错误"""
        # 第一个调用成功，第二个失败
        mock_fetch_hf.side_effect = [
            [RawRecord(language="zh", source="amazon_hf", original_text="好", sentiment_label=2)],
            Exception("Network error")
        ]

        records = load_multilingual_dataset(
            languages=["zh", "en"],
            samples_per_lang=100
        )

        # 应该返回中文数据，跳过英文
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].language, "zh")


class TestRawRecordConversion(unittest.TestCase):
    """测试 RawRecord 转换逻辑"""

    def test_record_has_uuid(self):
        """测试记录自动生成 UUID"""
        record1 = RawRecord(
            language="zh",
            source="amazon_hf",
            original_text="测试1",
            sentiment_label=2
        )
        record2 = RawRecord(
            language="zh",
            source="amazon_hf",
            original_text="测试2",
            sentiment_label=0
        )

        # UUID 应该不同
        self.assertNotEqual(record1.id, record2.id)
        # UUID 格式正确
        self.assertEqual(len(record1.id), 36)

    def test_record_has_timestamp(self):
        """测试记录包含创建时间"""
        record = RawRecord(
            language="en",
            source="amazon_hf",
            original_text="Test",
            sentiment_label=2
        )

        self.assertIsNotNone(record.created_at)


if __name__ == "__main__":
    unittest.main(verbosity=2)
