"""
测试数据契约 (schema.py)

使用 unittest 和 pydantic 验证 RawRecord 和 ProcessedRecord 的正确性。
"""

import unittest
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.schema import RawRecord, ProcessedRecord, classify_length


class TestRawRecord(unittest.TestCase):
    """测试 RawRecord 数据契约"""

    def test_valid_raw_record_creation(self):
        """测试创建有效的 RawRecord"""
        record = RawRecord(
            language="zh",
            source="amazon_hf",
            original_text="这个产品很好用",
            sentiment_label=2,
            rating=5,
            product_id="B12345",
            category="Electronics"
        )

        self.assertEqual(record.language, "zh")
        self.assertEqual(record.source, "amazon_hf")
        self.assertEqual(record.original_text, "这个产品很好用")
        self.assertEqual(record.sentiment_label, 2)
        self.assertEqual(record.rating, 5)
        self.assertEqual(record.product_id, "B12345")
        self.assertEqual(record.category, "Electronics")
        # UUID 应该自动生成
        self.assertIsNotNone(record.id)
        self.assertEqual(len(record.id), 36)  # UUID 长度

    def test_text_validation_and_strip(self):
        """测试文本验证和自动去除首尾空白"""
        record = RawRecord(
            language="en",
            source="amazon_hf",
            original_text="  Good product!  ",
            sentiment_label=2
        )
        # 首尾空白应该被去除
        self.assertEqual(record.original_text, "Good product!")

    def test_empty_text_raises_error(self):
        """测试空文本应该抛出异常"""
        with self.assertRaises(ValueError) as context:
            RawRecord(
                language="zh",
                source="amazon_hf",
                original_text="   ",  # 只有空白字符
                sentiment_label=0
            )
        self.assertIn("cannot be empty", str(context.exception))

    def test_invalid_language(self):
        """测试无效语言代码"""
        with self.assertRaises(ValueError):
            RawRecord(
                language="fr",  # 不支持法语
                source="amazon_hf",
                original_text="Bonjour",
                sentiment_label=0
            )

    def test_invalid_sentiment_label(self):
        """测试无效的情感标签"""
        with self.assertRaises(ValueError):
            RawRecord(
                language="zh",
                source="amazon_hf",
                original_text="测试",
                sentiment_label=5  # 只能是 0, 1, 2
            )

    def test_invalid_rating(self):
        """测试无效的星级评分"""
        with self.assertRaises(ValueError):
            RawRecord(
                language="zh",
                source="amazon_hf",
                original_text="测试",
                sentiment_label=0,
                rating=6  # 只能是 1-5
            )

    def test_optional_fields(self):
        """测试可选字段可以为 None"""
        record = RawRecord(
            language="ru",
            source="ozon_local",
            original_text="Хороший товар",
            sentiment_label=2
        )
        # 可选字段默认为 None
        self.assertIsNone(record.rating)
        self.assertIsNone(record.product_id)
        self.assertIsNone(record.category)


class TestProcessedRecord(unittest.TestCase):
    """测试 ProcessedRecord 数据契约"""

    def setUp(self):
        """设置基础 RawRecord"""
        self.base_record = RawRecord(
            id="test-uuid-123",
            language="zh",
            source="amazon_hf",
            original_text="这个产品非常好用！",
            sentiment_label=2,
            rating=5,
            product_id="B12345",
            category="Electronics"
        )

    def test_valid_processed_record(self):
        """测试创建有效的 ProcessedRecord"""
        processed = ProcessedRecord(
            **self.base_record.model_dump(),
            text_for_nlp="产品 非常 好用",
            text_for_llm="这个产品非常好用！",
            word_count=3,
            char_count=10,
            length_category="short"
        )

        self.assertEqual(processed.text_for_nlp, "产品 非常 好用")
        self.assertEqual(processed.text_for_llm, "这个产品非常好用！")
        self.assertEqual(processed.word_count, 3)
        self.assertEqual(processed.length_category, "short")

    def test_soft_label_validation(self):
        """测试软标签验证"""
        # 有效的软标签
        processed = ProcessedRecord(
            **self.base_record.model_dump(),
            text_for_nlp="test",
            text_for_llm="test",
            word_count=1,
            char_count=4,
            length_category="short",
            soft_label=[0.1, 0.2, 0.7]
        )
        self.assertEqual(processed.soft_label, [0.1, 0.2, 0.7])

    def test_invalid_soft_label_wrong_length(self):
        """测试无效软标签：长度不正确"""
        with self.assertRaises(ValueError) as context:
            ProcessedRecord(
                **self.base_record.model_dump(),
                text_for_nlp="test",
                text_for_llm="test",
                word_count=1,
                char_count=4,
                length_category="short",
                soft_label=[0.5, 0.5]  # 只有2个元素
            )
        self.assertIn("exactly 3 elements", str(context.exception))

    def test_invalid_soft_label_wrong_sum(self):
        """测试无效软标签：概率和不等于1"""
        with self.assertRaises(ValueError) as context:
            ProcessedRecord(
                **self.base_record.model_dump(),
                text_for_nlp="test",
                text_for_llm="test",
                word_count=1,
                char_count=4,
                length_category="short",
                soft_label=[0.3, 0.3, 0.3]  # 和为0.9
            )
        self.assertIn("sum to 1", str(context.exception))

    def test_invalid_soft_label_out_of_range(self):
        """测试无效软标签：概率超出范围"""
        with self.assertRaises(ValueError):
            ProcessedRecord(
                **self.base_record.model_dump(),
                text_for_nlp="test",
                text_for_llm="test",
                word_count=1,
                char_count=4,
                length_category="short",
                soft_label=[1.5, -0.5, 0.0]  # 超出[0,1]范围
            )

    def test_length_category_consistency(self):
        """测试长度分类与 word_count 一致性验证"""
        # 正确的 short 分类
        with self.assertRaises(ValueError):
            ProcessedRecord(
                **self.base_record.model_dump(),
                text_for_nlp="test",
                text_for_llm="test",
                word_count=50,  # 应该是 medium
                char_count=4,
                length_category="short"  # 错误的分类
            )

    def test_inheritance_from_raw(self):
        """测试 ProcessedRecord 正确继承 RawRecord 字段"""
        processed = ProcessedRecord(
            **self.base_record.model_dump(),
            text_for_nlp="词1 词2",
            text_for_llm="原文本",
            word_count=2,
            char_count=3,
            length_category="short"
        )

        # 验证继承的字段
        self.assertEqual(processed.id, "test-uuid-123")
        self.assertEqual(processed.language, "zh")
        self.assertEqual(processed.source, "amazon_hf")
        self.assertEqual(processed.original_text, "这个产品非常好用！")
        self.assertEqual(processed.sentiment_label, 2)


class TestClassifyLength(unittest.TestCase):
    """测试长度分类函数"""

    def test_short_text(self):
        """测试短文本分类"""
        self.assertEqual(classify_length(5), "short")
        self.assertEqual(classify_length(10), "short")
        self.assertEqual(classify_length(15), "short")  # 边界值

    def test_medium_text(self):
        """测试中长文本分类"""
        self.assertEqual(classify_length(16), "medium")
        self.assertEqual(classify_length(30), "medium")
        self.assertEqual(classify_length(50), "medium")  # 边界值

    def test_long_text(self):
        """测试长文本分类"""
        self.assertEqual(classify_length(51), "long")
        self.assertEqual(classify_length(100), "long")

    def test_custom_thresholds(self):
        """测试自定义阈值"""
        self.assertEqual(classify_length(10, short_threshold=5, long_threshold=20), "medium")
        self.assertEqual(classify_length(3, short_threshold=5, long_threshold=20), "short")
        self.assertEqual(classify_length(25, short_threshold=5, long_threshold=20), "long")


class TestRecordSerialization(unittest.TestCase):
    """测试记录的序列化和反序列化"""

    def test_raw_record_to_dict(self):
        """测试 RawRecord 转换为字典"""
        record = RawRecord(
            language="en",
            source="amazon_hf",
            original_text="Great product",
            sentiment_label=2
        )
        data = record.model_dump()

        self.assertEqual(data["language"], "en")
        self.assertEqual(data["sentiment_label"], 2)
        self.assertIn("id", data)
        self.assertIn("created_at", data)

    def test_raw_record_to_json(self):
        """测试 RawRecord 转换为 JSON"""
        record = RawRecord(
            language="zh",
            source="ozon_local",
            original_text="测试",
            sentiment_label=1
        )
        json_str = record.model_dump_json()

        self.assertIn("\"language\":\"zh\"", json_str)
        self.assertIn("\"sentiment_label\":1", json_str)

    def test_processed_record_serialization(self):
        """测试 ProcessedRecord 序列化包含所有字段"""
        raw = RawRecord(
            id="test-id",
            language="ru",
            source="ozon_local",
            original_text="Тестовый текст",
            sentiment_label=0
        )
        processed = ProcessedRecord(
            **raw.model_dump(),
            text_for_nlp="тестовый текст",
            text_for_llm="Тестовый текст",
            word_count=2,
            char_count=14,
            length_category="short",
            soft_label=[0.8, 0.15, 0.05],
            rationale="测试推理"
        )

        data = processed.model_dump()
        self.assertEqual(data["soft_label"], [0.8, 0.15, 0.05])
        self.assertEqual(data["rationale"], "测试推理")
        self.assertIn("processed_at", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
