"""
测试双流清洗架构 (preprocessor.py)

使用 unittest 和 mock 验证清洗功能的正确性。
遵循 CLAUDE.md 约束：
- 中文: jieba
- 英文: nltk
- 俄文: natasha
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import (
    clean_for_llm,
    clean_for_nlp,
    tokenize_chinese,
    tokenize_english,
    tokenize_russian,
    process_record,
    process_batch,
    StopwordManager,
    classify_length
)
from data.schema import RawRecord, ProcessedRecord


class TestCleanForLLM(unittest.TestCase):
    """测试 LLM 轻度清洗"""

    def test_remove_urls(self):
        """测试去除 URL"""
        text = "Check https://example.com for details"
        result = clean_for_llm(text)
        self.assertNotIn("https://example.com", result)
        self.assertIn("Check", result)
        self.assertIn("for details", result)

    def test_remove_emails(self):
        """测试去除邮箱"""
        text = "Contact me at test@email.com please"
        result = clean_for_llm(text)
        self.assertNotIn("test@email.com", result)
        self.assertIn("Contact me", result)

    def test_remove_html_tags(self):
        """测试去除 HTML 标签"""
        text = "<p>This is a <b>test</b></p>"
        result = clean_for_llm(text)
        self.assertNotIn("<p>", result)
        self.assertNotIn("</p>", result)
        self.assertNotIn("<b>", result)
        self.assertIn("This is a test", result)

    def test_normalize_whitespace(self):
        """测试规范化空白"""
        text = "Multiple    spaces   and\n\nnewlines"
        result = clean_for_llm(text)
        self.assertNotIn("    ", result)  # 多个空格
        self.assertIn("Multiple spaces", result)

    def test_html_entities(self):
        """测试 HTML 实体解码和 HTML 标签去除"""
        text = "Test &amp; Example &lt;script&gt;alert(1)&lt;/script&gt;"
        result = clean_for_llm(text)
        self.assertIn("&", result)  # &amp; -> &
        self.assertNotIn("<script>", result)  # HTML 标签被去除
        self.assertIn("Test", result)
        self.assertIn("Example", result)

    def test_empty_text(self):
        """测试空文本"""
        self.assertEqual(clean_for_llm(""), "")
        self.assertEqual(clean_for_llm("   "), "")

    def test_strip_whitespace(self):
        """测试去除首尾空白"""
        text = "  Hello World  "
        result = clean_for_llm(text)
        self.assertEqual(result, "Hello World")


class TestStopwordManager(unittest.TestCase):
    """测试停用词管理器"""

    def setUp(self):
        self.manager = StopwordManager()

    def test_zh_stopwords(self):
        """测试中文停用词"""
        zh_stopwords = self.manager.get_stopwords("zh")
        self.assertIn("的", zh_stopwords)
        self.assertIn("了", zh_stopwords)
        self.assertIn("商品", zh_stopwords)  # 电商领域

    def test_en_stopwords(self):
        """测试英文停用词"""
        en_stopwords = self.manager.get_stopwords("en")
        self.assertIn("the", en_stopwords)
        self.assertIn("and", en_stopwords)
        self.assertIn("product", en_stopwords)  # 电商领域

    def test_ru_stopwords(self):
        """测试俄文停用词"""
        ru_stopwords = self.manager.get_stopwords("ru")
        self.assertIn("и", ru_stopwords)
        self.assertIn("в", ru_stopwords)
        self.assertIn("товар", ru_stopwords)  # 电商领域

    def test_unknown_language(self):
        """测试未知语言返回空集合"""
        stopwords = self.manager.get_stopwords("fr")
        self.assertEqual(stopwords, set())


class TestTokenizeChinese(unittest.TestCase):
    """测试中文分词 (jieba)"""

    def test_tokenize_chinese_basic(self):
        """测试中文分词基本功能"""
        result = tokenize_chinese("这个手机很好用")
        # 无论使用 jieba 还是降级方案，都应该返回中文字符
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_empty_text(self):
        """测试空文本"""
        self.assertEqual(tokenize_chinese(""), [])


class TestTokenizeEnglish(unittest.TestCase):
    """测试英文分词 (nltk)"""

    def test_tokenize_english_basic(self):
        """测试英文分词基本功能"""
        result = tokenize_english("This is a good product!")
        # 应该返回小写的单词列表
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # 检查是否包含实词
        self.assertIn("good", result)
        self.assertIn("product", result)

    def test_tokenize_english_stemming(self):
        """测试英文分词带词干提取"""
        result = tokenize_english("running products", apply_stemming=True)
        self.assertIsInstance(result, list)


class TestTokenizeRussian(unittest.TestCase):
    """测试俄文分词 (natasha)"""

    def test_tokenize_russian_raises_without_natasha(self):
        """测试没有 natasha 时抛出错误（如果未安装）"""
        from data import preprocessor as pp
        if not pp.NATASHA_AVAILABLE:
            with self.assertRaises(ImportError) as context:
                tokenize_russian("хороший товар")
            self.assertIn("natasha is required", str(context.exception))
        else:
            # 如果已安装，测试基本功能
            result = tokenize_russian("хороший товар", apply_lemmatization=True)
            self.assertIsInstance(result, list)


class TestCleanForNLP(unittest.TestCase):
    """测试 NLP 深度清洗"""

    def test_chinese_cleaning(self):
        """测试中文清洗"""
        text = "这个手机真的很好用！推荐购买。"
        result = clean_for_nlp(text, "zh", remove_stopwords=True)

        # 结果应该是清洗后的词序列
        self.assertIsInstance(result, str)
        # 停用词应该被去除，实词保留
        self.assertNotIn("！", result)  # 标点被去除

    def test_english_cleaning(self):
        """测试英文清洗"""
        text = "This is a great product!"
        result = clean_for_nlp(text, "en", remove_stopwords=True)

        # 结果应该是清洗后的词序列
        self.assertIsInstance(result, str)
        # 停用词应该被去除，实词应该保留
        self.assertNotIn("!", result)
        self.assertNotIn("is", result)  # 停用词被去除

    def test_empty_text(self):
        """测试空文本"""
        self.assertEqual(clean_for_nlp("", "zh"), "")


class TestProcessRecord(unittest.TestCase):
    """测试记录处理"""

    def setUp(self):
        self.raw_record = RawRecord(
            id="test-uuid",
            language="en",
            source="amazon_hf",
            original_text="This is a great product! Check https://example.com",
            sentiment_label=2,
            rating=5
        )

    def test_process_record(self):
        """测试处理单条记录"""
        result = process_record(self.raw_record)

        self.assertIsInstance(result, ProcessedRecord)
        self.assertEqual(result.id, "test-uuid")
        # LLM 流保留完整语法
        self.assertNotIn("https://example.com", result.text_for_llm)  # URL 被去除
        # NLP 流深度清洗
        self.assertIsInstance(result.text_for_nlp, str)
        # 验证统计字段
        self.assertGreaterEqual(result.word_count, 0)
        self.assertGreaterEqual(result.char_count, 0)

    def test_word_count(self):
        """测试词数统计"""
        result = process_record(self.raw_record)
        self.assertGreaterEqual(result.word_count, 0)
        self.assertGreaterEqual(result.char_count, 0)

    def test_length_classification(self):
        """测试长度分类"""
        result = process_record(self.raw_record)
        self.assertIn(result.length_category, ["short", "medium", "long"])


class TestProcessBatch(unittest.TestCase):
    """测试批量处理"""

    def setUp(self):
        self.records = [
            RawRecord(
                language="zh",
                source="amazon_hf",
                original_text=f"测试文本{i}",
                sentiment_label=i % 3
            )
            for i in range(5)
        ]

    @patch('data.preprocessor.tokenize_chinese')
    def test_process_batch(self, mock_tokenize):
        """测试批量处理"""
        mock_tokenize.return_value = ["测试", "文本"]

        results = process_batch(self.records, show_progress=False)

        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertIsInstance(result, ProcessedRecord)
            self.assertEqual(result.language, "zh")

    @patch('data.preprocessor.process_record')
    def test_skip_error_records(self, mock_process):
        """测试跳过错误记录"""
        # 第二个记录处理失败
        mock_process.side_effect = [
            ProcessedRecord(**{**self.records[0].model_dump(), "text_for_nlp": "test", "text_for_llm": "test", "word_count": 1, "char_count": 4, "length_category": "short"}),
            Exception("Processing error"),
            ProcessedRecord(**{**self.records[2].model_dump(), "text_for_nlp": "test", "text_for_llm": "test", "word_count": 1, "char_count": 4, "length_category": "short"}),
        ]

        results = process_batch(self.records[:3], show_progress=False)

        self.assertEqual(len(results), 2)  # 跳过1条错误记录


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_pipeline(self):
        """测试完整流程：Raw -> Processed"""
        raw = RawRecord(
            language="en",
            source="amazon_hf",
            original_text="Amazing product! Love it. https://example.com",
            sentiment_label=2,
            rating=5
        )

        processed = process_record(raw)

        # 验证所有字段
        self.assertEqual(processed.id, raw.id)
        self.assertEqual(processed.language, raw.language)
        self.assertEqual(processed.original_text, raw.original_text)
        self.assertEqual(processed.sentiment_label, raw.sentiment_label)

        # 验证新增字段
        self.assertIsNotNone(processed.text_for_llm)
        self.assertIsNotNone(processed.text_for_nlp)
        self.assertIsInstance(processed.word_count, int)
        self.assertIn(processed.length_category, ["short", "medium", "long"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
