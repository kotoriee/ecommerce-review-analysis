"""
GSDMM 模型单元测试

测试 Gibbs Sampling Dirichlet Multinomial Mixture Model 的核心功能。
"""

import unittest
import numpy as np
from baseline.topic.gsdmm_model import GSDMMModel, GSDMMConfig, compute_coherence_score


class TestGSDMMConfig(unittest.TestCase):
    """测试 GSDMM 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = GSDMMConfig()
        self.assertEqual(config.K, 15)
        self.assertEqual(config.alpha, 0.1)
        self.assertEqual(config.beta, 0.1)
        self.assertEqual(config.n_iter, 50)
        self.assertEqual(config.random_state, 42)

    def test_custom_config(self):
        """测试自定义配置"""
        config = GSDMMConfig(K=20, alpha=0.5, beta=0.5, n_iter=100)
        self.assertEqual(config.K, 20)
        self.assertEqual(config.alpha, 0.5)
        self.assertEqual(config.beta, 0.5)
        self.assertEqual(config.n_iter, 100)


class TestGSDMMModel(unittest.TestCase):
    """测试 GSDMM 模型核心功能"""

    def setUp(self):
        """设置测试数据"""
        # 创建简单的测试数据
        self.sample_docs = [
            ["手机", "电池", "续航", "差"],
            ["手机", "屏幕", "显示", "好"],
            ["物流", "快递", "速度", "快"],
            ["物流", "配送", "慢", "差"],
            ["屏幕", "分辨率", "清晰"],
            ["电池", "充电", "慢"],
        ]

    def test_model_initialization(self):
        """测试模型初始化"""
        model = GSDMMModel()

        self.assertIsNotNone(model.config)
        self.assertEqual(model.config.K, 15)
        self.assertFalse(model.is_fitted)

    def test_fit(self):
        """测试模型训练"""
        config = GSDMMConfig(K=10, n_iter=10, random_state=42)
        model = GSDMMModel(config)

        # 训练模型
        model.fit(self.sample_docs, verbose=False)

        # 验证训练完成
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.vocab)
        self.assertIsNotNone(model.cluster_doc_count)
        self.assertIsNotNone(model.doc_cluster_assignments)

        # 验证词汇表
        self.assertGreater(len(model.vocab), 0)

        # 验证聚类分配
        self.assertEqual(len(model.doc_cluster_assignments), len(self.sample_docs))

    def test_predict(self):
        """测试预测功能"""
        config = GSDMMConfig(K=10, n_iter=10, random_state=42)
        model = GSDMMModel(config)
        model.fit(self.sample_docs, verbose=False)

        # 预测新文档
        new_docs = [
            ["手机", "电池", "好"],
            ["物流", "快"],
        ]

        predictions = model.predict(new_docs)

        # 验证预测结果
        self.assertEqual(len(predictions), len(new_docs))
        self.assertTrue(all(0 <= p < model.K for p in predictions if p >= 0))

    def test_get_topic_words(self):
        """测试获取主题词"""
        config = GSDMMConfig(K=10, n_iter=10, random_state=42)
        model = GSDMMModel(config)
        model.fit(self.sample_docs, verbose=False)

        # 获取活跃聚类
        active_clusters = model.get_active_clusters()
        self.assertGreater(len(active_clusters), 0)

        # 获取主题词
        for cluster_idx in active_clusters[:3]:  # 测试前3个聚类
            topic_words = model.get_topic_words(cluster_idx, top_n=5)

            self.assertIsInstance(topic_words, list)
            self.assertGreater(len(topic_words), 0)
            self.assertLessEqual(len(topic_words), 5)

            # 验证格式
            for word, prob in topic_words:
                self.assertIsInstance(word, str)
                self.assertIsInstance(prob, float)
                self.assertGreater(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_get_cluster_distribution(self):
        """测试聚类分布"""
        config = GSDMMConfig(K=10, n_iter=10, random_state=42)
        model = GSDMMModel(config)
        model.fit(self.sample_docs, verbose=False)

        distribution = model.get_cluster_distribution()

        # 验证分布
        self.assertIsInstance(distribution, dict)
        self.assertGreater(len(distribution), 0)

        # 验证文档数总和
        total_docs = sum(distribution.values())
        self.assertEqual(total_docs, len(self.sample_docs))

    def test_active_clusters(self):
        """测试活跃聚类数量"""
        config = GSDMMConfig(K=20, n_iter=20, random_state=42)
        model = GSDMMModel(config)
        model.fit(self.sample_docs, verbose=False)

        active_clusters = model.get_active_clusters()

        # 活跃聚类数应该小于等于初始K
        self.assertLessEqual(len(active_clusters), config.K)
        # 活跃聚类数应该大于0
        self.assertGreater(len(active_clusters), 0)

    def test_empty_document_handling(self):
        """测试空文档处理"""
        docs_with_empty = [
            ["手机", "好"],
            [],  # 空文档
            ["物流", "快"],
        ]

        config = GSDMMConfig(K=5, n_iter=5, random_state=42)
        model = GSDMMModel(config)

        # 应该能够处理空文档
        model.fit(docs_with_empty, verbose=False)

        self.assertTrue(model.is_fitted)

    def test_reproducibility(self):
        """测试结果可重现性"""
        config = GSDMMConfig(K=10, n_iter=10, random_state=42)

        # 训练两次
        model1 = GSDMMModel(config)
        model1.fit(self.sample_docs, verbose=False)

        model2 = GSDMMModel(config)
        model2.fit(self.sample_docs, verbose=False)

        # 验证聚类分配一致
        np.testing.assert_array_equal(
            model1.doc_cluster_assignments,
            model2.doc_cluster_assignments
        )


class TestGSDMMEdgeCases(unittest.TestCase):
    """测试边缘情况"""

    def test_single_document(self):
        """测试单个文档"""
        docs = [["手机", "好"]]

        config = GSDMMConfig(K=5, n_iter=5, random_state=42)
        model = GSDMMModel(config)
        model.fit(docs, verbose=False)

        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.get_active_clusters()), 1)

    def test_identical_documents(self):
        """测试完全相同的文档"""
        docs = [
            ["手机", "好"],
            ["手机", "好"],
            ["手机", "好"],
        ]

        config = GSDMMConfig(K=5, n_iter=10, random_state=42)
        model = GSDMMModel(config)
        model.fit(docs, verbose=False)

        # 所有文档应该分配到同一个聚类
        distribution = model.get_cluster_distribution()
        self.assertEqual(len(distribution), 1)
        self.assertEqual(list(distribution.values())[0], 3)


class TestCoherenceScore(unittest.TestCase):
    """测试主题一致性评估"""

    def test_coherence_computation(self):
        """测试一致性得分计算"""
        docs = [
            ["手机", "电池", "续航"],
            ["手机", "屏幕", "显示"],
            ["物流", "快递", "速度"],
        ]

        config = GSDMMConfig(K=5, n_iter=10, random_state=42)
        model = GSDMMModel(config)
        model.fit(docs, verbose=False)

        # 计算一致性得分
        try:
            score = compute_coherence_score(model, docs, top_n=5)

            # 验证得分范围
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

        except ImportError:
            # 如果没有 Gensim，跳过测试
            self.skipTest("Gensim not available")


if __name__ == '__main__':
    unittest.main()