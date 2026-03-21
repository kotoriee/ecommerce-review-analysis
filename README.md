# 论文项目管理中心

**题目:** Comparative Analysis of Traditional NLP vs. LLMs for E-Commerce Review Analysis
**更新:** 2026-03-21

---

## 当前状态总览

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 数据层 (schema / loader / preprocessor / merger) | ✅ 完成 (86 tests passing) |
| Phase 2 | 传统NLP基线 (SVM ✅, LDA ✅, GSDMM ✅) | ✅ 完成 |
| Phase 3 | 本地LLM (CoT数据标注 ✅, Qwen3-4B微调 ✅) | ✅ 完成 |
| Phase 4 | 云端Agent (Claude API批量推理) | 🔄 部分完成 |
| Phase 5 | 评估框架 (三路对比实验) | ❌ 未开始 |
| Phase 6 | 论文写作 | ❌ 未开始 |

**当前瓶颈:** Phase 3 Qwen3-8B 在 C2Net (Ascend 910) 上微调

---

## 关键文件索引

### 本文件夹
| 文件 | 内容 |
|------|------|
| [thesis_proposal.md](thesis_proposal.md) | 论文提案（研究目标/方法/假设） |
| [project_status.md](project_status.md) | 详细阶段进展报告 |
| [defense_action_plan.md](defense_action_plan.md) | 答辩准备路线图 |
| [defense_peer_review.md](defense_peer_review.md) | 同行评审反馈（找出论文弱点） |
| [literature/references.bib](literature/references.bib) | BibTeX 参考文献库 |
| [literature/reading_guide.md](literature/reading_guide.md) | 文献精读指南 |
| [literature/reading_prompt.md](literature/reading_prompt.md) | 文献精读提示词（直接用于AI） |
| [literature/literature_review.md](literature/literature_review.md) | 已完成的文献综述 |

### 代码库 (code/ecommerce_analysis/)
| 路径 | 内容 |
|------|------|
| [data/schema.py](../code/ecommerce_analysis/data/schema.py) | 全局数据契约（核心，勿随意改） |
| [data/loader.py](../code/ecommerce_analysis/data/loader.py) | HuggingFace 数据加载 |
| [baseline/sentiment/svm_classifier.py](../code/ecommerce_analysis/baseline/sentiment/svm_classifier.py) | SVM+TF-IDF 分类器 |
| [baseline/topic/gsdmm_model.py](../code/ecommerce_analysis/baseline/topic/gsdmm_model.py) | GSDMM短文本主题模型（学术创新） |
| [cloud_agent/scripts/batch_sentiment.py](../code/ecommerce_analysis/cloud_agent/scripts/batch_sentiment.py) | Claude API 批量推理 |
| [local_llm/inference/predictor.py](../code/ecommerce_analysis/local_llm/inference/predictor.py) | Ollama 本地推理 |

### 本地LLM训练 (code/local_llm/)
| 路径 | 内容 |
|------|------|
| [code/local_llm/train_sentiment.py](code/local_llm/train_sentiment.py) | Qwen3-4B QLoRA 微调脚本 |
| [code/local_llm/evaluate_unsloth.py](code/local_llm/evaluate_unsloth.py) | 模型评估脚本 |
| [models/qwen3-4b-sentiment-lora/](models/qwen3-4b-sentiment-lora/) | 训练好的 LoRA 适配器 |

### NPU训练 (code/npu_training/)
| 路径 | 内容 |
|------|------|
| [code/npu_training/scripts/train_qwen3_8b_npu.py](code/npu_training/scripts/train_qwen3_8b_npu.py) | Qwen3-8B C2Net微调（备用）|
| [code/npu_training/lib.md/训练.md](code/npu_training/lib.md/训练.md) | C2Net训练指南 |

---

## 三路对比架构

```
亚马逊评论数据 (英文电商评论)
        │
        ├──► [SVM + TF-IDF]     → 预测结果 A (baseline)
        │
        ├──► [Qwen3-4B QLoRA]   → 预测结果 B (本地LLM)
        │      └─ 在RTX 3060 Ti (8GB) 本地微调
        │      └─ 40% 准确率, 5.4条/秒推理速度
        │
        └──► [Claude Agent]     → 预测结果 C (云端)
                     │
                     └──► evaluation/compare.py → 论文结果
```

---

## 本地LLM训练结果

| 指标 | 数值 |
|------|------|
| **模型** | Qwen3-4B (unsloth-bnb-4bit) |
| **训练数据** | 6,033 条英文电商评论 |
| **训练配置** | 3 epochs, 1131 steps, LoRA r=16 |
| **训练时间** | 2小时40分 |
| **显存峰值** | 4.4 GB / 8 GB |
| **评估样本** | 1000 条测试集 |
| **准确率** | **40%** |
| **推理速度** | **5.4 条/秒** (325 条/分钟) |

**关键优化：**
- 修复 OOM：将 LoRA rank 从 32 降至 16
- 序列长度从 512 降至 256
- 改用 JSON 直接输出格式（移除 CoT），速度提升 13 倍

---

## 核心假设（来自提案）

1. **本地量化LLM** 微调后可达业务可用性能，推理成本降低10倍，无隐私风险
2. **云端Agent** 在讽刺/反语等复杂场景具有不可替代优势，但成本高
3. **传统NLP** 仅在超低延迟(<50ms)场景保持优势

---

## 下一步行动（优先级排序）

- [x] **[高]** 完成本地 Qwen3-4B 微调，获取 Phase 3 结果 ✅
- [ ] **[高]** 优化模型准确率（目标 60-70%，当前 40%）
- [ ] **[高]** 用 Claude API 跑完测试集，获取 Phase 4 结果
- [ ] **[中]** 实现 evaluation/compare.py 三路对比
- [ ] **[中]** 生成对比图表（F1/延迟/成本）
- [ ] **[低]** 撰写论文 Chapter 2（系统设计）和 Chapter 3（实验）

---

## 语言处理约束（CRITICAL）

| 语言 | 必须使用 | 禁止 |
|------|---------|------|
| 中文 | `jieba` | - |
| 英文 | `nltk` | - |
| 俄文 | `natasha` | **绝对禁止用 nltk** |
