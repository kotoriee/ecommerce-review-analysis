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
| Phase 4 | 云端Agent (API批量推理) | ✅ 完成 ([api_eval_sentiment.py](code/cloud_agent/api_eval_sentiment.py)) |
| Phase 5 | 评估框架 (三路对比实验) | ✅ 完成 ([metrics.py](code/evaluation/metrics.py), [run_comparison.py](code/evaluation/run_comparison.py)) |
| Phase 6 | 论文写作 | ✅ 完成 (6章, ~3万字, [d:/0321/thesis/](../thesis/)) |

---

## 关键文件索引

### Phase 4 & 5 (Cloud Agent + Evaluation) - 已就绪
| 路径 | 内容 | 状态 |
|------|------|------|
| [code/cloud_agent/api_eval_sentiment.py](code/cloud_agent/api_eval_sentiment.py) | API情感分析 (DeepSeek/Qwen/Claude) | ✅ 就绪 |
| [code/cloud_agent/run_3cls_annotation.py](code/cloud_agent/run_3cls_annotation.py) | R1蒸馏标注（软标签生成） | ✅ 已运行 (4564条) |
| [code/cloud_agent/batch_sentiment.py](code/cloud_agent/batch_sentiment.py) | Ollama本地批量推理 | ✅ 就绪 |
| [code/evaluation/run_comparison.py](code/evaluation/run_comparison.py) | 三路对比主程序 | ✅ 就绪 |
| [code/evaluation/metrics.py](code/evaluation/metrics.py) | 评估指标 (F1/Acc/Confusion) | ✅ 就绪 |
| [code/evaluation/run_svm_baseline.py](code/evaluation/run_svm_baseline.py) | SVM基线训练评估 | ✅ 就绪 |
| [code/evaluation/visualize.py](code/evaluation/visualize.py) | 对比图表生成 | ✅ 就绪 |
| [code/evaluation/generate_report.py](code/evaluation/generate_report.py) | Markdown/LaTeX报告 | ✅ 就绪 |

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

### 核心代码库 (code/)
| 路径 | 内容 |
|------|------|
| [code/data/schema.py](code/data/schema.py) | 全局数据契约（核心，勿随意改） |
| [code/data/loader.py](code/data/loader.py) | HuggingFace 数据加载 |
| [code/baseline/svm_classifier.py](code/baseline/svm_classifier.py) | SVM+TF-IDF 分类器 |
| [code/baseline/gsdmm_model.py](code/baseline/gsdmm_model.py) | GSDMM短文本主题模型（学术创新） |

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
        │      └─ 在RTX 3070 Ti Laptop (8GB) 本地微调
        │      └─ 早期训练 40% 准确率, 5.4条/秒推理速度（完整对比实验待执行）
        │
        └──► [Claude Agent]     → 预测结果 C (云端)
                     │
                     └──► evaluation/compare.py → 论文结果
```

---

## 本地LLM训练结果（早期训练）

> **注：** 以下为早期训练结果。完整三路对比实验（SVM vs QLoRA vs Claude Agent）代码已就绪，实验待执行。

| 指标 | 数值 |
|------|------|
| **模型** | Qwen3-4B (unsloth-bnb-4bit) |
| **硬件** | RTX 3070 Ti Laptop (8GB VRAM) |
| **训练数据** | 6,033 条英文电商评论 |
| **训练配置** | 3 epochs, 1131 steps, LoRA r=16 |
| **训练时间** | 2小时40分 |
| **显存峰值** | 4.4 GB / 8 GB |
| **准确率（早期）** | **40%** |
| **推理速度** | **5.4 条/秒** (325 条/分钟) |

**SVM 基线结果（已测试）：** 65.3% 准确率（TF-IDF Unigram+Bigram, max_features=10K）

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

## 下一步行动（执行阶段）

**Phase 4/5 代码已就绪，待执行：**

1. **SVM基线** - `python code/evaluation/run_svm_baseline.py`
   - 输入: data/processed/train_3cls.json + test_3cls.json
   - 输出: data/results/svm_predictions.jsonl

2. **API推理** - `python code/cloud_agent/api_eval_sentiment.py`
   - 输入: data/processed/test_3cls.json
   - 输出: data/results/api_predictions.jsonl

3. **本地LLM推理** - 跑 evaluate_unsloth.py 或 predictor.py
   - 输出: data/results/local_llm_predictions.jsonl

4. **三路对比** - `python code/evaluation/run_comparison.py`
   - 输入: 上述三个预测文件
   - 输出: data/results/comparison_results.json

5. **生成报告** - `python code/evaluation/generate_report.py`
   - 输出: data/results/summary.md + summary_latex.tex

6. **生成图表** - `python code/evaluation/visualize.py`
   - 输出: data/results/charts/ (F1对比/混淆矩阵/延迟成本)

---

## 论文写作（已完成）

**位置：** `d:/0321/thesis/`
**语言：** 中文，约 30,000 字
**题目：** 电商评论情感分析——传统NLP与大语言模型多路径对比研究

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| 摘要 | [abstract.md](../thesis/abstract.md) | 中英双语摘要 |
| 第1章 | [chapter1_intro.md](../thesis/chapters/chapter1_intro.md) | 研究背景、意义、目标 |
| 第2章 | [chapter2_literature.md](../thesis/chapters/chapter2_literature.md) | 情感分析演进、PEFT、知识蒸馏（最长章节）|
| 第3章 | [chapter3_dataset.md](../thesis/chapters/chapter3_dataset.md) | MARC数据集、DeepSeek-R1软标签流水线（~8,527条，91%一致率）|
| 第4章 | [chapter4_method.md](../thesis/chapters/chapter4_method.md) | 四路架构详细设计（SVM/GSDMM/QLoRA/Claude Agent）|
| 第5章 | [chapter5_experiment.md](../thesis/chapters/chapter5_experiment.md) | 对比实验设计、SVM基线结果（65.3%）、消融实验 |
| 第6章 | [chapter6_conclusion.md](../thesis/chapters/chapter6_conclusion.md) | 四大贡献、四点局限性、五个未来方向 |

**参考文献：** [references.bib](../thesis/references/references.bib) — 33条BibTeX，已全部核查修正

---

## 待办清单

- [x] **[高]** Phase 3: 本地 Qwen3-4B 微调完成 ✅
- [x] **[高]** Phase 6: 论文写作完成 ✅ (~3万字)
- [ ] **[高]** Phase 4: 执行 Claude API 批量推理（695条测试集）
- [ ] **[高]** Phase 5: 执行三路对比 + 生成报告
- [ ] **[中]** 用实际对比实验结果更新论文第5章数据

---

## 语言处理约束（CRITICAL）

| 语言 | 必须使用 | 禁止 |
|------|---------|------|
| 中文 | `jieba` | - |
| 英文 | `nltk` | - |
| 俄文 | `natasha` | **绝对禁止用 nltk** |
