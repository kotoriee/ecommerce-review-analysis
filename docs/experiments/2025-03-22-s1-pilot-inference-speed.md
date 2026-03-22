# 实验日记：S1 Pilot 推理速度分析

## 日期
2025-03-22

## 实验背景
- **模型**: Qwen3-4B + LoRA (600条数据微调)
- **硬件**: RTX 3070 Ti Laptop (8GB VRAM)
- **框架**: Unsloth FastLanguageModel
- **任务**: 三分类情感分析 (negative/neutral/positive)

## 推理速度实测

### Unsloth 单条推理
```
速度: ~10秒/条
200条预计: 33分钟
8616条全量预计: 24小时
```

**结论**: 单条推理太慢，无法满足批量评估需求。

---

## 推理方案对比

| 方案 | 速度 | 显存占用 | 适用场景 | 复杂度 |
|------|------|---------|---------|--------|
| **Unsloth 单条** | ~0.1条/秒 (10s/it) | ~4.5GB | 开发调试 | 低 |
| **Unsloth 批量** | ~1-5条/秒 | ~6GB | 小批量测试 | 中 |
| **vLLM 批量** | 50-100条/秒 | ~6GB | 大量数据批处理 | 中 |
| **Ollama/GGUF** | 10-30条/秒 | ~4GB | 本地服务/API | 低 |
| **合并模型+vLLM** | 50+条/秒 | ~8GB | 生产部署 | 高 |

---

## 当前瓶颈分析

1. **单条 generate() 调用**: 每次都要重新准备 input tensors
2. **KV Cache 未预热**: 每条独立计算，无法复用
3. **Python 循环开销**: tqdm + 单线程处理
4. **模型加载方式**: base + LoRA 动态合并有 overhead

---

## 优化路径

### 短期（当前阶段）
- [ ] 使用小样本 (n=200) 快速验证模型质量
- [ ] 接受较慢速度，重点观察准确率是否提升

### 中期（S2-S5阶段）
- [ ] 实现 vLLM 批量推理脚本
- [ ] 导出合并模型 (16-bit) 供 vLLM 使用
- [ ] 预期加速: 50-100x

### 长期（生产部署）
- [ ] GGUF 量化 + Ollama 本地服务
- [ ] 或 vLLM 部署为 API 服务
- [ ] 支持并发请求处理

---

## 关键决策

**当前决策**: 继续用 Unsloth 单条完成 S1 Pilot 评估 (200条)
- 原因: 快速验证 600条训练效果，确认流程正确
- 时间成本: ~30分钟可接受

**下一阶段**: 必须切换到 vLLM 批量推理
- 原因: S2-S5 数据量 1200-8500条，单条不可行
- 预计实现时间: 2-3小时

---

## 相关文件

- `code/local_llm/evaluate_unsloth.py` - 当前慢速评估
- `code/local_llm/evaluate_vllm.py` - 待完善的快速评估
- `docs/experiments/` - 实验记录目录

---

## 参考链接

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Unsloth 推理优化](https://docs.unsloth.ai/basics/inference)
- [Ollama 本地部署](https://ollama.com/)
