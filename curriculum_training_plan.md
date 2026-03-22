# Qwen3-4B 阶梯式微调计划 (Curriculum Training)

## 背景

- **总数据量**: ~8,500 条软标注数据 (R1蒸馏生成)
- **起始点**: 600 条
- **目标**: 验证数据效率，找到准确率提升的关键节点
- **硬件**: RTX 3070 Ti Laptop (8GB VRAM)

---

## 阶梯设计

| Stage | 数据量 | 增量 | 数据文件 | 输出模型 | 预期训练时间 |
|-------|--------|------|----------|----------|--------------|
| S1 | 600 | baseline | train_600.json | lora_s1_600 | ~15 min |
| S2 | 1,200 | +600 | train_1200.json | lora_s2_1200 | ~25 min |
| S3 | 2,400 | +1,200 | train_2400.json | lora_s3_2400 | ~45 min |
| S4 | 4,800 | +2,400 | train_4800.json | lora_s4_4800 | ~90 min |
| S5 | 8,500 | +3,700 | train_full.json | lora_s5_full | ~150 min |

---

## 数据分割策略

### 方法: 分层随机采样

```python
# data_split.py
import json
import random
from pathlib import Path

# 固定随机种子确保可复现
random.seed(42)

# 加载完整数据
with open("data/processed/soft_labels_raw.jsonl") as f:
    all_data = [json.loads(line) for line in f]

# 按情感标签分层
by_label = {0: [], 1: [], 2: []}
for item in all_data:
    label = item.get("soft_label", 1)  # 默认中性
    by_label[label].append(item)

print(f"数据分布: 负={len(by_label[0])}, 中={len(by_label[1])}, 正={len(by_label[2])}")

# 各阶段采样
def sample_balanced(n_total):
    """均衡采样，确保三类比例大致相等"""
    n_per_class = n_total // 3
    samples = []
    for label in [0, 1, 2]:
        pool = by_label[label]
        if len(pool) >= n_per_class:
            samples.extend(random.sample(pool, n_per_class))
        else:
            samples.extend(pool)  # 如果某类不足，全取
    # 补足剩余
    while len(samples) < n_total:
        samples.append(random.choice(all_data))
    random.shuffle(samples)
    return samples

# 生成各阶段数据
stages = [600, 1200, 2400, 4800, 8500]
for n in stages:
    samples = sample_balanced(n)
    output = f"data/curriculum/train_{n}.json"
    with open(output, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"生成 {output}: {len(samples)} 条")
```

### 验证集固定

```bash
# 使用统一的 1000 条验证集
cp data/processed/val_3cls.json data/curriculum/val_fixed.json
# 或使用完整的 val + test 作为验证 (~2000条)
cat data/processed/val_3cls.json data/processed/test_3cls.json > data/curriculum/val_combined.json
```

---

## 训练配置

### 阶段 S1 (600条) - 探索阶段

```bash
python code/local_llm/train_sentiment.py \
    --data data/curriculum/train_600.json \
    --output models/curriculum/lora_s1_600 \
    --epochs 5 \
    --lr 3e-4 \
    --batch 2 \
    --grad-acc 8
```

**策略**: 高学习率 + 多epoch，快速拟合小数据

### 阶段 S2-S5 (增量训练)

```bash
# S2: 1200条，从S1继续
python code/local_llm/train_sentiment.py \
    --data data/curriculum/train_1200.json \
    --output models/curriculum/lora_s2_1200 \
    --epochs 3 \
    --lr 2e-4 \
    --batch 2 \
    --grad-acc 8

# S3: 2400条
python code/local_llm/train_sentiment.py \
    --data data/curriculum/train_2400.json \
    --output models/curriculum/lora_s3_2400 \
    --epochs 3 \
    --lr 2e-4

# S4: 4800条
python code/local_llm/train_sentiment.py \
    --data data/curriculum/train_4800.json \
    --output models/curriculum/lora_s4_4800 \
    --epochs 3 \
    --lr 1e-4

# S5: 全量8500条
python code/local_llm/train_sentiment.py \
    --data data/curriculum/train_full.json \
    --output models/curriculum/lora_s5_full \
    --epochs 3 \
    --lr 1e-4
```

---

## 统一评估脚本

```python
# code/local_llm/eval_curriculum.py
import json
import subprocess
import sys
from pathlib import Path

STAGES = [600, 1200, 2400, 4800, 8500]
RESULTS = []

for n in STAGES:
    model_path = f"models/curriculum/lora_s{n}_{n if n != 8500 else 'full'}"

    print(f"\n{'='*50}")
    print(f"评估 Stage: {n} 条数据")
    print(f"{'='*50}")

    # 运行评估
    result = subprocess.run([
        sys.executable, "code/local_llm/evaluate_unsloth.py",
        "--model", model_path,
        "--data", "data/curriculum/val_fixed.json",
        "--output", f"results/curriculum/eval_{n}.json"
    ], capture_output=True, text=True)

    # 解析结果
    accuracy = parse_accuracy(result.stdout)
    speed = parse_speed(result.stdout)

    RESULTS.append({
        "stage": n,
        "accuracy": accuracy,
        "speed": speed,
        "model_path": model_path
    })

# 保存对比结果
with open("results/curriculum/summary.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

# 打印表格
print("\n" + "="*60)
print("阶梯式训练结果汇总")
print("="*60)
print(f"{'数据量':>8} | {'准确率':>8} | {'推理速度':>10} | {'提升':>8}")
print("-"*60)
base_acc = RESULTS[0]["accuracy"]
for r in RESULTS:
    delta = r["accuracy"] - base_acc
    print(f"{r['stage']:>8} | {r['accuracy']:>7.1%} | {r['speed']:>9.1f} | +{delta:>6.1%}")
```

---

## 预期结果分析

### 关键观察点

1. **数据效率转折点**: 在哪个数据量之后准确率提升开始放缓？
2. **边际收益**: 每增加一倍数据，准确率提升多少？
3. **过拟合风险**: 小数据量 (600) 是否严重过拟合？

### 预期曲线

```
准确率
  │
70%│                      ★目标
  │                    ／
60%│              ★---／
  │         ★----／
50%│    ★----／
  │★--／
40%│ 基线
  └───────────────────────────
    600  1200  2400  4800  8500
              数据量
```

---

## 快速启动命令

```bash
# 1. 准备数据
mkdir -p data/curriculum results/curriculum models/curriculum
python code/local_llm/data_split_curriculum.py

# 2. 运行完整训练流水线
bash code/local_llm/run_curriculum.sh

# 3. 生成对比报告
python code/local_llm/eval_curriculum.py
python code/local_llm/plot_curriculum.py
```

---

## 论文价值

该阶梯式实验可直接用于论文的 **Data Efficiency Analysis** 章节：

> "We conduct a curriculum-style training study to understand the data efficiency of Qwen3-4B on e-commerce sentiment analysis. Starting from 600 samples, we progressively double the training data until reaching the full 8,500 samples..."

图表建议:
1. **准确率-数据量曲线** (Figure X)
2. **边际收益递减分析** (Figure Y)
3. **与SVM基线的对比** (Figure Z)
