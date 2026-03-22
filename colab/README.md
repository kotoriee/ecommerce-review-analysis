# Google Colab 分流推理指南

在 Colab T4 GPU 上批量评估 Curriculum 模型，分流本地计算压力。

## 速度对比

| 环境 | 方法 | 速度 | 500条耗时 |
|------|------|------|-----------|
| 本地 RTX 3070Ti | Unsloth 单条 | 0.11 条/秒 | ~75 分钟 |
| 本地 RTX 3070Ti | vLLM 批量 | ~10 条/秒 | ~1 分钟 |
| **Colab T4** | **vLLM 批量** | **~15 条/秒** | **~30 秒** |

## 快速开始

### 方法 1: 直接运行 (推荐)

在 Colab notebook 中执行:

```python
# 1. 克隆仓库
!git clone https://github.com/kotoriee/ecommerce-review-analysis.git
%cd ecommerce-review-analysis

# 2. 安装依赖
!pip install vllm transformers -q

# 3. 下载模型 (从 HuggingFace 或 Google Drive)
# 选项 A: 从 HuggingFace (如果已上传)
# !huggingface-cli download your-username/qwen3-4b-curriculum --local-dir ./models

# 选项 B: 从 Google Drive 挂载
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/models/curriculum ./models/

# 4. 运行评估
!python colab/curriculum_eval_colab.py --stage s5 --samples 500
```

### 方法 2: 分阶段评估所有模型

```python
# 评估所有 5 个阶段
stages = ["s1", "s2", "s3", "s4", "s5"]

for stage in stages:
    print(f"\n{'='*60}")
    print(f"评估 {stage.upper()}")
    print(f"{'='*60}")
    !python colab/curriculum_eval_colab.py --stage {stage} --samples 500
```

### 方法 3: 只评估特定阶段

```python
# 只评估 S5 (全量数据训练的最终模型)
!python colab/curriculum_eval_colab.py --stage s5 --samples 1000

# 评估 S2 并保存到特定文件
!python colab/curriculum_eval_colab.py --stage s2 --samples 500 --output results/s2_eval.json
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stage` | 评估阶段: s1/s2/s3/s4/s5 | 必填 |
| `--samples` | 评估样本数 | 500 |
| `--batch-size` | vLLM 批大小 | 64 |
| `--model-dir` | 模型目录 | ./models/curriculum |
| `--data-dir` | 数据目录 | ./data/curriculum |
| `--output` | 输出文件 | eval_{stage}_colab.json |

## 上传模型到 Colab

### 方式 1: Google Drive (推荐)

1. 将本地 `models/curriculum/` 压缩上传
2. 在 Colab 挂载 Drive
3. 解压到项目目录

```python
from google.colab import drive
drive.mount('/content/drive')

# 解压模型
!unzip /content/drive/MyDrive/curriculum_models.zip -d ./models/
```

### 方式 2: HuggingFace Hub

1. 安装 huggingface-cli
2. 登录并上传模型
3. 在 Colab 下载

```bash
!pip install huggingface_hub -q
!huggingface-cli login
!huggingface-cli download your-username/qwen3-curriculum --local-dir ./models/curriculum
```

### 方式 3: 直接 scp/上传

如果模型较小 (LoRA ~100MB)，可以直接拖拽上传到 Colab:
```python
# 创建目录并上传
!mkdir -p models/curriculum
# 然后使用 Colab 左侧文件面板拖拽上传
```

## 批量评估脚本

评估所有阶段并生成对比图表:

```python
import json
import matplotlib.pyplot as plt

results = {}
for stage in ["s1", "s2", "s3", "s4", "s5"]:
    with open(f"eval_{stage}_colab.json") as f:
        data = json.load(f)
        results[stage] = data["accuracy"]

# 绘制学习曲线
stages = ["S1(600)", "S2(1200)", "S3(2400)", "S4(4800)", "S5(8500)"]
accuracies = [results[f"s{i+1}"] for i in range(5)]

plt.figure(figsize=(10, 6))
plt.plot(stages, accuracies, marker='o', linewidth=2)
plt.axhline(y=40, color='r', linestyle='--', label='Baseline (40%)')
plt.ylabel('Accuracy (%)')
plt.title('Qwen3-4B Curriculum Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('curriculum_curve.png', dpi=150)
plt.show()
```

## 结果下载

评估完成后下载结果文件:

```python
from google.colab import files

# 下载所有评估结果
for stage in ["s1", "s2", "s3", "s4", "s5"]:
    files.download(f"eval_{stage}_colab.json")

# 下载图表
files.download("curriculum_curve.png")
```

## 故障排除

### 显存不足 (OOM)
- 减小 `--batch-size`: 64 → 32 → 16
- 减小 `--samples`: 1000 → 500 → 200
- 使用 `--max-tokens 64` 缩短生成长度

### 模型加载失败
- 检查模型路径是否正确
- 确认 LoRA 适配器文件存在
- 验证 base model 名称正确

### 数据加载失败
- 确认 `val_fixed.json` 在 `data/curriculum/` 目录
- 检查 JSON 格式是否为 Alpaca 格式

## 参考

- [vLLM 文档](https://docs.vllm.ai/)
- [Colab 免费 GPU 限制](https://research.google.com/colaboratory/faq.html#gpu-availability)
- 本地训练日志: `results/curriculum/overnight.log`
