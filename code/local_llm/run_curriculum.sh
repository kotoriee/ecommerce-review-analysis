#!/bin/bash
# 阶梯式微调训练流水线
# 从600条开始，逐步增加到8500条

set -e  # 遇到错误停止

cd /mnt/d/0321/ecommerce-review-analysis
source .venv/bin/activate

STAGES=(600 1200 2400 4800 8500)
EPOCHS=(5 3 3 3 3)
LRS=(3e-4 2e-4 2e-4 1e-4 1e-4)
MODEL_DIR="models/curriculum"
mkdir -p $MODEL_DIR results/curriculum

echo "========================================"
echo "Qwen3-4B 阶梯式微调"
echo "========================================"

for i in "${!STAGES[@]}"; do
    N=${STAGES[$i]}
    EP=${EPOCHS[$i]}
    LR=${LRS[$i]}

    # 确定文件名和模型名
    if [ "$N" -eq 8500 ]; then
        DATA_FILE="train_full.json"
        MODEL_NAME="lora_s5_full"
        STAGE_NAME="S5(全量)"
    else
        DATA_FILE="train_$N.json"
        MODEL_NAME="lora_s$((i+1))_$N"
        STAGE_NAME="S$((i+1))($N条)"
    fi

    echo ""
    echo "========================================"
    echo "阶段 $STAGE_NAME"
    echo "数据: $DATA_FILE, Epochs: $EP, LR: $LR"
    echo "========================================"

    # 训练
    python code/local_llm/train_sentiment.py \
        --data "data/curriculum/$DATA_FILE" \
        --output "$MODEL_DIR/$MODEL_NAME" \
        --epochs $EP \
        --lr $LR \
        --batch 2 \
        --grad-acc 8 \
        2>&1 | tee "results/curriculum/train_${N}.log"

    # 评估
    echo ""
    echo "评估 $MODEL_NAME..."
    python code/local_llm/evaluate_unsloth.py \
        --model "$MODEL_DIR/$MODEL_NAME" \
        --data "data/curriculum/val_fixed.json" \
        --output "results/curriculum/eval_${N}.json" \
        --max-samples 500 \
        2>&1 | tee "results/curriculum/eval_${N}.log"

done

echo ""
echo "========================================"
echo "全部训练完成！"
echo "========================================"
echo ""
echo "生成对比报告..."
python code/local_llm/summarize_curriculum.py
