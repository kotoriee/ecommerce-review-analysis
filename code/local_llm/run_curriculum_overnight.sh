#!/bin/bash
# 夜间训练任务 - 预计7小时完成
# S1已完成(64%)，继续S2-S5

set -e

cd /mnt/d/0321/ecommerce-review-analysis
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com

STAGES=(1200 2400 4800 8500)
EPOCHS=(3 3 3 3)
LRS=(2e-4 2e-4 1e-4 1e-4)
BATCH=(2 2 1 1)
GRAD_ACC=(8 8 16 16)
MODEL_DIR="models/curriculum"
RESULTS_DIR="results/curriculum"

mkdir -p $MODEL_DIR $RESULTS_DIR

START_TIME=$(date +%s)
START_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")

echo "========================================"
echo "Qwen3-4B 夜间训练任务"
echo "开始时间: $START_HUMAN"
echo "预计完成: 7小时后"
echo "========================================"

# 记录开始
 echo "{\"start\": \"$START_HUMAN\", \"status\": \"running\"}" > $RESULTS_DIR/overnight_status.json

for i in "${!STAGES[@]}"; do
    N=${STAGES[$i]}
    EP=${EPOCHS[$i]}
    LR=${LRS[$i]}
    BS=${BATCH[$i]}
    GA=${GRAD_ACC[$i]}

    STAGE_NUM=$((i+2))  # S2, S3, S4, S5

    if [ "$N" -eq 8500 ]; then
        DATA_FILE="train_full.json"
        MODEL_NAME="lora_s5_full"
        STAGE_NAME="S5(全量)"
    else
        DATA_FILE="train_$N.json"
        MODEL_NAME="lora_s${STAGE_NUM}_$N"
        STAGE_NAME="S${STAGE_NUM}($N条)"
    fi

    echo ""
    echo "========================================"
    echo "[$STAGE_NUM/4] 阶段 $STAGE_NAME"
    echo "数据: $DATA_FILE, Epochs: $EP, LR: $LR"
    echo "Batch: $BS, GradAcc: $GA"
    echo "开始时间: $(date '+%H:%M:%S')"
    echo "========================================"

    # 更新状态
    echo "{\"current_stage\": \"$STAGE_NAME\", \"progress\": \"$STAGE_NUM/4\"}" > $RESULTS_DIR/overnight_status.json

    # 训练
    STAGE_START=$(date +%s)

    python code/local_llm/train_sentiment.py \
        --data "data/curriculum/$DATA_FILE" \
        --output "$MODEL_DIR/$MODEL_NAME" \
        --epochs $EP \
        --lr $LR \
        --batch $BS \
        --grad-acc $GA \
        2>&1 | tee "$RESULTS_DIR/train_${N}.log"

    TRAIN_TIME=$(($(date +%s) - STAGE_START))
    echo "训练耗时: $((TRAIN_TIME/60))分$((TRAIN_TIME%60))秒"

    # 快速评估 (200条，约30分钟)
    echo ""
    echo "评估 $MODEL_NAME (200条快速验证)..."

    EVAL_START=$(date +%s)

    python code/local_llm/evaluate_unsloth.py \
        --model "$MODEL_DIR/$MODEL_NAME" \
        --data "data/curriculum/val_fixed.json" \
        --max-samples 200 \
        2>&1 | tee "$RESULTS_DIR/eval_${N}.log"

    EVAL_TIME=$(($(date +%s) - EVAL_START))
    echo "评估耗时: $((EVAL_TIME/60))分$((EVAL_TIME%60))秒"

    # 提取准确率
    ACC=$(grep "准确率:" "$RESULTS_DIR/eval_${N}.log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
    echo "阶段准确率: $ACC%"

    # 记录结果
    echo "{\"stage\": \"$STAGE_NAME\", \"accuracy\": \"$ACC\", \"train_time\": $TRAIN_TIME, \"eval_time\": $EVAL_TIME}" >> $RESULTS_DIR/overnight_results.jsonl

done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
END_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")

echo ""
echo "========================================"
echo "全部训练完成！"
echo "========================================"
echo "结束时间: $END_HUMAN"
echo "总耗时: $((TOTAL_TIME/3600))小时$(((TOTAL_TIME%3600)/60))分"
echo ""

# 汇总结果
echo "各阶段结果:"
grep -h "准确率:" $RESULTS_DIR/eval_*.log | grep -oP 'eval_[0-9]+\.log.*准确率: \K[0-9]+\.[0-9]+' || echo "请查看 results/curriculum/eval_*.log"

# 生成最终报告
echo ""
echo "生成对比报告..."
python code/local_llm/summarize_curriculum.py || echo "报告生成失败，请手动运行"

# 更新状态
echo "{\"end\": \"$END_HUMAN\", \"total_time\": $TOTAL_TIME, \"status\": \"completed\"}" > $RESULTS_DIR/overnight_status.json

echo ""
echo "✓ 任务完成，可以查看结果:"
echo "  - 模型: models/curriculum/"
echo "  - 日志: results/curriculum/*.log"
echo "  - 汇总: results/curriculum/curriculum_summary.json"
