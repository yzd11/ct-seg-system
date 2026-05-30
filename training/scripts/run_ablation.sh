#!/bin/bash
# 消融实验 — 本地 Windows Git Bash 执行
# 3 维度 × 1 种子 = 4 次训练（基线复用 U-Net fold=0）
#
# 前置条件：
#   1. 数据 H:\LiTS 可访问
#   2. python train.py 可直接运行（venv 已激活）

set -e

EXPERIMENTS=(
    "ablation_weight1"
    "ablation_weight5"
    "ablation_ceonly"
    "ablation_diceonly"
)
LOGDIR="logs/ablation"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Ablation Experiments — U-Net fold 0"
echo "  $(date)"
echo "============================================"

for exp in "${EXPERIMENTS[@]}"; do
    LOGFILE="${LOGDIR}/${exp}.log"
    echo ""
    echo ">>> $exp  ($(date +%H:%M:%S))"
    python train.py \
        --config "experiments/configs/${exp}.yaml" \
        --fold_id 0 \
        > "$LOGFILE" 2>&1
    echo "<<< $exp  DONE ($(date +%H:%M:%S))"
done

echo ""
echo "============================================"
echo "  ALL ABLATION EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "============================================"
echo ""
echo "Run: python scripts/summarize_ablation.py"
