#!/bin/bash
# 链式训练：等 U-Net++ 跑完 → 自动启动 Attention U-Net++
# 用法：nohup bash scripts/chain_train.sh > logs/chain_master.log 2>&1 &
set -e

cd /root/UNet-LiTS2017-main
LOG1="logs/unet_pp_master.log"
MODEL2="attention_unet_pp"

echo "============================================"
echo "  Chain: unet_pp -> $MODEL2"
echo "  Started at $(date)"
echo "============================================"

while true; do
    if grep -q "ALL DONE" "$LOG1" 2>/dev/null; then
        echo ""
        echo "=== unet_pp FINISHED at $(date) ==="
        grep "Fold.*done" "$LOG1" | tail -5
        break
    fi
    # 打印 U-Net++ 最新进度
    PROGRESS=$(grep "Epoch \[" logs/unet_pp_fold*.log 2>/dev/null | tail -1 | grep -oP 'Epoch \[\s*\d+/\d+\].*' || echo "no epoch yet")
    echo "[$(date +%H:%M)] $PROGRESS"
    sleep 300
done

# 创建配置
if [ ! -f "experiments/configs/${MODEL2}.yaml" ]; then
    sed "s/model_name: unet_pp/model_name: $MODEL2/" \
        experiments/configs/unet_pp.yaml \
        > experiments/configs/${MODEL2}.yaml
    echo "Created experiments/configs/${MODEL2}.yaml"
fi

# 启动 Attention U-Net++
echo "Starting $MODEL2 at $(date)..."
nohup bash -c "
for f in 0 1 2 3 4; do
    python train.py \
        --config experiments/configs/${MODEL2}.yaml \
        --fold_id \$f \
        --num_workers 4 \
        > logs/${MODEL2}_fold\${f}.log 2>&1
    echo \"Fold \$f done at \$(date)\"
done
echo 'ALL DONE at \$(date)'
" > "logs/${MODEL2}_master.log" 2>&1 &

echo "PID: $!"
echo "Log: logs/${MODEL2}_master.log"
echo "=== Chain script exiting at $(date) ==="
