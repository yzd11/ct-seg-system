#!/usr/bin/env python3
# 从训练文本日志解析指标，回填 TensorBoard 事件文件
# 用法：python scripts/replay_tb.py logs/unet_fold0.log runs/unet/fold0_seed3407

import os, re, sys
from torch.utils.tensorboard import SummaryWriter

def parse_log(log_path):
    pattern = re.compile(
        r'Epoch \[\s*(\d+)/\s*\d+\] \| '
        r'Train Loss: ([\d.]+) \| Valid Loss: ([\d.]+) \| '
        r'Dice L: ([\d.]+) \| Dice T: ([\d.]+) \| '
        r'Combined: ([\d.]+) \| LR: ([\d.e+\-]+)'
    )
    records = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    'epoch': int(m.group(1)),
                    'loss_train': float(m.group(2)),
                    'loss_valid': float(m.group(3)),
                    'dice_liver_valid': float(m.group(4)),
                    'dice_tumor_valid': float(m.group(5)),
                    'combined_dice': float(m.group(6)),
                    'lr': float(m.group(7)),
                })
    return records

if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/unet_fold0.log'
    tb_dir = sys.argv[2] if len(sys.argv) > 2 else 'runs/unet/fold0_seed3407'

    records = parse_log(log_path)
    print(f'Parsed {len(records)} epochs from {log_path}')

    writer = SummaryWriter(tb_dir)
    for r in records:
        for k, v in r.items():
            if k != 'epoch':
                writer.add_scalar(k, v, r['epoch'])
    writer.close()
    print(f'TensorBoard events written to {tb_dir}')
    print(f'Run: tensorboard --logdir runs')
