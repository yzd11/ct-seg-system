# data/split_kfold.py
# K折交叉验证数据集划分（按case分层，保证每折肿瘤像素分布一致）
#
# 用法：
#   python data/split_kfold.py --data_root /root/autodl-tmp/LiTS --k 5 --seed 3407
#
# 输出（在 data/ 目录下）：
#   folds.json                          — 每折的 train/val case列表 + 肿瘤统计
#   fold_{i}/train.txt, fold_{i}/val.txt — 供 dataset.py 按折加载

import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np


def compute_tumor_ratio(case_path: str) -> float:
    """统计一个case的肿瘤像素占比"""
    gt_dir = os.path.join(case_path, 'GT')
    if not os.path.isdir(gt_dir):
        return 0.0
    total_px = 0
    tumor_px = 0
    for fname in os.listdir(gt_dir):
        if not fname.endswith('.png'):
            continue
        label = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        total_px += label.size
        tumor_px += int((label == 255).sum())
    return tumor_px / total_px if total_px > 0 else 0.0


def stratified_kfold(case_ids: list, tumor_ratios: dict, k: int, seed: int):
    """
    按肿瘤占比分层后K折划分
    层(strata)：无肿瘤(ratio=0) / 低肿瘤(0~0.005) / 中肿瘤(0.005~0.02) / 高肿瘤(>0.02)
    """
    random.seed(seed)

    strata = defaultdict(list)
    for cid in case_ids:
        r = tumor_ratios.get(cid, 0.0)
        if r == 0:
            strata['none'].append(cid)
        elif r < 0.005:
            strata['low'].append(cid)
        elif r < 0.02:
            strata['mid'].append(cid)
        else:
            strata['high'].append(cid)

    # 每层内打乱后均分到k折
    folds = [{'train': [], 'val': []} for _ in range(k)]
    for name, ids in strata.items():
        random.shuffle(ids)
        n = len(ids)
        fold_size = n // k
        remainder = n % k
        start = 0
        for i in range(k):
            extra = 1 if i < remainder else 0
            end = start + fold_size + extra
            val_ids = ids[start:end]
            train_ids = [x for j, x in enumerate(ids) if j < start or j >= end]
            folds[i]['val'].extend(val_ids)
            folds[i]['train'].extend(train_ids)
            start = end
        print(f'  Stratum [{name}]: {n} cases, ~{fold_size}/fold')

    return folds


def main():
    parser = argparse.ArgumentParser(description='K折交叉验证数据集划分')
    parser.add_argument('--data_root', type=str, required=True,
                        help='PNG切片数据根目录（如 /root/autodl-tmp/LiTS）')
    parser.add_argument('--k', type=int, default=5, help='折数（默认5）')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认为脚本所在目录）')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    # 发现所有case目录
    case_ids = sorted(d for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
    print(f'Found {len(case_ids)} cases in {args.data_root}')

    # 统计每个case的肿瘤占比
    print('Computing tumor ratios...')
    tumor_ratios = {}
    for cid in case_ids:
        r = compute_tumor_ratio(os.path.join(args.data_root, cid))
        tumor_ratios[cid] = r

    no_tumor = sum(1 for r in tumor_ratios.values() if r == 0)
    has_tumor = len(case_ids) - no_tumor
    print(f'Cases with tumor: {has_tumor}, without tumor: {no_tumor}')

    # K折分层划分
    folds = stratified_kfold(case_ids, tumor_ratios, args.k, args.seed)

    # 保存fold信息
    fold_info = {}
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(output_dir, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)

        # 统计该折的肿瘤分布
        val_tumor_ratios = [tumor_ratios[c] for c in fold['val']]
        train_tumor_ratios = [tumor_ratios[c] for c in fold['train']]

        fold_info[f'fold_{i}'] = {
            'train_cases': fold['train'],
            'val_cases': fold['val'],
            'train_count': len(fold['train']),
            'val_count': len(fold['val']),
            'val_mean_tumor_ratio': round(float(np.mean(val_tumor_ratios)), 6),
            'train_mean_tumor_ratio': round(float(np.mean(train_tumor_ratios)), 6),
        }

        # 写train.txt
        with open(os.path.join(fold_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(fold['train']))

        # 写val.txt
        with open(os.path.join(fold_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(fold['val']))

        print(f'Fold {i}: train={len(fold["train"])}, val={len(fold["val"])}, '
              f'val_tumor_ratio={fold_info[f"fold_{i}"]["val_mean_tumor_ratio"]:.4%}')

    # 保存全局fold信息
    json_path = os.path.join(output_dir, 'folds.json')
    with open(json_path, 'w') as f:
        json.dump(fold_info, f, indent=2, ensure_ascii=False)
    print(f'\nFold info saved to: {json_path}')
    print(f'Done. {args.k}-fold split with seed={args.seed}')


if __name__ == '__main__':
    main()
