# evaluate_nnunet.py
# 对 nnU-Net (3D) 的预测结果计算 Dice、IoU、HD95，与 2D 模型使用相同测试集
#
# 与 test.py 的区别：
#   test.py  : 逐切片推理，Dice/IoU 按切片计算再平均，HD95 为 2D（像素单位）
#   本脚本   : 逐 case（3D 体积）计算，Dice/IoU 按体积计算，HD95 为 3D（mm 单位）
#
# 运行前提：
#   - nnU-Net 预测已完成，预测文件位于 --pred_dir 指定目录
#   - 预测文件命名：liver_047.nii.gz（与 imagesTs 对应）
#   - 原始标签位于 /root/autodl-tmp/segmentations/segmentation-X.nii
#
# 运行命令：
#   python evaluate_nnunet.py \
#       --pred_dir /root/autodl-tmp/nnunet_predictions \
#       --seg_dir  /root/autodl-tmp/segmentations \
#       --test_txt preprocess/test.txt

import argparse
import csv
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import KDTree


# ── 3D 指标计算 ───────────────────────────────────────────────────────────────

def dice_3d(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    """
    计算单个 case 单个类别的 3D Dice
    pred/gt: 整个体积的标签数组 (D, H, W)
    """
    pred_mask = (pred == cls)
    gt_mask   = (gt   == cls)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0   # 预测和真值均为空，视为完全正确
    return float(2.0 * intersection / denom)


def iou_3d(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    """计算单个 case 单个类别的 3D IoU"""
    pred_mask = (pred == cls)
    gt_mask   = (gt   == cls)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union        = np.logical_or(pred_mask,  gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def hd95_3d(pred: np.ndarray, gt: np.ndarray, cls: int,
            spacing: tuple) -> float | None:
    """
    计算单个 case 单个类别的 3D HD95（mm 单位）

    参数：
      pred/gt : 整个体积的标签数组 (D, H, W)
      cls     : 目标类别
      spacing : 体素物理间距 (z_mm, y_mm, x_mm)，来自 NIfTI header

    返回：
      HD95（mm），若预测或真值为空则返回 None
    """
    pred_mask = (pred == cls)
    gt_mask   = (gt   == cls)

    if not pred_mask.any() or not gt_mask.any():
        return None

    def surface_coords(mask):
        """提取 3D 表面点坐标并转换为物理坐标（mm）"""
        struct = generate_binary_structure(3, 1)   # 6-连通
        eroded = binary_erosion(mask, structure=struct, border_value=1)
        surface = mask & ~eroded
        coords = np.argwhere(surface).astype(np.float64)
        coords *= np.array(spacing)                # 转为 mm 单位
        return coords

    pred_pts = surface_coords(pred_mask)
    gt_pts   = surface_coords(gt_mask)

    # 用 KDTree 加速最近邻查询（3D 体积点数多，暴力计算太慢）
    tree_gt   = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    d_pred2gt, _ = tree_gt.query(pred_pts)
    d_gt2pred, _ = tree_pred.query(gt_pts)

    all_distances = np.concatenate([d_pred2gt, d_gt2pred])
    return float(np.percentile(all_distances, 95))


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(args):
    # 读取测试集 case ID
    with open(args.test_txt) as f:
        test_ids = [int(line.strip()) for line in f if line.strip()]
    print(f'Test cases: {len(test_ids)}')

    results = []
    liver_dice_list, tumor_dice_list = [], []
    liver_iou_list,  tumor_iou_list  = [], []
    liver_hd95_list, tumor_hd95_list = [], []

    for case_id in sorted(test_ids):
        pred_path = os.path.join(args.pred_dir, f'liver_{case_id:03d}.nii.gz')
        gt_path   = os.path.join(args.seg_dir,  f'segmentation-{case_id}.nii')
        # 兼容 .nii.gz 格式的标签
        if not os.path.exists(gt_path):
            gt_path = gt_path + '.gz'

        if not os.path.exists(pred_path):
            print(f'  [SKIP] case {case_id}: prediction not found at {pred_path}')
            continue

        # 加载预测和标签
        pred_nii = nib.load(pred_path)
        gt_nii   = nib.load(gt_path)

        pred = np.round(pred_nii.get_fdata()).astype(np.int32)
        gt   = np.round(gt_nii.get_fdata()).astype(np.int32)

        # 从 NIfTI header 获取体素物理间距 (z, y, x) mm
        # zooms 顺序对应数组轴顺序
        spacing = tuple(float(s) for s in gt_nii.header.get_zooms()[:3])

        # 计算指标
        ld = dice_3d(pred, gt, cls=1)
        td = dice_3d(pred, gt, cls=2)
        li = iou_3d(pred, gt, cls=1)
        ti = iou_3d(pred, gt, cls=2)
        lh = hd95_3d(pred, gt, cls=1, spacing=spacing)
        th = hd95_3d(pred, gt, cls=2, spacing=spacing)

        liver_dice_list.append(ld)
        tumor_dice_list.append(td)
        liver_iou_list.append(li)
        tumor_iou_list.append(ti)
        if lh is not None:
            liver_hd95_list.append(lh)
        if th is not None:
            tumor_hd95_list.append(th)

        results.append({
            'case_id':    case_id,
            'dice_liver': round(ld, 4),
            'dice_tumor': round(td, 4),
            'iou_liver':  round(li, 4),
            'iou_tumor':  round(ti, 4),
            'hd95_liver': round(lh, 2) if lh is not None else 'N/A',
            'hd95_tumor': round(th, 2) if th is not None else 'N/A',
        })

        print(f'  case {case_id:3d} | Dice L={ld:.4f} T={td:.4f} | '
              f'IoU L={li:.4f} T={ti:.4f} | '
              f'HD95 L={lh:.1f}mm T={("N/A" if th is None else f"{th:.1f}mm")}')

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    mean = lambda lst: float(np.mean(lst)) if lst else float('nan')

    print('\n[nnU-Net (3D)] Test Results:')
    print(f'  Dice_liver    : {mean(liver_dice_list):.4f}')
    print(f'  Dice_tumor    : {mean(tumor_dice_list):.4f}')
    print(f'  IoU_liver     : {mean(liver_iou_list):.4f}')
    print(f'  IoU_tumor     : {mean(tumor_iou_list):.4f}')
    print(f'  HD95_liver    : {mean(liver_hd95_list):.2f} mm  ({len(liver_hd95_list)} valid cases)')
    print(f'  HD95_tumor    : {mean(tumor_hd95_list):.2f} mm  ({len(tumor_hd95_list)} valid cases)')
    print(f'  注意：HD95 单位为 mm（物理距离），2D 模型为 px（像素），不可直接比较')

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    score_path = os.path.join(args.output_dir, 'score.csv')
    with open(score_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary_path = os.path.join(args.output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'note'])
        writer.writerow(['Dice_liver',  f'{mean(liver_dice_list):.4f}', '3D volumetric'])
        writer.writerow(['Dice_tumor',  f'{mean(tumor_dice_list):.4f}', '3D volumetric'])
        writer.writerow(['IoU_liver',   f'{mean(liver_iou_list):.4f}',  '3D volumetric'])
        writer.writerow(['IoU_tumor',   f'{mean(tumor_iou_list):.4f}',  '3D volumetric'])
        writer.writerow(['HD95_liver',  f'{mean(liver_hd95_list):.2f}', 'mm, 3D surface'])
        writer.writerow(['HD95_tumor',  f'{mean(tumor_hd95_list):.2f}', 'mm, 3D surface'])

    print(f'\nScores saved to : {score_path}')
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nnU-Net 3D 预测结果评估脚本')
    parser.add_argument('--pred_dir',   type=str,
                        default='/root/autodl-tmp/nnunet_predictions',
                        help='nnU-Net 预测输出目录')
    parser.add_argument('--seg_dir',    type=str,
                        default='/root/autodl-tmp/segmentations',
                        help='原始标签目录（segmentation-X.nii）')
    parser.add_argument('--test_txt',   type=str,
                        default='preprocess/test.txt',
                        help='测试集 case ID 文件')
    parser.add_argument('--output_dir', type=str,
                        default='results/nnunet',
                        help='评估结果保存目录')
    args = parser.parse_args()
    main(args)
