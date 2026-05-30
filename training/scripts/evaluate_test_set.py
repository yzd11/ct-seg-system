"""测试集评估：HD95, ASSD, Precision, Recall（per-case fold-holdout）"""
import os, sys, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import distance_transform_edt

from dataset import label_mapping
from models import MODEL_REGISTRY
from utils.metrics import calculate_hd95_single

MODELS = ['unet', 'resunet', 'unet_pp', 'attention_unet_pp']
DATA_ROOT = 'H:\\LiTS'
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def surface_points(mask):
    """提取二值掩码的表面像素（mask 减去腐蚀结果）"""
    from scipy.ndimage import binary_erosion
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros((0, 2), dtype=int)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)))
    surf = mask & (~eroded)
    return np.argwhere(surf)


def assd_single(pred, gt, spacing=1.0):
    """平均对称表面距离 (Average Symmetric Surface Distance)"""
    surf_pred = surface_points(pred)
    surf_gt = surface_points(gt)

    if len(surf_pred) == 0 or len(surf_gt) == 0:
        return float('nan')

    # Distance from each pred surface point to nearest gt surface point
    dist_map_gt = distance_transform_edt(~gt.astype(bool))
    d1 = dist_map_gt[surf_pred[:, 0], surf_pred[:, 1]] * spacing

    # Distance from each gt surface point to nearest pred surface point
    dist_map_pred = distance_transform_edt(~pred.astype(bool))
    d2 = dist_map_pred[surf_gt[:, 0], surf_gt[:, 1]] * spacing

    return (np.mean(d1) + np.mean(d2)) / 2.0


def precision_single(pred, gt):
    """Precision = TP / (TP + FP) = |pred ∩ gt| / |pred|"""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = (pred_bool & gt_bool).sum()
    pred_total = pred_bool.sum()
    if pred_total == 0:
        return float('nan')
    return tp / pred_total


def recall_single(pred, gt):
    """Recall = TP / (TP + FN) = |pred ∩ gt| / |gt|"""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = (pred_bool & gt_bool).sum()
    gt_total = gt_bool.sum()
    if gt_total == 0:
        return float('nan')
    return tp / gt_total


# ── 构建 test case → fold 映射 ─────────────────────────────────────
test_cases = []
with open('preprocess/test.txt') as f:
    for line in f:
        cid = line.strip()
        if cid:
            test_cases.append(cid)

case_to_fold = {}
for fold in range(5):
    with open(f'data/fold_{fold}/val.txt') as f:
        for line in f:
            case = line.strip()
            if not case:
                continue
            if not case.startswith('volume-'):
                case = f'volume-{case}'
            case_to_fold[case] = fold

print(f'Test cases: {len(test_cases)}')
for case in test_cases:
    fold = case_to_fold.get(case, 'NOT FOUND')
    print(f'  {case} -> fold {fold}')
print()

# ── 构建 per-fold 的 test case 列表 ─────────────────────────────────
fold_test_cases = defaultdict(list)
for case in test_cases:
    fold = case_to_fold.get(case)
    if fold is not None:
        fold_test_cases[fold].append(case)

# ── 主循环 ──────────────────────────────────────────────────────────
all_results = {}

for model_name in MODELS:
    print(f'{"="*60}')
    print(f'  {model_name}')
    print(f'{"="*60}')

    model_results = []

    for fold in range(5):
        cases = fold_test_cases.get(fold, [])
        if not cases:
            continue

        weight_path = f'models/{model_name}/best_fold{fold}_seed3407.pth'
        if not os.path.exists(weight_path):
            print(f'  Fold {fold}: SKIP (no checkpoint)')
            continue

        print(f'  Fold {fold}: {len(cases)} cases <- {weight_path}')

        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(in_channels=3, num_classes=3).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()

        for case in cases:
            img_dir = os.path.join(DATA_ROOT, case, 'Image')
            gt_dir = os.path.join(DATA_ROOT, case, 'GT')

            if not os.path.exists(img_dir):
                print(f'    {case}: SKIP (no images)')
                continue

            fnames = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))

            # 逐 slice 累积像素级统计
            tp1_sum, fp1_sum, fn1_sum = 0, 0, 0  # liver
            tp2_sum, fp2_sum, fn2_sum = 0, 0, 0  # tumor
            hd95_l, hd95_t = [], []
            assd_l, assd_t = [], []

            for fname in fnames:
                img_path = os.path.join(img_dir, fname)
                gt_path = os.path.join(gt_dir, fname)

                image = Image.open(img_path).convert('RGB')
                label = Image.open(gt_path).convert('L')

                transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE),
                                      interpolation=transforms.InterpolationMode.NEAREST),
                ])
                image = transform(image)
                label = transform(label)

                image_t = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
                label_np = np.array(label)
                label_np = np.vectorize(label_mapping.get)(label_np)
                label_t = torch.from_numpy(label_np).long().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(image_t)
                    pred = torch.argmax(output, dim=1)

                pred_np = pred.cpu().numpy()[0]
                label_np2 = label_t.cpu().numpy()[0]

                # Precision / Recall 累积
                for cls, tp_s, fp_s, fn_s in [(1, None, None, None)]:
                    pass  # handled below

                # Liver (class 1)
                p1 = (pred_np == 1)
                g1 = (label_np2 == 1)
                tp1_sum += (p1 & g1).sum()
                fp1_sum += (p1 & (~g1)).sum()
                fn1_sum += ((~p1) & g1).sum()

                # Tumor (class 2)
                p2 = (pred_np == 2)
                g2 = (label_np2 == 2)
                tp2_sum += (p2 & g2).sum()
                fp2_sum += (p2 & (~g2)).sum()
                fn2_sum += ((~p2) & g2).sum()

                # HD95
                hd_l = calculate_hd95_single(p1, g1)
                hd_t = calculate_hd95_single(p2, g2)
                if hd_l is not None:
                    hd95_l.append(hd_l)
                if hd_t is not None:
                    hd95_t.append(hd_t)

                # ASSD
                a_l = assd_single(p1, g1)
                a_t = assd_single(p2, g2)
                if not math.isnan(a_l):
                    assd_l.append(a_l)
                if not math.isnan(a_t):
                    assd_t.append(a_t)

            # 汇总该 case
            prec1 = tp1_sum / (tp1_sum + fp1_sum) if (tp1_sum + fp1_sum) > 0 else float('nan')
            rec1 = tp1_sum / (tp1_sum + fn1_sum) if (tp1_sum + fn1_sum) > 0 else float('nan')
            prec2 = tp2_sum / (tp2_sum + fp2_sum) if (tp2_sum + fp2_sum) > 0 else float('nan')
            rec2 = tp2_sum / (tp2_sum + fn2_sum) if (tp2_sum + fn2_sum) > 0 else float('nan')

            model_results.append({
                'case': case,
                'fold': fold,
                'hd95_liver': sum(hd95_l) / len(hd95_l) if hd95_l else float('nan'),
                'hd95_tumor': sum(hd95_t) / len(hd95_t) if hd95_t else float('nan'),
                'assd_liver': sum(assd_l) / len(assd_l) if assd_l else float('nan'),
                'assd_tumor': sum(assd_t) / len(assd_t) if assd_t else float('nan'),
                'precision_liver': prec1,
                'recall_liver': rec1,
                'precision_tumor': prec2,
                'recall_tumor': rec2,
            })
            print(f'    {case}: HD95_T={model_results[-1]["hd95_tumor"]:.2f}  ASSD_T={model_results[-1]["assd_tumor"]:.2f}  Prec_T={prec2:.4f}  Rec_T={rec2:.4f}')

        del model
        torch.cuda.empty_cache()

    all_results[model_name] = model_results

# ── 汇总 ────────────────────────────────────────────────────────────
print(f'\n{"="*90}')
print(f'  Test Set Results (per-case fold-holdout, mean over {len(test_cases)} cases)')
print(f'{"="*90}')
header = f'{"Model":<22} {"HD95_Liver":>10} {"HD95_Tumor":>10} {"ASSD_Liver":>10} {"ASSD_Tumor":>10} {"Prec_Tumor":>10} {"Rec_Tumor":>10}'
print(header)
print('-' * len(header))

for model_name in MODELS:
    mr = all_results[model_name]
    if len(mr) < 10:
        continue

    def nanmean(vals):
        v = [x for x in vals if not math.isnan(x)]
        return sum(v) / len(v) if v else float('nan')

    hl = nanmean([r['hd95_liver'] for r in mr])
    ht = nanmean([r['hd95_tumor'] for r in mr])
    al = nanmean([r['assd_liver'] for r in mr])
    at = nanmean([r['assd_tumor'] for r in mr])
    pt = nanmean([r['precision_tumor'] for r in mr])
    rt = nanmean([r['recall_tumor'] for r in mr])

    print(f'{model_name:<22} {hl:8.2f} px {ht:8.2f} px {al:8.2f} px {at:8.2f} px {pt:10.4f} {rt:10.4f}')

print(f'{"="*90}\n')
print('Metrics: HD95=95% Hausdorff Distance, ASSD=Avg Symmetric Surface Distance,')
print('  Precision=TP/(TP+FP), Recall=TP/(TP+FN) — all tumor-focused.')
print('Each test case evaluated only by the fold where it was in VALIDATION (held out).')
