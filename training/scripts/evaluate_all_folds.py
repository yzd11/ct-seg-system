"""批量 HD95 评估：4 模型 × 5 折 = 20 次推理"""
import os, sys, statistics as s
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import test as _test
from test import test_model, set_seed
_test.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models import MODEL_REGISTRY
from dataset import CustomImageDataset
from utils.losses import CombinedLoss
from torch.utils.data import DataLoader
from torchvision import transforms

MODELS = ['unet', 'resunet', 'unet_pp', 'attention_unet_pp']
DATA_ROOT = 'H:\\LiTS'
IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 3407

print(f'Device: {DEVICE}')
print(f'Data: {DATA_ROOT}')
set_seed(SEED)

# 测试集（只加载一次，复用）
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
])
test_dataset = CustomImageDataset(data_type='test', data_root=DATA_ROOT, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f'Test size: {len(test_dataset)} slices\n')

results = {}  # {model: {fold: {liver, tumor, ...}}}

for model_name in MODELS:
    print(f'{"="*60}')
    print(f'  {model_name}')
    print(f'{"="*60}')
    model_results = {}

    for fold in range(5):
        weight_path = f'models/{model_name}/best_fold{fold}_seed3407.pth'
        if not os.path.exists(weight_path):
            print(f'  Fold {fold}: SKIP (no checkpoint)')
            continue

        print(f'  Fold {fold}: loading {weight_path}...')

        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(in_channels=3, num_classes=3).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

        class_weights = torch.tensor([1.0, 1.0, 3.0]).to(DEVICE)
        criterion = CombinedLoss(class_weights=class_weights)

        save_dir = f'results/{model_name}_fold{fold}'
        _, _, _, _ = test_model(model, test_loader, criterion, save_dir, f'{model_name}_f{fold}')

        # Parse the summary.csv that test_model just wrote
        import csv
        summary_path = os.path.join(save_dir, 'summary.csv')
        with open(summary_path, 'r') as fh:
            reader = list(csv.reader(fh))
            if len(reader) >= 2:
                row = reader[1]
                model_results[fold] = {
                    'dice_liver': float(row[1]), 'dice_tumor': float(row[2]),
                    'iou_liver': float(row[3]), 'iou_tumor': float(row[4]),
                    'hd95_liver': float(row[5]), 'hd95_tumor': float(row[6]),
                    'infer_ms': float(row[7]), 'fps': float(row[8]),
                }
                print(f'    HD95_liver={row[5]} px  HD95_tumor={row[6]} px')

        del model
        torch.cuda.empty_cache()

    results[model_name] = model_results

# 汇总
print(f'\n{"="*80}')
print(f'  Final HD95 Summary (mean ± std over 5 folds)')
print(f'{"="*80}')
print(f'{"Model":<22} {"HD95_Liver":>16} {"HD95_Tumor":>16} {"Dice_Liver":>12} {"Dice_Tumor":>12}')
print(f'{'-'*22} {'-'*16} {'-'*16} {'-'*12} {'-'*12}')

for model_name in MODELS:
    if model_name not in results:
        continue
    mr = results[model_name]
    if len(mr) < 3:
        continue

    h_l = [mr[f]['hd95_liver'] for f in mr]
    h_t = [mr[f]['hd95_tumor'] for f in mr]
    d_l = [mr[f]['dice_liver'] for f in mr]
    d_t = [mr[f]['dice_tumor'] for f in mr]

    print(f'{model_name:<22} {s.mean(h_l):6.2f}±{s.stdev(h_l):.2f} px'
          f'  {s.mean(h_t):6.2f}±{s.stdev(h_t):.2f} px'
          f'  {s.mean(d_l):.4f}±{s.stdev(d_l):.4f}'
          f'  {s.mean(d_t):.4f}±{s.stdev(d_t):.4f}')

print(f'{"="*80}\n')
