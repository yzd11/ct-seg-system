# test.py
# 通用测试脚本，通过 --model 参数加载对应权重，输出指标并保存预测图
#
# 使用示例：
#   python test.py --model unet
#   python test.py --model unet --data_root /root/autodl-tmp/LiTS
#
# 输出：
#   results/{model_name}/score.csv         —— 每个 batch 的 Dice/IoU 详细数值
#   results/{model_name}/img_pred_gt/      —— 前 1000 张的 [原图|预测|真值] 拼接图
#
# 注意：测试集不做任何数据增强，只做尺寸对齐（保证评估客观）

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import CustomImageDataset, inverse_mapping
from models import MODEL_REGISTRY
from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics, calculate_hd95_single


def set_seed(seed):
    """固定随机种子（测试阶段一般不影响结果，但保持习惯）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image(images, preds, labels, name, save_dir):
    """
    将原图、预测掩码、真实标签横向拼接并保存为 PNG

    拼接顺序（从上到下）：
      1. 原始 CT 图像（灰度转 RGB 显示）
      2. 模型预测的分割掩码（类别索引反映射为 0/128/255）
      3. 真实分割标签（同上）

    参数：
      images  : (B, C, H, W) float tensor，值域 [0, 1]
      preds   : (B, H, W)    long tensor，值 0/1/2
      labels  : (B, H, W)    long tensor，值 0/1/2
      name    : 保存文件名（含后缀）
      save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    images = images.to('cpu')
    preds  = preds.to('cpu')
    labels = labels.to('cpu')

    # 图像：[0,1] → [0,255] uint8
    images = (images * 255).to(torch.uint8)

    # 预测/标签：类别索引 (0/1/2) → 像素值 (0/128/255)，再复制为 RGB
    preds_vis  = torch.from_numpy(np.vectorize(inverse_mapping.get)(preds.numpy()))
    preds_vis  = torch.stack([preds_vis,  preds_vis,  preds_vis],  dim=1).to(torch.uint8)

    labels_vis = torch.from_numpy(np.vectorize(inverse_mapping.get)(labels.numpy()))
    labels_vis = torch.stack([labels_vis, labels_vis, labels_vis], dim=1).to(torch.uint8)

    # make_grid 将 batch 内所有样本横向排列成一张大图
    grid_img   = make_grid(images)     # 原图
    grid_pred  = make_grid(preds_vis)  # 预测
    grid_label = make_grid(labels_vis) # 真值

    # 三者纵向拼接（上:原图，中:预测，下:真值）
    concat = torch.cat([grid_img, grid_pred, grid_label], dim=1)

    # (C, H, W) → (W, H, C) → PIL Image（注意 permute 顺序）
    concat = concat.permute(2, 1, 0)
    concat_img = Image.fromarray(concat.numpy())
    concat_img.save(os.path.join(save_dir, name))


def test_model(model, test_loader, criterion, save_dir, model_name):
    """
    测试主函数

    指标计算方式：
      - Dice / IoU：对每个 batch 计算后取均值（batch_size=1 时等价于样本均值）
      - HD95：逐张计算，跳过预测或真值为空的切片，最终取有效切片的均值
      - 推理速度：GPU 计时（torch.cuda.synchronize），跳过第一个 batch 的预热开销

    参数：
      model      : 已加载权重的模型（eval 模式）
      test_loader: 测试集 DataLoader（建议 batch_size=1）
      criterion  : 损失函数（仅用于记录 test_loss，不影响评估指标）
      save_dir   : 结果保存根目录
      model_name : 模型名称（用于日志打印）
    """
    os.makedirs(save_dir, exist_ok=True)
    img_save_dir = os.path.join(save_dir, 'img_pred_gt')

    model.eval()

    test_dice_1 = []   # 肝脏 Dice（每个 batch 一个值）
    test_dice_2 = []   # 肿瘤 Dice
    test_iou_1  = []   # 肝脏 IoU
    test_iou_2  = []   # 肿瘤 IoU
    hd95_liver  = []   # 肝脏 HD95（仅含有效值，空掩码切片跳过）
    hd95_tumor  = []   # 肿瘤 HD95
    test_loss   = 0.0
    save_count  = 0    # 已保存的可视化图片数量

    # 推理计时：跳过第一个 batch（GPU 预热），从第二个 batch 开始计时
    infer_times = []   # 单位：毫秒/张
    use_cuda    = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(
                tqdm(test_loader, desc=f'Testing [{model_name}]')):

            inputs, labels = inputs.to(device), labels.to(device)

            # ── 推理计时 ──────────────────────────────────────────────────────
            if use_cuda:
                torch.cuda.synchronize()   # 确保上一个 GPU 操作完成
            t_start = time.perf_counter()

            outputs = model(inputs)

            if use_cuda:
                torch.cuda.synchronize()   # 确保当前推理完成再读时间
            t_end = time.perf_counter()

            # 跳过第一个 batch（预热），之后每张图的平均时间
            if batch_idx > 0:
                elapsed_ms = (t_end - t_start) * 1000 / inputs.size(0)
                infer_times.append(elapsed_ms)

            # ── Dice & IoU ────────────────────────────────────────────────────
            preds = torch.argmax(outputs, dim=1)
            batch_dices, batch_ious = calculate_metrics(preds, labels)
            test_dice_1.append(batch_dices[0])
            test_dice_2.append(batch_dices[1])
            test_iou_1.append(batch_ious[0])
            test_iou_2.append(batch_ious[1])

            # ── HD95（逐张计算，batch_size=1 时 i=0 即当前张）────────────────
            preds_np  = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(preds_np.shape[0]):
                pred_i   = preds_np[i]
                label_i  = labels_np[i]

                # 肝脏（class 1）
                hd_liver = calculate_hd95_single(pred_i == 1, label_i == 1)
                if hd_liver is not None:
                    hd95_liver.append(hd_liver)

                # 肿瘤（class 2）
                hd_tumor = calculate_hd95_single(pred_i == 2, label_i == 2)
                if hd_tumor is not None:
                    hd95_tumor.append(hd_tumor)

            # ── 可视化保存（前 1000 个 batch）────────────────────────────────
            if save_count < 1000:
                save_image(inputs, preds, labels,
                           name=f'{save_count}.png',
                           save_dir=img_save_dir)
                save_count += 1

            # ── Loss（与评估指标独立）────────────────────────────────────────
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # ── 汇总指标 ──────────────────────────────────────────────────────────────
    test_loss  = test_loss / len(test_loader)
    dice_liver = sum(test_dice_1) / len(test_dice_1)
    dice_tumor = sum(test_dice_2) / len(test_dice_2)
    iou_liver  = sum(test_iou_1)  / len(test_iou_1)
    iou_tumor  = sum(test_iou_2)  / len(test_iou_2)
    hd95_l     = sum(hd95_liver)  / len(hd95_liver)  if hd95_liver else float('nan')
    hd95_t     = sum(hd95_tumor)  / len(hd95_tumor)  if hd95_tumor else float('nan')
    avg_ms     = sum(infer_times) / len(infer_times)  if infer_times else float('nan')
    fps        = 1000.0 / avg_ms                      if avg_ms > 0  else float('nan')

    tqdm.write(
        f'\n[{model_name}] Test Results:\n'
        f'  Loss          : {test_loss:.4f}\n'
        f'  Dice_liver    : {dice_liver:.4f}\n'
        f'  Dice_tumor    : {dice_tumor:.4f}\n'
        f'  IoU_liver     : {iou_liver:.4f}\n'
        f'  IoU_tumor     : {iou_tumor:.4f}\n'
        f'  HD95_liver    : {hd95_l:.2f} px  ({len(hd95_liver)} valid slices)\n'
        f'  HD95_tumor    : {hd95_t:.2f} px  ({len(hd95_tumor)} valid slices)\n'
        f'  Infer speed   : {avg_ms:.2f} ms/img  ({fps:.1f} FPS)'
    )

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(save_dir, 'score.csv')
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dice_liver', 'Dice_tumor', 'IoU_liver', 'IoU_tumor'])
        for d1, d2, i1, i2 in zip(test_dice_1, test_dice_2, test_iou_1, test_iou_2):
            writer.writerow([d1, d2, i1, i2])

    # 汇总行单独保存，便于跨模型对比
    summary_path = os.path.join(save_dir, 'summary.csv')
    with open(summary_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Dice_liver', 'Dice_tumor',
                         'IoU_liver', 'IoU_tumor',
                         'HD95_liver', 'HD95_tumor',
                         'Infer_ms', 'FPS'])
        writer.writerow([model_name,
                         f'{dice_liver:.4f}', f'{dice_tumor:.4f}',
                         f'{iou_liver:.4f}',  f'{iou_tumor:.4f}',
                         f'{hd95_l:.2f}',     f'{hd95_t:.2f}',
                         f'{avg_ms:.2f}',     f'{fps:.1f}'])

    print(f'Scores saved to : {csv_path}')
    print(f'Summary saved to: {summary_path}')

    return test_dice_1, test_dice_2, test_iou_1, test_iou_2


if __name__ == '__main__':
    # ── 命令行参数解析 ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='LiTS2017 医学图像分割测试脚本')

    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'resunet', 'unet_pp', 'attention_unet_pp'],
                        help='选择测试的模型（需与训练时一致，默认: unet）')

    # !! 服务器上运行时请传入 --data_root /root/autodl-tmp/LiTS !!
    parser.add_argument('--data_root', type=str, default='F:\\LiTS\\LiTS',
                        help='PNG 切片数据根目录（默认为本地路径）')

    # 权重路径：默认加载对应模型的 best.pth
    parser.add_argument('--weights', type=str, default=None,
                        help='模型权重路径（默认: models/{model}/best.pth）')

    # 肿瘤权重需与训练时一致，用于 test_loss 计算（不影响 Dice/IoU 评估）
    parser.add_argument('--tumor_weight', type=float, default=3.0,
                        help='肿瘤类别损失权重（需与训练时一致，默认: 3.0）')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='测试批大小（默认: 1，保证每张图单独评估）')
    parser.add_argument('--seed',       type=int, default=3407,
                        help='随机种子（默认: 3407）')
    # 必须与训练时的 --img_size 保持一致，否则模型输入尺寸不匹配
    parser.add_argument('--img_size',   type=int, default=256,
                        help='输入图像分辨率（默认: 256，需与训练时一致）')

    args = parser.parse_args()

    # ── 基础设置 ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    set_seed(args.seed)

    # ── 测试集 DataLoader ─────────────────────────────────────────────────────
    # 测试集只做尺寸对齐，严禁数据增强
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
    ])
    test_dataset = CustomImageDataset(data_type='test', data_root=args.data_root,
                                      transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    print(f'Test size: {len(test_dataset)} slices')

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(in_channels=3, num_classes=3).to(device)

    # 默认从 models/{model_name}/best.pth 加载
    weights_path = args.weights or os.path.join('models', args.model, 'best.pth')
    print(f'Loading weights from: {weights_path}')
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # ── 损失函数（与训练时保持一致）─────────────────────────────────────────
    class_weights = torch.tensor([1.0, 1.0, args.tumor_weight]).to(device)
    criterion = CombinedLoss(class_weights=class_weights)

    # ── 测试结果目录 ──────────────────────────────────────────────────────────
    results_dir = os.path.join('results', args.model)

    # ── 开始测试 ──────────────────────────────────────────────────────────────
    test_model(model, test_loader, criterion, results_dir, args.model)
