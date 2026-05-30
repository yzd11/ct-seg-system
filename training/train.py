# train.py
# 通用训练脚本，支持两种调用方式：
#
# 方式1 — YAML配置（推荐，新实验体系）：
#   python train.py --config experiments/configs/unet_baseline.yaml --fold_id 0
#
# 方式2 — 命令行参数（向后兼容，旧版接口）：
#   python train.py --model unet --tumor_weight 3.0 \
#                   --data_root /root/autodl-tmp/LiTS \
#                   --epochs 50 --batch_size 16 --lr 0.001 \
#                   --img_size 256 --num_workers 4
#
# 新增功能（相对旧版）：
#   - 早停（--patience 15，默认开启）
#   - 混合精度训练（--no_amp 关闭）
#   - MLflow实验追踪（安装mlflow后自动启用）
#   - K折交叉验证支持（--fold_id 0..4）

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import CustomImageDataset
from models import MODEL_REGISTRY
from utils.losses import CombinedLoss
from utils.metrics import calculate_metrics

# 新模块（阶段1新增）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.configs import ExperimentConfig, load_config
from training_lib.trainer import Trainer
from training_lib.augmentations import AugmentationPipeline


def set_seed(seed):
    """全局随机种子固定，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(config_or_args, is_train: bool = True):
    """
    构建数据增强pipeline
    config_or_args: ExperimentConfig 或 argparse.Namespace
    """
    img_size = getattr(config_or_args, 'img_size', 256)
    aug_level = getattr(config_or_args, 'augmentation_level', None)

    if aug_level is None:
        # 命令行模式 → 使用旧版transforms
        if is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90, interpolation=Image.NEAREST),
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            ])

    # YAML配置模式 → 使用新版AugmentationPipeline
    level = aug_level if is_train else 'none'
    return AugmentationPipeline(level=level, img_size=img_size)


def build_dataloaders(config_or_args, fold_id: int = 0):
    """
    构建训练/验证DataLoader
    支持K折交叉验证（从 data/fold_{fold_id}/ 读取train.txt和val.txt）
    """
    data_root = config_or_args.data_root
    batch_size = getattr(config_or_args, 'batch_size', 8)
    num_workers = getattr(config_or_args, 'num_workers', 4)
    img_size = getattr(config_or_args, 'img_size', 256)

    # 检查是否使用K折划分
    fold_train_txt = f'data/fold_{fold_id}/train.txt'
    fold_val_txt = f'data/fold_{fold_id}/val.txt'

    if os.path.exists(fold_train_txt) and os.path.exists(fold_val_txt):
        train_txt = fold_train_txt
        val_txt = fold_val_txt
        print(f'Using fold {fold_id} data split')
    else:
        train_txt = 'preprocess/train.txt'
        val_txt = 'preprocess/valid.txt'
        print('Using default 60/20/20 split (run data/split_kfold.py for CV)')

    # 构建数据集（新版AugmentationPipeline / 旧版transforms）
    train_transform = build_transforms(config_or_args, is_train=True)
    val_transform = build_transforms(config_or_args, is_train=False)

    # AugmentationPipeline需要paired模式
    is_paired = isinstance(train_transform, AugmentationPipeline)

    train_dataset = CustomImageDataset(
        data_type='train', data_root=data_root, transform=train_transform,
        txt_path=train_txt, paired_transform=is_paired
    )
    val_dataset = CustomImageDataset(
        data_type='valid', data_root=data_root, transform=val_transform,
        txt_path=val_txt, paired_transform=is_paired
    )

    print(f'Train size : {len(train_dataset)} slices')
    print(f'Valid size : {len(val_dataset)} slices')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def run_with_config(config: ExperimentConfig, fold_id: int = 0, seed: int = None):
    """使用ExperimentConfig运行训练（新版）"""
    if seed is not None:
        config.seed = seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(config.seed)

    # 数据
    train_loader, val_loader = build_dataloaders(config, fold_id=fold_id)

    # 模型
    ModelClass = MODEL_REGISTRY[config.model_name]
    model = ModelClass(in_channels=config.in_channels, num_classes=config.num_classes).to(device)
    print(f'Model: {config.model_name} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # 损失
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
    criterion = CombinedLoss(alpha=config.ce_weight, class_weights=class_weights)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 学习率调度
    if config.scheduler_name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
        )
    elif config.scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                          factor=0.5, patience=5)
    else:
        scheduler = None

    # 训练器
    trainer = Trainer(
        model, train_loader, val_loader, criterion, config,
        fold_id=fold_id, seed=config.seed, device=device
    )
    trainer.set_optimizer(optimizer, scheduler)

    final_metrics = trainer.fit()
    return final_metrics


def run_with_args(args):
    """使用命令行参数运行训练（旧版接口，内部转接Trainer）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(args.seed)

    # 构造简易config兼容Trainer
    class SimpleConfig:
        pass
    cfg = SimpleConfig()
    cfg.model_name = args.model
    cfg.data_root = args.data_root
    cfg.img_size = args.img_size
    cfg.in_channels = 3
    cfg.num_classes = 3
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.lr = args.lr
    cfg.weight_decay = 0
    cfg.scheduler_name = 'steplr'
    cfg.scheduler_step_size = 20
    cfg.scheduler_gamma = 0.1
    cfg.use_amp = not args.no_amp
    cfg.early_stopping_patience = args.patience
    cfg.ce_weight = 0.5
    cfg.dice_weight = 0.5
    cfg.tumor_weight = args.tumor_weight
    cfg.class_weights = [1.0, 1.0, args.tumor_weight]
    cfg.save_metric_liver_weight = 0.4
    cfg.save_metric_tumor_weight = 0.6
    cfg.n_folds = 5
    cfg.augmentation_level = args.augmentation_level
    cfg.save_dir = os.path.join('models', args.model)
    cfg.experiment_name = f'{args.model}_cli'

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, interpolation=Image.NEAREST),
        transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
    ])

    train_dataset = CustomImageDataset(data_type='train', data_root=args.data_root,
                                       transform=train_transform)
    val_dataset = CustomImageDataset(data_type='valid', data_root=args.data_root,
                                     transform=val_transform)
    print(f'Train size: {len(train_dataset)} slices')
    print(f'Valid size: {len(val_dataset)} slices')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(in_channels=3, num_classes=3).to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    class_weights = torch.tensor([1.0, 1.0, args.tumor_weight]).to(device)
    criterion = CombinedLoss(class_weights=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, cfg,
        fold_id=getattr(args, 'fold_id', 0), seed=args.seed, device=device
    )
    trainer.set_optimizer(optimizer, scheduler)
    trainer.fit()
    print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LiTS2017 医学图像分割训练脚本')

    # ── 两种模式 ──
    parser.add_argument('--config', type=str, default=None,
                        help='YAML实验配置文件路径（新版模式）')

    # ── 交叉验证参数（适用于两种模式）──
    parser.add_argument('--fold_id', type=int, default=0,
                        help='K折交叉验证的折编号（0~n_folds-1）')

    # ── 模型选择 ──
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'resunet', 'unet_pp', 'attention_unet_pp'],
                        help='模型名称（命令行模式；YAML模式忽略）')
    # 数据
    parser.add_argument('--data_root', type=str, default='F:\\LiTS\\LiTS',
                        help='PNG切片数据根目录')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮次（默认100）')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tumor_weight', type=float, default=3.0)
    # 新参数
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值（默认15，设为0禁用）')
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='禁用混合精度训练')
    # 数据增强
    parser.add_argument('--augmentation_level', type=str, default='basic',
                        choices=['none', 'basic', 'medium', 'strong'],
                        help='数据增强等级（命令行模式下默认basic）')

    args = parser.parse_args()

    if args.config:
        # ── YAML配置模式 ──
        config = load_config(args.config)
        # 命令行fold_id覆盖YAML中的fold_id
        run_with_config(config, fold_id=args.fold_id)
    else:
        # ── 命令行模式（向后兼容）──
        # 将早停相关参数注入args
        args.patience = args.patience
        run_with_args(args)
