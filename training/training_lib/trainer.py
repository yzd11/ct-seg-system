# training/trainer.py
# 通用训练器：TensorBoard + 早停 + AMP + 深度监督 + MLflow(可选)
#
# 日志策略（优先级从高到低）：
#   1. TensorBoard — 始终启用，曲线可视化
#   2. MLflow — 可选，实验追踪对比
#   3. 本地 CSV — MLflow不可用时自动降级
#
# 使用示例：
#   from training.trainer import Trainer
#   trainer = Trainer(model, train_loader, val_loader, criterion, config,
#                     fold_id=0, seed=3407)
#   best_metrics = trainer.fit()

import csv
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.metrics import calculate_metrics


class EarlyStopping:
    """早停管理器"""
    def __init__(self, patience: int = 15, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if self.mode == 'max':
            improved = metric > self.best_metric + self.min_delta
        else:
            improved = metric < self.best_metric - self.min_delta

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class Trainer:
    """通用训练器（TensorBoard + MLflow双日志）"""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion,
                 config,
                 fold_id: int = 0,
                 seed: int = 3407,
                 device: Optional[torch.device] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.fold_id = fold_id
        self.seed = seed

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        cfg = config
        self.epochs = getattr(cfg, 'epochs', 100)
        self.save_metric_liver_w = getattr(cfg, 'save_metric_liver_weight', 0.4)
        self.save_metric_tumor_w = getattr(cfg, 'save_metric_tumor_weight', 0.6)
        self.use_amp = getattr(cfg, 'use_amp', True) and self.device.type == 'cuda'
        self.patience = getattr(cfg, 'early_stopping_patience', 15)
        self.model_name = getattr(cfg, 'model_name', 'unet')

        self.early_stopping = EarlyStopping(patience=self.patience, mode='max')
        self.scaler = GradScaler() if self.use_amp else None

        # ── TensorBoard（始终启用）─────────────────────────────────────────
        tb_dir = os.path.join('runs', self.model_name, f'fold{fold_id}_seed{seed}')
        self.writer = SummaryWriter(tb_dir)
        print(f'TensorBoard: {tb_dir}')

        # ── MLflow（可选）──────────────────────────────────────────────────
        self._mlflow = self._init_mlflow()

        # ── 本地CSV（MLflow不可用时兜底）─────────────────────────────────
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None

        # 权重保存路径
        self.save_dir = getattr(cfg, 'save_dir', os.path.join('models', self.model_name))
        os.makedirs(self.save_dir, exist_ok=True)

        self._best_metrics = {}

    def _init_mlflow(self):
        try:
            import mlflow
            exp_name = f"{self.model_name}_fold{self.fold_id}_seed{self.seed}"
            mlflow.set_experiment(exp_name)
            mlflow.start_run(run_name=self.config.experiment_name)
            params = {
                'model': self.model_name, 'fold': self.fold_id, 'seed': self.seed,
                'img_size': getattr(self.config, 'img_size', 256),
                'batch_size': getattr(self.config, 'batch_size', 8),
                'lr': getattr(self.config, 'lr', 0.001),
                'tumor_weight': getattr(self.config, 'tumor_weight', 3.0),
                'augmentation': getattr(self.config, 'augmentation_level', 'basic'),
            }
            mlflow.log_params(params)
            print(f'MLflow: {exp_name} initialized')
            return mlflow
        except Exception as e:
            print(f'MLflow unavailable ({e}), using CSV fallback')

            # 创建本地CSV
            os.makedirs('experiments/logs', exist_ok=True)
            csv_name = f'{self.model_name}_fold{self.fold_id}_seed{self.seed}.csv'
            self._csv_path = os.path.join('experiments/logs', csv_name)
            self._csv_file = open(self._csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                'epoch', 'loss_train', 'loss_valid',
                'dice_liver_train', 'dice_liver_valid',
                'dice_tumor_train', 'dice_tumor_valid',
                'combined_dice', 'lr', 'epoch_time_s',
            ])
            print(f'CSV log: {self._csv_path}')
            return None

    def _log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """记录一个epoch到 TensorBoard + MLflow/CSV"""
        # TensorBoard（始终记录）
        for k, v in metrics.items():
            # 统一指标名以便跨模型对比
            tag = k.replace('/', '_')
            self.writer.add_scalar(tag, v, epoch)

        # MLflow 或 CSV
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=epoch)
        elif self._csv_writer is not None:
            self._csv_writer.writerow([
                epoch,
                metrics.get('loss_train', 0),
                metrics.get('loss_valid', 0),
                metrics.get('dice_liver_train', 0),
                metrics.get('dice_liver_valid', 0),
                metrics.get('dice_tumor_train', 0),
                metrics.get('dice_tumor_valid', 0),
                metrics.get('combined_dice', 0),
                metrics.get('lr', 0),
                metrics.get('epoch_time_s', 0),
            ])
            self._csv_file.flush()

    def _compute_combined_dice(self, dice_liver: float, dice_tumor: float) -> float:
        return self.save_metric_liver_w * dice_liver + self.save_metric_tumor_w * dice_tumor

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_dice_1 = 0.0
        running_dice_2 = 0.0
        n_batches = len(self.train_loader)

        with tqdm(self.train_loader,
                  desc=f'[Fold{self.fold_id}] Train',
                  unit='batch', leave=False) as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self._get_optimizer())
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, labels)
                    loss.backward()
                    self._get_optimizer().step()

                self._get_optimizer().zero_grad()

                if isinstance(outputs, list):
                    pred = torch.argmax(outputs[-1], dim=1)
                else:
                    pred = torch.argmax(outputs, dim=1)
                batch_dices, _ = calculate_metrics(pred, labels)

                running_dice_1 += batch_dices[0]
                running_dice_2 += batch_dices[1]
                running_loss += loss.item()

                t.set_postfix(
                    loss=f'{running_loss / (t.n + 1):.3f}',
                    L=f'{running_dice_1 / (t.n + 1):.3f}',
                    T=f'{running_dice_2 / (t.n + 1):.3f}',
                )

        return {
            'loss/train': running_loss / n_batches,
            'dice_liver/train': running_dice_1 / n_batches,
            'dice_tumor/train': running_dice_2 / n_batches,
        }

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        val_dice_1 = 0.0
        val_dice_2 = 0.0
        n_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader,
                                       desc=f'[Fold{self.fold_id}] Valid',
                                       unit='batch', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                if isinstance(outputs, list):
                    loss = self.criterion(outputs[-1], labels)
                    pred = torch.argmax(outputs[-1], dim=1)
                else:
                    loss = self.criterion(outputs, labels)
                    pred = torch.argmax(outputs, dim=1)

                val_dices, _ = calculate_metrics(pred, labels)
                val_dice_1 += val_dices[0]
                val_dice_2 += val_dices[1]
                val_loss += loss.item()

        return {
            'loss/valid': val_loss / n_batches,
            'dice_liver/valid': val_dice_1 / n_batches,
            'dice_tumor/valid': val_dice_2 / n_batches,
        }

    def fit(self) -> Dict[str, float]:
        best_combined = 0.0
        best_epoch = 0
        best_val_metrics = {}
        best_model_state = None

        print(f'\n{"="*60}')
        print(f'Training {self.model_name} | Fold {self.fold_id}/{self.config.n_folds}'
              f' | Seed {self.seed}')
        print(f'Epochs: {self.epochs} | Batch: {self.config.batch_size}'
              f' | AMP: {self.use_amp} | Patience: {self.patience}')
        print(f'Train: {len(self.train_loader.dataset)} | Valid: {len(self.val_loader.dataset)}')
        print(f'TensorBoard: runs/{self.model_name}/fold{self.fold_id}_seed{self.seed}')
        if self._csv_path:
            print(f'CSV log: {self._csv_path}')
        print(f'{"="*60}\n')

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            current_lr = self._get_optimizer().param_groups[0]['lr']
            self._scheduler_step()

            combined_dice = self._compute_combined_dice(
                val_metrics['dice_liver/valid'],
                val_metrics['dice_tumor/valid']
            )
            epoch_time = time.time() - epoch_start

            # 日志输出
            tqdm.write(
                f'Epoch [{epoch:3d}/{self.epochs}] | '
                f'Train Loss: {train_metrics["loss/train"]:.4f} | '
                f'Valid Loss: {val_metrics["loss/valid"]:.4f} | '
                f'Dice L: {val_metrics["dice_liver/valid"]:.4f} | '
                f'Dice T: {val_metrics["dice_tumor/valid"]:.4f} | '
                f'Combined: {combined_dice:.4f} | '
                f'LR: {current_lr:.2e} | '
                f'Time: {epoch_time:.1f}s'
            )

            # 三路日志：TensorBoard + MLflow/CSV
            self._log_epoch(epoch, {
                'loss_train': train_metrics['loss/train'],
                'loss_valid': val_metrics['loss/valid'],
                'dice_liver_train': train_metrics['dice_liver/train'],
                'dice_liver_valid': val_metrics['dice_liver/valid'],
                'dice_tumor_train': train_metrics['dice_tumor/train'],
                'dice_tumor_valid': val_metrics['dice_tumor/valid'],
                'combined_dice': combined_dice,
                'lr': current_lr,
                'epoch_time_s': epoch_time,
            })

            # 最优模型检查
            improved = self.early_stopping.step(combined_dice)
            if improved:
                best_combined = combined_dice
                best_epoch = epoch
                best_val_metrics = {
                    'best_dice_liver': val_metrics['dice_liver/valid'],
                    'best_dice_tumor': val_metrics['dice_tumor/valid'],
                    'best_combined_dice': combined_dice,
                    'best_epoch': best_epoch,
                }
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

                save_path = os.path.join(self.save_dir,
                                         f'best_fold{self.fold_id}_seed{self.seed}.pth')
                torch.save(self.model.state_dict(), save_path)
                tqdm.write(f'  >>> Best model saved (epoch {best_epoch}, '
                           f'combined={combined_dice:.4f})')
            else:
                tqdm.write(f'  No improvement ({self.early_stopping.counter}/'
                           f'{self.patience}, best={best_combined:.4f} @ epoch {best_epoch})')

            if self.early_stopping.should_stop:
                tqdm.write(f'\nEarly stopping triggered at epoch {epoch}')
                break

        # 收尾
        self.writer.add_text('summary/best', str(best_val_metrics))
        self.writer.close()

        if self._csv_file:
            self._csv_file.close()

        if self._mlflow is not None:
            self._mlflow.log_metrics(best_val_metrics)
            self._mlflow.end_run()

        print(f'\n{"="*60}')
        print(f'Finished: {self.model_name} | Fold {self.fold_id} | Seed {self.seed}')
        print(f'Best combined_dice: {best_combined:.4f} @ epoch {best_epoch}')
        for k, v in best_val_metrics.items():
            print(f'  {k}: {v:.4f}')
        print(f'TensorBoard: tensorboard --logdir runs')
        print(f'{"="*60}')

        return best_val_metrics

    def _compute_loss(self, outputs, labels):
        if isinstance(outputs, list):
            return sum(self.criterion(o, labels) for o in outputs) / len(outputs)
        else:
            return self.criterion(outputs, labels)

    def _get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer, scheduler=None):
        self._optimizer = optimizer
        self._scheduler = scheduler

    def _scheduler_step(self):
        if self._scheduler is not None:
            self._scheduler.step()
