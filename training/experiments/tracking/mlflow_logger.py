# experiments/tracking/mlflow_logger.py
# MLflow 实验追踪集成
#
# 功能：
#   - 自动记录超参数、每epoch指标、最优模型权重
#   - 记录GPU内存、训练时间等系统信息
#   - 支持离线模式（MLFLOW_TRACKING_URI 未设置时仅写本地）
#
# 使用：
#   logger = MLflowLogger(config)
#   for epoch in range(epochs):
#       metrics = {...}
#       logger.log_epoch(epoch, metrics)
#   logger.log_best_model(best_metric)
#   logger.finish()

import os
import time
from typing import Dict, Optional

import torch


class MLflowLogger:
    """MLflow实验追踪日志器"""

    def __init__(self, config, fold_id: int = 0, seed: int = 3407):
        """
        config: ExperimentConfig dataclass 实例
        fold_id: 当前折编号（用于交叉验证）
        seed: 随机种子（用于区分同折不同种子的运行）
        """
        self.config = config
        self.fold_id = fold_id
        self.seed = seed
        self.start_time = time.time()
        self.best_metric = 0.0
        self._enabled = self._init_mlflow()

    def _init_mlflow(self) -> bool:
        """初始化MLflow，若不可用则降级为本地文件日志"""
        try:
            import mlflow
            self.mlflow = mlflow
            # 用模型名+折号作为实验名
            exp_name = f"{self.config.model_name}_fold{self.fold_id}_seed{self.seed}"
            self.mlflow.set_experiment(exp_name)
            run = self.mlflow.start_run(run_name=self.config.experiment_name)
            self.run = run
            self._log_params()
            self._log_system_info()
            return True
        except ImportError:
            print('[MLflowLogger] mlflow not installed, logging to local file only')
            return False
        except Exception as e:
            print(f'[MLflowLogger] mlflow init failed: {e}, logging to local file only')
            return False

    def _log_params(self):
        """记录所有超参数"""
        params = {
            'model_name': self.config.model_name,
            'img_size': self.config.img_size,
            'batch_size': self.config.batch_size,
            'lr': self.config.lr,
            'weight_decay': self.config.weight_decay,
            'scheduler': self.config.scheduler_name,
            'tumor_weight': self.config.tumor_weight,
            'ce_weight': self.config.ce_weight,
            'dice_weight': self.config.dice_weight,
            'augmentation': self.config.augmentation_level,
            'fold_id': self.fold_id,
            'seed': self.seed,
            'n_folds': self.config.n_folds,
            'early_stopping_patience': self.config.early_stopping_patience,
            'use_amp': self.config.use_amp,
        }
        self.mlflow.log_params(params)

    def _log_system_info(self):
        """记录GPU信息和Python环境"""
        import sys
        info = {'python_version': sys.version.split()[0]}
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['cuda_version'] = torch.version.cuda
        self.mlflow.log_params(info)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """记录一个epoch的所有指标"""
        if not self._enabled:
            return
        # epoch从1开始计数
        self.mlflow.log_metrics(metrics, step=epoch)

        # 记录GPU内存使用
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            self.mlflow.log_metrics({
                'gpu_mem_allocated_gb': mem_allocated,
                'gpu_mem_reserved_gb': mem_reserved,
            }, step=epoch)

    def log_best_model(self, combined_dice: float, model: torch.nn.Module,
                       save_path: str):
        """
        记录最优模型
        当 combined_dice > 历史最优时才保存
        """
        if combined_dice > self.best_metric:
            self.best_metric = combined_dice
            if self._enabled:
                self.mlflow.log_metric('best_combined_dice', combined_dice)
                self.mlflow.pytorch.log_model(model, 'best_model')

            # 同时保存原始权重文件
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    def log_artifact(self, local_path: str):
        """记录本地文件到MLflow"""
        if self._enabled and os.path.exists(local_path):
            self.mlflow.log_artifact(local_path)

    def log_figure(self, fig, name: str):
        """记录matplotlib图表"""
        if self._enabled:
            try:
                self.mlflow.log_figure(fig, name)
            except Exception:
                pass

    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """结束实验，记录最终汇总指标和训练时长"""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        if self._enabled:
            self.mlflow.log_metric('training_time_hours', hours)
            if final_metrics:
                self.mlflow.log_metrics(final_metrics)
            self.mlflow.end_run()

        # 始终打印关键汇总
        print(f'\n{"="*60}')
        print(f'Experiment finished: {self.config.experiment_name}')
        print(f'Fold: {self.fold_id}, Seed: {self.seed}')
        print(f'Training time: {hours:.2f} hours')
        print(f'Best combined_dice: {self.best_metric:.4f}')
        if final_metrics:
            for k, v in final_metrics.items():
                print(f'  {k}: {v:.4f}')
        print(f'{"="*60}')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
