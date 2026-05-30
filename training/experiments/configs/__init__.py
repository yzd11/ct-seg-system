# experiments/configs/__init__.py
# 实验配置加载器：从YAML文件读取超参数，返回dataclass实例

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """单个实验的完整配置"""
    # 基本标识
    experiment_name: str = 'default'
    model_name: str = 'unet'
    seed: int = 3407

    # 数据
    data_root: str = '/root/autodl-tmp/LiTS'
    img_size: int = 256
    in_channels: int = 3
    num_classes: int = 3

    # 训练
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 0.001
    weight_decay: float = 0.0

    # 学习率调度
    scheduler_name: str = 'steplr'  # steplr / cosine / plateau
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.1

    # 早停
    early_stopping_patience: int = 15
    early_stopping_metric: str = 'combined_dice'  # combined_dice / dice_tumor / val_loss

    # 混合精度
    use_amp: bool = True

    # 损失函数
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    tumor_weight: float = 3.0
    class_weights: list = field(default_factory=lambda: [1.0, 1.0, 3.0])

    # 模型保存
    save_metric_liver_weight: float = 0.4
    save_metric_tumor_weight: float = 0.6

    # 交叉验证
    fold_id: int = 0
    n_folds: int = 5

    # 数据增强等级: none / basic / medium / strong
    augmentation_level: str = 'basic'

    def __post_init__(self):
        """同步 class_weights 与 tumor_weight"""
        self.class_weights = [1.0, 1.0, self.tumor_weight]


def load_config(yaml_path: str) -> ExperimentConfig:
    """从YAML文件加载实验配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


def save_config(config: ExperimentConfig, yaml_path: str):
    """将配置保存为YAML文件"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False, allow_unicode=True)


def get_default_config(model_name: str = 'unet') -> ExperimentConfig:
    """获取默认配置（按模型名预设差异化参数）"""
    return ExperimentConfig(model_name=model_name, experiment_name=f'{model_name}_baseline')
