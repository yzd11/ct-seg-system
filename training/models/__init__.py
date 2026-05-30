# models/__init__.py
# 模型注册表：通过字符串名称动态导入对应模型类
# 在 train.py / test.py 中使用 args.model 选择模型，无需手动修改 import

from models.unet import UNet
from models.attention_unet import AttentionGate, AttentionUNet  # AttentionGate 被 attention_unet_pp 依赖
from models.resunet import ResUNet
from models.unet_pp import UNetPP
from models.attention_unet_pp import AttentionUNetPP

# 模型名称 → 类的映射字典，train.py 通过 MODEL_REGISTRY[args.model] 实例化
# 注：Attention U-Net（独立版）未纳入论文最终实验，仅保留 AttentionGate 作为依赖
MODEL_REGISTRY = {
    'unet':              UNet,
    'resunet':           ResUNet,
    'unet_pp':           UNetPP,
    'attention_unet_pp': AttentionUNetPP,
}
