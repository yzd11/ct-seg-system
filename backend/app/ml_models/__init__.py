# models/__init__.py
# 模型注册表：通过字符串名称动态导入对应模型类
# 在 train.py / test.py 中使用 args.model 选择模型，无需手动修改 import

from app.ml_models.unet import UNet
from app.ml_models.resunet import ResUNet                       # ② ResU-Net
from app.ml_models.unet_pp import UNetPP                        # ③ U-Net++
from app.ml_models.attention_unet_pp import AttentionUNetPP     # ④ Attention U-Net++

# 模型名称 → 类的映射字典，train.py 通过 MODEL_REGISTRY[args.model] 实例化
MODEL_REGISTRY = {
    'unet':        UNet,
    'resunet':     ResUNet,
    'unet_pp':     UNetPP,
    'att_unet_pp': AttentionUNetPP,
}
