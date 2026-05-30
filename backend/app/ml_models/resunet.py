# models/resunet.py
# ResU-Net 实现
#
# 核心改动（相对于标准 U-Net）：
#   将编码器和解码器中的 DoubleConv 全部替换为 ResBlock。
#   ResBlock 引入残差连接：输入通过 1×1 卷积对齐通道后与主路径输出相加，
#   缓解深层网络梯度消失，并让每个块专注于学习"残差"而非完整映射。
#
# 消融意义：与 U-Net 对比，单独验证残差连接对分割性能的贡献。
#
# 参考设计：He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
#
# 与 U-Net 的结构对比：
#   U-Net   : DoubleConv = Conv3×3+BN+ReLU → Conv3×3+BN+ReLU
#   ResU-Net: ResBlock   = (Conv3×3+BN+ReLU → Conv3×3+BN) + shortcut(Conv1×1+BN) → ReLU

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml_models.unet import OutConv   # 输出层与 U-Net 完全相同，直接复用


class ResBlock(nn.Module):
    """
    残差卷积块（替代 U-Net 中的 DoubleConv）

    主路径：Conv3×3 + BN + ReLU → Conv3×3 + BN
    shortcut：Conv1×1 + BN（通道对齐，当 in_channels ≠ out_channels 时必须）
    输出：ReLU(主路径输出 + shortcut输出)

    与标准 ResNet 的区别：
      - 不使用 bottleneck（无 1×1→3×3→1×1），保持与 DoubleConv 相近的参数规模
      - shortcut 始终存在（即使通道数相同也保留，保证梯度路径一致性）
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()

        # 主路径：两层 3×3 卷积
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # shortcut：1×1 卷积对齐通道数
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class ResDown(nn.Sequential):
    """
    编码器下采样块：MaxPool2d(2) → ResBlock
    与 U-Net 的 Down 结构一致，仅将 DoubleConv 替换为 ResBlock
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResDown, self).__init__(
            nn.MaxPool2d(2, stride=2),
            ResBlock(in_channels, out_channels)
        )


class ResUp(nn.Module):
    """
    解码器上采样块：上采样 + skip connection（concat）+ ResBlock

    与 U-Net 的 Up 结构一致，仅将 DoubleConv 替换为 ResBlock。
    concat 后的通道数为 in_channels（skip 通道 + 上采样通道），
    ResBlock 的 shortcut 负责将其对齐到 out_channels。
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(ResUp, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)

        # concat 后通道数为 in_channels，ResBlock 内 shortcut 负责通道对齐
        self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: 来自下一层（深层）的特征，需要上采样
        x2: 来自编码器同层的 skip connection 特征
        """
        x1 = self.up(x1)

        # 尺寸对齐（处理奇数尺寸边界情况，与 U-Net 保持一致）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 沿通道维度拼接（skip connection）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    """
    ResU-Net：将 U-Net 中所有 DoubleConv 替换为 ResBlock

    编码器/解码器的通道配置与标准 U-Net 完全相同，
    仅将卷积单元从 DoubleConv 改为 ResBlock，保证消融实验的可比性。

    各层通道数（base_c=64, bilinear=True）：
      in_conv : 3   → 64
      down1   : 64  → 128
      down2   : 128 → 256
      down3   : 256 → 512
      down4   : 512 → 512   bottleneck
      up1     : 512+512=1024 → 256
      up2     : 256+256=512  → 128
      up3     : 128+128=256  → 64
      up4     : 64+64=128    → 64
      out     : 64 → num_classes

    参数：
      in_channels : 输入通道数（默认 3，灰度图 RGB 复制）
      num_classes : 输出类别数（默认 3：背景/肝脏/肿瘤）
      bilinear    : 上采样方式（默认 True，双线性插值）
      base_c      : 基础通道数（默认 64）
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # 编码器（DoubleConv → ResBlock，通道配置与 UNet 一致）
        self.in_conv = ResBlock(in_channels, base_c)
        self.down1   = ResDown(base_c,     base_c * 2)
        self.down2   = ResDown(base_c * 2, base_c * 4)
        self.down3   = ResDown(base_c * 4, base_c * 8)
        self.down4   = ResDown(base_c * 8, base_c * 16 // factor)   # bottleneck

        # 解码器（Up → ResUp，通道配置与 UNet 一致）
        self.up1 = ResUp(base_c * 16,     base_c * 8 // factor, bilinear)
        self.up2 = ResUp(base_c * 8,      base_c * 4 // factor, bilinear)
        self.up3 = ResUp(base_c * 4,      base_c * 2 // factor, bilinear)
        self.up4 = ResUp(base_c * 2,      base_c,               bilinear)

        # 输出层（与 UNet 完全相同）
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，接口与 UNet.forward 完全一致。
        输入：(B, in_channels, H, W)
        输出：(B, num_classes, H, W) logits
        """
        # 编码器路径
        x1 = self.in_conv(x)    # (B, 64,  H,    W)
        x2 = self.down1(x1)     # (B, 128, H/2,  W/2)
        x3 = self.down2(x2)     # (B, 256, H/4,  W/4)
        x4 = self.down3(x3)     # (B, 512, H/8,  W/8)
        x5 = self.down4(x4)     # (B, 512, H/16, W/16)  bottleneck

        # 解码器路径（逐层上采样 + skip connection）
        x = self.up1(x5, x4)    # (B, 256, H/8,  W/8)
        x = self.up2(x,  x3)    # (B, 128, H/4,  W/4)
        x = self.up3(x,  x2)    # (B, 64,  H/2,  W/2)
        x = self.up4(x,  x1)    # (B, 64,  H,    W)

        logits = self.out_conv(x)   # (B, num_classes, H, W)
        return logits


if __name__ == '__main__':
    # 验证模型结构、输出尺寸和参数量
    from thop import profile

    model = ResUNet(in_channels=3, num_classes=3)
    input_tensor = torch.randn(1, 3, 256, 256)

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs     : {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")   # 期望 (1, 3, 256, 256)
