# models/unet.py
# 标准 2D U-Net 实现
#
# 网络结构（base_c=64 时的通道数）：
#   编码器：64 → 128 → 256 → 512 → 512(bottleneck, bilinear模式下512//2=256)
#   解码器：逐层上采样并与编码器对应层做 skip connection（concat）
#   输出：num_classes 个通道的 logits（未经 softmax）
#
# 参考论文：Ronneberger et al., "U-Net: Convolutional Networks for
#           Biomedical Image Segmentation", MICCAI 2015.

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """
    U-Net 的基本卷积单元：Conv → BN → ReLU → Conv → BN → ReLU
    bias=False：与 BatchNorm 联用时 bias 项被 BN 的 beta 参数替代，节省参数
    mid_channels：两层卷积间的通道数，默认等于 out_channels；
                  在 bilinear 上采样模式下解码器会设为 in_channels // 2 以减少参数量
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    """
    编码器下采样块：MaxPool2d(2) → DoubleConv
    每次将特征图尺寸减半，通道数翻倍（逐步提取高语义特征）
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    """
    解码器上采样块：上采样 + skip connection（concat） + DoubleConv

    两种上采样方式：
      bilinear=True ：双线性插值（无可学习参数，更轻量，推荐）
                      上采样后通道不变，DoubleConv 的 mid_channels = in_channels//2
      bilinear=False：转置卷积（有可学习参数，通道先减半再 concat）

    skip connection 的尺寸对齐：
      当输入尺寸不能被 2 整除时，双线性上采样后尺寸与 skip 特征可能差 1 像素，
      使用 F.pad 补零对齐（仅在右侧/底部补）
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: 来自下一层（深层）的特征，需要上采样
        x2: 来自编码器同层的 skip connection 特征
        """
        x1 = self.up(x1)

        # 计算尺寸差值并补零（处理奇数尺寸边界情况）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 顺序：padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 沿通道维度拼接（skip connection 核心操作）
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    """
    输出层：1×1 卷积将通道数映射到类别数
    输出为 logits（未归一化），损失函数内部会做 softmax/log_softmax
    """
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    """
    完整 U-Net 网络

    参数：
      in_channels  : 输入通道数。当前使用 RGB 复制的灰度图，设为 3；
                     若改为单通道灰度输入需同步修改 dataset.py 中的 convert("L")
      num_classes  : 输出类别数。本项目：0=背景，1=肝脏，2=肿瘤，共 3 类
      bilinear     : True 使用双线性上采样（参数量更少），False 使用转置卷积
      base_c       : 第一层卷积的基础通道数，后续按 2 倍递增
                     base_c=64 时总参数量约 31M（in_channels=3）

    各层通道数（bilinear=True, base_c=64）：
      in_conv : 3   → 64
      down1   : 64  → 128
      down2   : 128 → 256
      down3   : 256 → 512
      down4   : 512 → 512  （bottleneck，bilinear 时 factor=2 所以是 512//2=256 → 存疑，实际是 512）
      up1     : 512+512 → 256
      up2     : 256+256 → 128
      up3     : 128+128 → 64
      up4     : 64+64   → 64
      out     : 64 → num_classes
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # bilinear=True 时 bottleneck 通道减半（由 DoubleConv 内部的 mid_channels 控制）
        factor = 2 if bilinear else 1

        # 编码器
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)   # bottleneck

        # 解码器（in_channels = 上采样通道 + skip 通道）
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8,  base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4,  base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2,  base_c,               bilinear)

        # 输出层
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        输入：x，shape (B, in_channels, H, W)
        输出：logits，shape (B, num_classes, H, W)
        """
        # 编码器路径（保存各层特征供 skip connection 使用）
        x1 = self.in_conv(x)     # (B, 64,  H,   W)
        x2 = self.down1(x1)      # (B, 128, H/2, W/2)
        x3 = self.down2(x2)      # (B, 256, H/4, W/4)
        x4 = self.down3(x3)      # (B, 512, H/8, W/8)
        x5 = self.down4(x4)      # (B, 512, H/16,W/16)  bottleneck

        # 解码器路径（逐层上采样 + skip connection）
        x = self.up1(x5, x4)     # (B, 256, H/8, W/8)
        x = self.up2(x,  x3)     # (B, 128, H/4, W/4)
        x = self.up3(x,  x2)     # (B, 64,  H/2, W/2)
        x = self.up4(x,  x1)     # (B, 64,  H,   W)

        logits = self.out_conv(x) # (B, num_classes, H, W)
        return logits


if __name__ == '__main__':
    # 验证模型结构和参数量
    from thop import profile

    model = UNet(in_channels=3, num_classes=3)
    input_tensor = torch.randn(1, 3, 256, 256)

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")   # 期望 (1, 3, 256, 256)
