# models/unet_pp.py
# U-Net++ 实现（含深度监督）
#
# 核心改动（相对于标准 U-Net）：
#   在编码器和解码器之间插入嵌套密集节点 X[i][j]，
#   每个节点聚合同层所有前序节点输出 + 下层上采样特征，丰富跨尺度特征融合路径。
#
# 深度监督：
#   训练时对 X[0][1]~X[0][4] 各接一个输出头，返回 list[logits×4]，
#   train.py 对四路 loss 取均值，梯度更直接地回传到浅层节点；
#   推理时（model.eval()）仅返回最终 X[0][4] 的 logits，test.py 无需改动。
#
# 节点命名：X[i][j]，i=深度（0=最浅/最高分辨率），j=密集列索引（0=编码器原始节点）
#
# 参考论文：Zhou et al., "UNet++: A Nested U-Net Architecture for Medical
#           Image Segmentation", DLMIA 2018.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import DoubleConv, Down, OutConv   # 直接复用 U-Net 基础模块


def _upsample_like(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    双线性上采样 src 至 target 的空间尺寸（处理奇数尺寸边界）
    src   : 待上采样特征，shape (B, C, H_s, W_s)
    target: 目标尺寸参考，shape (B, C', H_t, W_t)
    """
    src = F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)
    return src


class UNetPP(nn.Module):
    """
    U-Net++（嵌套密集 U-Net，含深度监督）

    通道配置（base_c=64, bilinear=True）：
      nb_filter = [64, 128, 256, 512, 512]

    密集节点输入通道公式：
      X[i][j] 的 in_ch = j × nb_filter[i] + nb_filter[i+1]

    展开后所有节点：
      X[3][1]: 1×512+512=1024 → 512
      X[2][1]: 1×256+512=768  → 256
      X[2][2]: 2×256+512=1024 → 256
      X[1][1]: 1×128+256=384  → 128
      X[1][2]: 2×128+256=512  → 128
      X[1][3]: 3×128+256=640  → 128
      X[0][1]: 1×64 +128=192  → 64
      X[0][2]: 2×64 +128=256  → 64
      X[0][3]: 3×64 +128=320  → 64
      X[0][4]: 4×64 +128=384  → 64  ← 最终输出

    深度监督输出头：out1~out4 分别对应 X[0][1]~X[0][4]
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNetPP, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        nb = [base_c, base_c*2, base_c*4, base_c*8, base_c*8]
        # nb = [64, 128, 256, 512, 512]（bilinear 模式下 bottleneck 不扩张）

        # ── 编码器（与 U-Net 完全相同）────────────────────────────────────────
        self.conv0_0 = DoubleConv(in_channels, nb[0])          # X[0][0]
        self.down1   = Down(nb[0], nb[1])                      # X[1][0]
        self.down2   = Down(nb[1], nb[2])                      # X[2][0]
        self.down3   = Down(nb[2], nb[3])                      # X[3][0]
        self.down4   = Down(nb[3], nb[4])                      # X[4][0] bottleneck

        # ── 密集节点（level 3，1个）────────────────────────────────────────────
        self.conv3_1 = DoubleConv(1*nb[3] + nb[4], nb[3])     # X[3][1]: 1024→512

        # ── 密集节点（level 2，2个）────────────────────────────────────────────
        self.conv2_1 = DoubleConv(1*nb[2] + nb[3], nb[2])     # X[2][1]: 768→256
        self.conv2_2 = DoubleConv(2*nb[2] + nb[3], nb[2])     # X[2][2]: 1024→256

        # ── 密集节点（level 1，3个）────────────────────────────────────────────
        self.conv1_1 = DoubleConv(1*nb[1] + nb[2], nb[1])     # X[1][1]: 384→128
        self.conv1_2 = DoubleConv(2*nb[1] + nb[2], nb[1])     # X[1][2]: 512→128
        self.conv1_3 = DoubleConv(3*nb[1] + nb[2], nb[1])     # X[1][3]: 640→128

        # ── 密集节点（level 0，4个）────────────────────────────────────────────
        self.conv0_1 = DoubleConv(1*nb[0] + nb[1], nb[0])     # X[0][1]: 192→64
        self.conv0_2 = DoubleConv(2*nb[0] + nb[1], nb[0])     # X[0][2]: 256→64
        self.conv0_3 = DoubleConv(3*nb[0] + nb[1], nb[0])     # X[0][3]: 320→64
        self.conv0_4 = DoubleConv(4*nb[0] + nb[1], nb[0])     # X[0][4]: 384→64

        # ── 深度监督输出头（训练时使用，推理时只用 out4）──────────────────────
        self.out1 = OutConv(nb[0], num_classes)   # 对应 X[0][1]
        self.out2 = OutConv(nb[0], num_classes)   # 对应 X[0][2]
        self.out3 = OutConv(nb[0], num_classes)   # 对应 X[0][3]
        self.out4 = OutConv(nb[0], num_classes)   # 对应 X[0][4]（最终输出）

    def forward(self, x: torch.Tensor):
        """
        训练时（self.training=True）：返回 list[logits×4]，供 train.py 计算多路 loss
        推理时（self.training=False）：返回单个 logits tensor，接口与其他模型一致
        输入：(B, in_channels, H, W)
        输出：(B, num_classes, H, W) 或 [(B, num_classes, H, W) × 4]
        """
        # 编码器路径
        x0_0 = self.conv0_0(x)         # (B, 64,  H,    W)
        x1_0 = self.down1(x0_0)        # (B, 128, H/2,  W/2)
        x2_0 = self.down2(x1_0)        # (B, 256, H/4,  W/4)
        x3_0 = self.down3(x2_0)        # (B, 512, H/8,  W/8)
        x4_0 = self.down4(x3_0)        # (B, 512, H/16, W/16)

        # 密集节点：level 3
        x3_1 = self.conv3_1(torch.cat([x3_0, _upsample_like(x4_0, x3_0)], dim=1))

        # 密集节点：level 2
        x2_1 = self.conv2_1(torch.cat([x2_0, _upsample_like(x3_0, x2_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, _upsample_like(x3_1, x2_0)], dim=1))

        # 密集节点：level 1
        x1_1 = self.conv1_1(torch.cat([x1_0, _upsample_like(x2_0, x1_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, _upsample_like(x2_1, x1_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, _upsample_like(x2_2, x1_0)], dim=1))

        # 密集节点：level 0（四个，对应深度监督的四路输出）
        x0_1 = self.conv0_1(torch.cat([x0_0, _upsample_like(x1_0, x0_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, _upsample_like(x1_1, x0_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, _upsample_like(x1_2, x0_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, _upsample_like(x1_3, x0_0)], dim=1))

        # 深度监督：训练时返回四路 logits，推理时只返回最终输出
        if self.training:
            return [self.out1(x0_1), self.out2(x0_2), self.out3(x0_3), self.out4(x0_4)]
        else:
            return self.out4(x0_4)


if __name__ == '__main__':
    from thop import profile

    model = UNetPP(in_channels=3, num_classes=3)
    input_tensor = torch.randn(1, 3, 256, 256)

    # 推理模式验证输出尺寸和参数量
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Output shape : {output.shape}")   # 期望 (1, 3, 256, 256)

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs        : {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters   : {params / 1e6:.2f} M")

    # 训练模式验证深度监督输出
    model.train()
    outputs = model(input_tensor)
    print(f"Deep supervision outputs: {len(outputs)} heads")
    for i, o in enumerate(outputs):
        print(f"  out{i+1}: {o.shape}")
