# utils/losses.py
# 损失函数定义
#
# 本项目使用 CE + Dice 的组合损失（各占 50%），并支持类别加权以缓解肿瘤类别不平衡问题。
#
# 背景：LiTS2017 数据集中肿瘤（类别2）像素占比极小（通常 < 1%），
#       若不加权，模型会倾向于预测背景/肝脏，导致肿瘤 Dice 极低。
#
# 加权策略：
#   - CrossEntropyLoss(weight=...)：PyTorch 原生支持，直接传入权重向量
#   - DiceLoss：对各类别的 Dice 值做加权平均（而非等权均值）
#   推荐权重：[1.0, 1.0, 3.0]（背景:肝脏:肿瘤），可通过 --tumor_weight 参数调整

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    多类别 Dice Loss（基于 softmax 概率）

    公式（对每个类别 c）：
        Dice_c = (2 * sum(p_c * y_c) + smooth) / (sum(p_c) + sum(y_c) + smooth)
        Loss   = 1 - weighted_mean(Dice_c)

    参数：
      smooth       : 平滑项，防止分母为零（默认 1e-5）
      class_weights: 各类别的损失权重，shape (num_classes,)，dtype float
                     None 时退化为等权均值
                     建议传入 torch.tensor([1.0, 1.0, tumor_weight])
    """
    def __init__(self, smooth=1e-5, class_weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        # 将权重注册为 buffer，使其随模型一起 .to(device)，但不参与梯度计算
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, pred, target):
        """
        pred  : (B, C, H, W)  网络输出的 logits（未归一化）
        target: (B, H, W)     真实标签，值为类别索引 [0, C-1]
        """
        num_classes = pred.shape[1]

        # one-hot 编码 target：(B, H, W) → (B, C, H, W)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # softmax 将 logits 转为概率
        pred_softmax = F.softmax(pred, dim=1)

        # 对空间维度（H, W）求和，得到 (B, C)
        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        # 各类别的 Dice 值，shape (B, C)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.class_weights is not None:
            # 加权均值：先对 batch 维度取均值，再对类别维度做加权
            # dice.mean(dim=0) shape: (C,)，class_weights shape: (C,)
            dice_per_class = dice.mean(dim=0)                              # (C,)
            weighted_dice = (dice_per_class * self.class_weights).sum()    # scalar
            return 1. - weighted_dice / self.class_weights.sum()
        else:
            return 1. - dice.mean()


class CombinedLoss(nn.Module):
    """
    组合损失：alpha * CrossEntropyLoss + (1-alpha) * DiceLoss

    两种损失的互补作用：
      - CrossEntropyLoss：逐像素分类损失，对多数类（背景）响应强，收敛快
      - DiceLoss        ：基于面积重叠的损失，对少数类（肿瘤）更敏感，解决不平衡

    参数：
      alpha        : CE 损失的权重（默认 0.5，即各占 50%）
      class_weights: 类别权重向量，同时传给 CE 和 Dice
                     形如 torch.tensor([1.0, 1.0, 3.0])
                     None 时两者均不加权

    使用示例：
      # 不加权（纯基线）
      criterion = CombinedLoss()

      # 肿瘤加权 3 倍
      weights = torch.tensor([1.0, 1.0, 3.0])
      criterion = CombinedLoss(class_weights=weights)
    """
    def __init__(self, alpha=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

        # CrossEntropyLoss 的 weight 参数需要是 1D tensor，shape (num_classes,)
        # 注意：CE 中 weight 会自动归一化，不需要手动除以 sum
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        self.dice = DiceLoss(class_weights=class_weights)

    def forward(self, pred, target):
        """
        pred  : (B, C, H, W)  网络输出 logits
        target: (B, H, W)     真实标签索引
        """
        ce_loss   = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


if __name__ == '__main__':
    # 功能验证
    B, C, H, W = 2, 3, 4, 4
    pred   = torch.rand(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))

    # 无权重
    criterion_base = CombinedLoss()
    loss_base = criterion_base(pred, target)
    print(f"无权重 Loss: {loss_base.item():.4f}")

    # 肿瘤加权 3 倍
    weights = torch.tensor([1.0, 1.0, 3.0])
    criterion_weighted = CombinedLoss(class_weights=weights)
    loss_weighted = criterion_weighted(pred, target)
    print(f"加权 Loss   : {loss_weighted.item():.4f}")
