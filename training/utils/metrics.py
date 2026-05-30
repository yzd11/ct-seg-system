import numpy as np
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure


def calculate_metrics(preds, targets, n_classes=3):
    """
    计算 Dice 和 IoU，跳过背景（class 0）

    参数：
      preds   : (B, H, W) long tensor，argmax 后的预测类别图
      targets : (B, H, W) long tensor，真实标签
      n_classes: 类别总数（含背景），默认 3

    返回：
      dice_scores : [Dice_liver, Dice_tumor]，当前 batch 的均值
      iou_scores  : [IoU_liver,  IoU_tumor]，当前 batch 的均值

    说明：
      当某张切片该类别既无预测也无真值（全为背景）时，
      视为"正确预测了空"，Dice 和 IoU 均记为 1.0
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dice_scores = []
    iou_scores = []

    for cls in range(1, n_classes):  # 忽略背景
        num = targets.shape[0]
        dice_sum = torch.tensor([0.0]).to(device)
        iou_sum = torch.tensor([0.0]).to(device)
        for i in range(num):
            pred = preds[i]
            target = targets[i]
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = torch.logical_and(pred_cls, target_cls).sum().float()
            union = torch.logical_or(pred_cls, target_cls).sum().float()

            if intersection + union > 0:
                dice = (2. * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-6)
                iou = intersection / (union + 1e-6)
            else:
                dice = torch.tensor([1.0]).to(device)
                iou = torch.tensor([1.0]).to(device)
            dice_sum += dice
            iou_sum += iou

        dice_scores.append((dice_sum / num).item())
        iou_scores.append((iou_sum / num).item())

    return dice_scores, iou_scores


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """
    提取二值掩码的表面（边界）点坐标

    原理：用结构元素做腐蚀，原掩码减去腐蚀结果即为边界；
    返回所有边界像素的 (row, col) 坐标数组，shape (N, 2)

    参数：
      mask : (H, W) bool 数组

    返回：
      coords : (N, 2) float64 数组，每行为一个边界点坐标
               若掩码为空返回 None
    """
    if not mask.any():
        return None
    struct = generate_binary_structure(mask.ndim, 1)   # 4-连通结构元素
    eroded = binary_erosion(mask, structure=struct, border_value=1)
    surface = mask & ~eroded
    return np.argwhere(surface).astype(np.float64)


def calculate_hd95_single(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算单张图单个类别的 HD95（95% Hausdorff Distance）

    HD95 定义：
      d(A→B) = 对 A 中每个点，求到 B 中最近点的距离，取所有距离的第 95 百分位
      HD95   = max(d(A→B), d(B→A))  的 95 百分位版本
      （与标准 HD 的区别：标准取 max，HD95 取 95 百分位，对离群边界点更鲁棒）

    参数：
      pred   : (H, W) bool 数组，预测二值掩码
      target : (H, W) bool 数组，真实二值掩码

    返回：
      HD95 值（像素单位）；若预测或真值任一为空，返回 None（调用方跳过）
    """
    pred_pts   = _surface_points(pred)
    target_pts = _surface_points(target)

    # 任一为空则无法计算，返回 None 由调用方跳过
    if pred_pts is None or target_pts is None:
        return None

    # 计算双向距离：对每个点求到另一集合的最小欧氏距离
    # 使用广播：(N,1,2) - (1,M,2) → (N,M,2) → norm → (N,M) → min(axis=1) → (N,)
    def min_distances(A, B):
        diff = A[:, None, :] - B[None, :, :]          # (N, M, 2)
        dists = np.sqrt((diff ** 2).sum(axis=2))       # (N, M)
        return dists.min(axis=1)                        # (N,)

    d_pred2target = min_distances(pred_pts,   target_pts)
    d_target2pred = min_distances(target_pts, pred_pts)

    all_distances = np.concatenate([d_pred2target, d_target2pred])
    return float(np.percentile(all_distances, 95))


if __name__ == '__main__':
    pred = torch.rand(2, 3, 4, 4)
    pred = torch.argmax(pred, dim=1)
    target = torch.tensor(
        [[[1, 2, 0, 0],
          [0, 0, 1, 1],
          [0, 2, 0, 0],
          [2, 1, 0, 0]],
         [[1, 2, 0, 0],
          [0, 0, 1, 1],
          [0, 2, 0, 0],
          [2, 1, 0, 0]]]
    )
    dice, iou = calculate_metrics(pred, target)
    print('Dice:', dice)
    print('IoU: ', iou)