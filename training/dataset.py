# dataset.py
# 自定义数据集类，读取预处理后的 2D PNG 切片
#
# 数据来源：preprocess/split_nii.py 将 NIfTI 文件按 z 轴切片并保存为 PNG
# 目录结构（data_root 下）：
#   {data_root}/{case_id}/Image/xxx.png   —— CT 图像（uint8 灰度，已做窗宽/窗位调整）
#   {data_root}/{case_id}/GT/xxx.png      —— 标签（uint8，0=背景,128=肝脏,255=肿瘤）
#
# 标签映射：PNG 存储值 → 训练用类别索引
#   0   → 0（背景）
#   128 → 1（肝脏）
#   255 → 2（肿瘤）
#
# 数据增强注意事项：
#   图像和标签必须施加完全相同的空间变换，否则像素与标签不对齐。
#   做法：对同一 idx 固定随机种子，先变换图像再变换标签，保证两者一致。

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

# 标签值 → 类别索引的映射（PNG 像素值 → 网络训练用）
label_mapping = {
    0:   0,   # 背景
    128: 1,   # 肝脏
    255: 2,   # 肿瘤
}

# 类别索引 → 标签值的逆映射（用于可视化预测结果）
inverse_mapping = {v: k for k, v in label_mapping.items()}


class CustomImageDataset(Dataset):
    """
    LiTS2017 2D 切片数据集

    参数：
      data_type : 'train' / 'valid' / 'test'，对应 preprocess/ 下的 txt 文件
      data_root : PNG 数据根目录，默认为本地路径。
                  !! 部署到 AutoDL 服务器时需修改此默认值，或通过 train.py 的 --data_root 参数传入 !!
                  本地路径示例  ：'F:\\LiTS\\LiTS'
                  服务器路径示例：'/root/autodl-tmp/LiTS'   （AutoDL 数据盘挂载点）
      transform : torchvision.transforms 变换，图像和标签会使用相同随机种子同步变换
    """
    def __init__(self, data_type, data_root='F:\\LiTS\\LiTS', transform=None,
                 txt_path=None, paired_transform=False):
        self.img_dir = data_root
        self.data_type = data_type
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        self.paired_transform = paired_transform

        # 读取 txt 文件中的 case id 列表，构建完整路径
        # txt 文件格式：每行一个 case_id，例如 "volume-0"
        if txt_path is not None:
            _txt_path = txt_path
        else:
            _txt_path = os.path.join('./preprocess', self.data_type + '.txt')
        with open(_txt_path, 'r') as txt_file:
            for row in txt_file:
                case_id = row.strip()   # 去除换行符
                if not case_id:
                    continue            # 跳过空行

                image_case = os.path.join(self.img_dir, case_id, 'Image')
                label_case = os.path.join(self.img_dir, case_id, 'GT')

                # 遍历该 case 下所有切片（文件名相同，只是所在目录不同）
                for case in sorted(f for f in os.listdir(image_case) if f.endswith('.png')):
                    self.image_paths.append(os.path.join(image_case, case))
                    self.label_paths.append(os.path.join(label_case, case))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path   = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 图像：转为 RGB（3 通道，复制灰度值到 R/G/B 三个通道）
        # 注意：CT 灰度图做 convert("RGB") 后 in_channels=3，参数量更多但兼容预训练特征
        # 若要改为单通道，将此处改为 convert("L")，并同步修改 UNet(in_channels=1)
        image = Image.open(img_path).convert("RGB")

        # 标签：保持单通道灰度（像素值 0/128/255）
        label = Image.open(label_path).convert("L")

        if self.transform:
            if self.paired_transform:
                # 新版AugmentationPipeline：同时处理图像和标签
                image, label = self.transform(image, label)
            else:
                # 旧版torchvision transforms：分别处理但使用相同种子
                # seed 同时包含固定基准(3407)和样本索引(idx)，保证：
                #   - 不同 idx 的增强不相同（有随机性）
                #   - 相同 idx 每次 epoch 得到相同的增强（严格可复现）
                seed = 3407 + idx

                torch.manual_seed(seed)
                image = self.transform(image)

                torch.manual_seed(seed)    # 重置为相同种子，保证与 image 变换完全一致
                label = self.transform(label)

        # 图像转为 float tensor，shape (C, H, W)，值域 [0, 1]
        image = transforms.ToTensor()(image)

        # 标签处理：PIL → numpy → 映射像素值 → long tensor
        label = np.array(label)                              # shape (H, W)，值 0/128/255
        label = np.vectorize(label_mapping.get)(label)      # 0/128/255 → 0/1/2
        label = torch.from_numpy(label).long()              # shape (H, W)，值 0/1/2

        return image, label


def set_seed(seed):
    """全局随机种子设置（用于 __main__ 测试块）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 数据集可视化验证：随机抽取一批次，并排展示原图、标签
    seed = 3407
    set_seed(seed)
    print(f'Random seed: {seed}')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90, interpolation=Image.NEAREST),
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    # !! 本地测试时使用默认路径，服务器上请传入正确的 data_root !!
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    valid_dataset = CustomImageDataset(data_type='valid', transform=transform)
    print(f'Train size: {len(train_dataset)}')
    print(f'Valid size: {len(valid_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 可视化：将标签的类别索引反映射回 0/128/255 方便观察
    images = (images * 255).to(torch.uint8)
    labels_vis = torch.from_numpy(np.vectorize(inverse_mapping.get)(labels.numpy()))
    labels_vis = torch.stack([labels_vis, labels_vis, labels_vis], dim=1)

    grid_img   = make_grid(images)
    grid_label = make_grid(labels_vis)
    concat = torch.cat((grid_img, grid_label), dim=1)

    plt.figure(figsize=(16, 8))
    plt.imshow(concat.permute(1, 2, 0))
    plt.title('上排：CT图像  下排：分割标签（黑=背景，灰=肝脏，白=肿瘤）')
    plt.show()
