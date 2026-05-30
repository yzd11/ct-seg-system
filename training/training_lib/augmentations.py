# training/augmentations.py
# 医学图像数据增强模块
#
# 增强等级定义：
#   none   : 仅 Resize（验证/测试用）
#   basic  : 随机翻转 + 90°旋转 + Resize（与原项目等价）
#   medium : basic + 亮度/对比度扰动 + Gamma校正
#   strong : medium + 弹性形变 + 随机缩放
#
# 所有空间变换使用 NEAREST 插值以保证标签像素值不被污染。

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ── 空间变换（已有）───────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    """随机水平翻转，p=0.5"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:
    """随机垂直翻转，p=0.5"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotation90:
    """随机90°倍数旋转"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        if random.random() < self.p:
            k = random.randint(0, 3)
            # 90°倍数旋转 + NEAREST 确保标签值的完整性
            return img.rotate(90 * k, resample=Image.NEAREST, expand=True)
        return img


class ResizeTo:
    """缩放到目标尺寸"""
    def __init__(self, size: int = 256):
        self.size = size

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        return img.resize((self.size, self.size), Image.NEAREST)


# ── 强度变换（仅应用于图像，不应用于标签）────────────────────────────────

class RandomBrightnessContrast:
    """随机亮度和对比度调整（仅对图像）"""
    def __init__(self, brightness_range: float = 0.15, contrast_range: float = 0.15, p: float = 0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        if is_label or random.random() >= self.p:
            return img

        brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomGamma:
    """随机Gamma校正（仅对图像）"""
    def __init__(self, gamma_range: tuple = (0.7, 1.3), p: float = 0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        if is_label or random.random() >= self.p:
            return img

        gamma = random.uniform(*self.gamma_range)
        # PIL方式：用ImageOps.autocontrast或转换为numpy处理
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma) * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


# ── 弹性形变（仅应用于图像+标签的同时空间变换）───────────────────────────

class ElasticDeformation:
    """
    弹性形变（Elastic Deformation）
    参考：Simard et al., "Best Practices for CNN Applied to Visual Document Analysis", ICDAR 2003

    对图像和标签施加相同的随机位移场，模拟组织的自然形变。
    """
    def __init__(self, alpha: float = 30.0, sigma: float = 4.0, p: float = 0.3):
        """
        alpha: 位移强度（越大形变越剧烈）
        sigma: 高斯平滑程度（越大形变越平滑）
        p: 应用概率
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image, is_label: bool = False) -> Image.Image:
        raise NotImplementedError(
            'ElasticDeformation requires paired (img, label) call. '
            'Use apply_paired() instead.'
        )

    def apply_paired(self, img: Image.Image, label: Image.Image):
        """同时对图像和标签施加弹性形变"""
        if random.random() >= self.p:
            return img, label

        img_np = np.array(img)
        label_np = np.array(label)
        h, w = img_np.shape[:2]

        # 生成随机位移场
        dx = np.random.randn(h, w).astype(np.float32) * self.alpha
        dy = np.random.randn(h, w).astype(np.float32) * self.alpha

        # 高斯平滑
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma=self.sigma)
        dy = gaussian_filter(dy, sigma=self.sigma)

        # 生成采样网格
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x_new = x + dx
        y_new = y + dy

        # 双线性插值采样图像
        from scipy.ndimage import map_coordinates
        if img_np.ndim == 3:
            deformed_img = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                deformed_img[:, :, c] = map_coordinates(
                    img_np[:, :, c], [y_new, x_new], order=1, mode='constant'
                )
        else:
            deformed_img = map_coordinates(
                img_np, [y_new, x_new], order=1, mode='constant'
            )

        # 最近邻插值采样标签（防止像素值污染）
        deformed_label = map_coordinates(
            label_np.astype(np.float32), [y_new, x_new],
            order=0, mode='constant'
        ).astype(label_np.dtype)

        return Image.fromarray(deformed_img.astype(np.uint8)), \
            Image.fromarray(deformed_label)


# ── 增强管道构建 ────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    数据增强管道
    将多个变换组合，对图像和标签同步应用空间变换，仅对图像应用强度变换
    """

    def __init__(self, level: str = 'basic', img_size: int = 256):
        """
        level: 'none' | 'basic' | 'medium' | 'strong'
        """
        self.level = level
        self.img_size = img_size

        # 空间变换（图像和标签都需要）
        self.spatial_transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(p=0.5),
            ResizeTo(size=img_size),
        ]

        # 强度变换（仅图像）
        if level in ('medium', 'strong'):
            self.intensity_transforms = [
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.3),
            ]
        else:
            self.intensity_transforms = []

        # 弹性形变（仅strong等级）
        if level == 'strong':
            self.elastic = ElasticDeformation(alpha=30.0, sigma=4.0, p=0.3)
        else:
            self.elastic = None

    def __call__(self, img: Image.Image, label: Image.Image):
        """
        返回 (augmented_image, augmented_label)
        """
        # 1. 弹性形变先做（因为之后尺寸会变化）
        if self.elastic is not None:
            img, label = self.elastic.apply_paired(img, label)

        # 2. 空间变换（图像和标签使用相同随机种子）
        for t in self.spatial_transforms:
            seed = random.randint(0, 2**32 - 1)

            # 对图像应用
            random.seed(seed)
            img = t(img, is_label=False)

            # 对标签应用（相同种子=相同变换）
            random.seed(seed)
            label = t(label, is_label=True)

        # 3. 强度变换（仅图像）
        for t in self.intensity_transforms:
            img = t(img, is_label=False)

        return img, label


# ── 快速测试 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 生成假CT和标签
    rng = np.random.RandomState(42)
    ct = rng.randint(0, 255, (256, 256), dtype=np.uint8)
    ct[80:160, 80:160] = rng.randint(100, 200, (80, 80), dtype=np.uint8)
    label = np.zeros((256, 256), dtype=np.uint8)
    label[70:150, 70:150] = 128
    label[90:120, 90:120] = 255

    img = Image.fromarray(ct)
    lbl = Image.fromarray(label)

    for level in ['basic', 'medium', 'strong']:
        pipeline = AugmentationPipeline(level=level, img_size=256)
        aug_img, aug_lbl = pipeline(img, lbl)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original CT')
        axes[1].imshow(lbl, cmap='gray'); axes[1].set_title('Original Label')
        axes[2].imshow(aug_img, cmap='gray'); axes[2].set_title(f'Augmented CT ({level})')
        axes[3].imshow(aug_lbl, cmap='gray'); axes[3].set_title(f'Augmented Label ({level})')
        for ax in axes:
            ax.axis('off')
        plt.suptitle(f'Augmentation Level: {level}')
        plt.tight_layout()
        plt.show()
