# preprocess/prepare_nnunet.py
# 将 LiTS 原始 NIfTI 数据转换为 nnU-Net v2 所需格式，并生成 dataset.json
#
# 运行前提：
#   - 原始数据位于 /root/autodl-tmp/volumes/ 和 /root/autodl-tmp/segmentations/
#   - 本脚本需在 /root/UNet-LiTS2017-main/preprocess/ 目录下执行
#
# 输出目录结构（nnUNet_raw 下）：
#   Dataset001_LiTS/
#   ├── imagesTr/          # train + valid，共 104 个 case
#   │   ├── liver_000_0000.nii.gz
#   │   └── ...
#   ├── labelsTr/          # 对应标签
#   ├── imagesTs/          # test，共 27 个 case（用于预测，无标签）
#   └── dataset.json
#
# nnU-Net 文件命名规则（官方文档）：
#   图像：{CASE_IDENTIFIER}_{XXXX}.nii.gz，XXXX 为 4 位模态索引，CT 只有 _0000
#   标签：{CASE_IDENTIFIER}.nii.gz
#
# 运行命令：
#   cd /root/UNet-LiTS2017-main/preprocess
#   python prepare_nnunet.py

import gc
import gzip
import json
import os
import shutil

import nibabel as nib
import numpy as np

# ── 路径配置 ─────────────────────────────────────────────────────────────────
VOLUMES_DIR = '/root/autodl-tmp/volumes'
SEG_DIR     = '/root/autodl-tmp/segmentations'
NNUNET_RAW  = '/root/autodl-tmp/nnunet_raw'   # 对应环境变量 nnUNet_raw
DATASET_DIR = os.path.join(NNUNET_RAW, 'Dataset001_LiTS')

# train/valid/test 划分文件（与 2D 模型完全一致，保证测试集相同）
SPLIT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_ids(txt_path):
    """读取 txt 文件中的 case ID 列表"""
    with open(txt_path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def copy_as_nii_gz(src, dst):
    """
    将 .nii 文件流式压缩为 .nii.gz，不将体积数据加载到内存。

    使用 gzip + shutil.copyfileobj 逐块（1 MB）拷贝，
    峰值内存占用约 1 MB，与文件大小无关。
    """
    with open(src, 'rb') as f_in:
        with gzip.open(dst, 'wb', compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out, length=1 << 20)


def convert_seg(seg_src, vol_src, dst):
    """
    修复标签 header 并保存为 .nii.gz。

    LiTS 已知问题：segmentation-X.nii 的 spacing/origin/direction 均为
    identity 默认值，与对应 volume 的真实物理信息不一致。
    nnUNetv2_plan_and_preprocess --verify_dataset_integrity 会因此报错。
    解决方案：用 CT volume 的 affine/header 重建标签 NIfTI，只保留标签数组。

    内存优化：
      - nib.load 对 .nii 默认使用 memmap，访问 .affine/.header 不触发数据加载
      - np.asarray(seg_nii.dataobj) 直接读取原始整数数据，跳过 get_fdata() 的
        float64 中间态（512×512×1000 体积下 float64 约占 2 GB）
      - 每步处理完后显式 del + gc.collect() 确保内存立即释放
    """
    seg_nii = nib.load(seg_src)
    vol_nii = nib.load(vol_src)   # memmap，不加载体积数据

    # 直接读取原始整数数组，避免 get_fdata() 的 float64 转换
    seg_data = np.asarray(seg_nii.dataobj).astype(np.uint8)
    del seg_nii
    gc.collect()

    new_seg = nib.Nifti1Image(seg_data, affine=vol_nii.affine, header=vol_nii.header)
    new_seg.set_data_dtype(np.uint8)
    del vol_nii, seg_data
    gc.collect()

    nib.save(new_seg, dst)
    del new_seg
    gc.collect()


def main():
    train_ids = read_ids(os.path.join(SPLIT_DIR, 'train.txt'))
    valid_ids  = read_ids(os.path.join(SPLIT_DIR, 'valid.txt'))
    test_ids   = read_ids(os.path.join(SPLIT_DIR, 'test.txt'))

    train_val_ids = sorted(train_ids + valid_ids)   # 104 个 case → imagesTr

    print(f'Train+Valid : {len(train_val_ids)} cases → imagesTr / labelsTr')
    print(f'Test        : {len(test_ids)} cases → imagesTs')

    images_tr = os.path.join(DATASET_DIR, 'imagesTr')
    labels_tr = os.path.join(DATASET_DIR, 'labelsTr')
    images_ts = os.path.join(DATASET_DIR, 'imagesTs')
    for d in [images_tr, labels_tr, images_ts]:
        os.makedirs(d, exist_ok=True)

    # ── imagesTr + labelsTr（train + valid）───────────────────────────────────
    for case_id in train_val_ids:
        name    = f'liver_{case_id:03d}'
        vol_src = os.path.join(VOLUMES_DIR, f'volume-{case_id}.nii')
        seg_src = os.path.join(SEG_DIR,     f'segmentation-{case_id}.nii')

        copy_as_nii_gz(vol_src, os.path.join(images_tr, f'{name}_0000.nii.gz'))
        convert_seg(seg_src, vol_src,        os.path.join(labels_tr, f'{name}.nii.gz'))

        print(f'  [TrainVal] {case_id:3d} → {name}')

    # ── imagesTs（test，只需图像，标签留给 evaluate_nnunet.py）────────────────
    for case_id in sorted(test_ids):
        name    = f'liver_{case_id:03d}'
        vol_src = os.path.join(VOLUMES_DIR, f'volume-{case_id}.nii')

        copy_as_nii_gz(vol_src, os.path.join(images_ts, f'{name}_0000.nii.gz'))

        print(f'  [Test    ] {case_id:3d} → {name}')

    # ── dataset.json（官方规范字段）──────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "CT"},      # "CT" 触发 nnU-Net CT 专用归一化
        "labels": {
            "background": 0,
            "liver":      1,
            "tumor":      2
        },
        "numTraining": len(train_val_ids),
        "file_ending": ".nii.gz",
        "name":        "LiTS",
        "description": "LiTS2017 Liver Tumor Segmentation Challenge"
    }
    json_path = os.path.join(DATASET_DIR, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f'\ndataset.json → {json_path}')
    print('\n数据准备完成，接下来在服务器上执行：')
    print()
    print('  # 1. 设置环境变量（如已写入 ~/.bashrc 可跳过）')
    print('  export nnUNet_raw=/root/autodl-tmp/nnunet_raw')
    print('  export nnUNet_preprocessed=/root/autodl-tmp/nnunet_preprocessed')
    print('  export nnUNet_results=/root/autodl-tmp/nnunet_results')
    print()
    print('  # 2. 规划 + 预处理')
    print('  nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity')
    print()
    print('  # 3. 训练（fold 0）')
    print('  nohup nnUNetv2_train 001 3d_fullres 0 --npz > /root/nnunet_train.log 2>&1 &')
    print()
    print('  # 4. 预测')
    print('  nnUNetv2_predict \\')
    print('      -i $nnUNet_raw/Dataset001_LiTS/imagesTs \\')
    print('      -o /root/autodl-tmp/nnunet_predictions \\')
    print('      -d 001 -c 3d_fullres -f 0')
    print()
    print('  # 5. 评估')
    print('  python evaluate_nnunet.py \\')
    print('      --pred_dir /root/autodl-tmp/nnunet_predictions \\')
    print('      --seg_dir  /root/autodl-tmp/segmentations \\')
    print('      --test_txt preprocess/test.txt')


if __name__ == '__main__':
    main()
