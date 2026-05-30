"""本地预处理脚本 — 将 NIfTI 转为 2D PNG 切片"""
import os, cv2, nibabel, numpy as np
from concurrent.futures import ThreadPoolExecutor

# === 修改这三行路径 ===
INPUT_IMAGE_DIR = r'H:\volumes'
INPUT_LABEL_DIR = r'H:\segmentations'
OUTPUT_DIR      = r'H:\LiTS'

def window_image(image, window_width=400, window_level=50):
    lo = window_level - window_width // 2
    hi = window_level + window_width // 2
    image = np.clip(image, lo, hi)
    image = ((image - lo) / (hi - lo) * 255).astype(np.uint8)
    return image

def process_image(image_file):
    label_file = image_file.replace('volume', 'segmentation')
    image_path = os.path.join(INPUT_IMAGE_DIR, image_file)
    label_path = os.path.join(INPUT_LABEL_DIR, label_file)
    if not os.path.exists(label_path):
        print(f"Skip {image_file}: label not found")
        return
    img_nii = nib.load(image_path)
    lbl_nii = nib.load(label_path)
    img_data = img_nii.get_fdata()
    lbl_data = lbl_nii.get_fdata()

    case_id = image_file.split('.')[0].split('-')[1]
    img_out = os.path.join(OUTPUT_DIR, f'volume-{case_id}', 'Image')
    lbl_out = os.path.join(OUTPUT_DIR, f'volume-{case_id}', 'GT')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for i in range(img_data.shape[2]):
        img_slice = window_image(img_data[:, :, i])
        lbl_slice = lbl_data[:, :, i].copy()
        if not np.any(lbl_slice):  # 跳过全背景切片
            continue
        lbl_slice[lbl_slice == 1] = 128
        lbl_slice[lbl_slice == 2] = 255
        cv2.imwrite(os.path.join(img_out, f'{i}.png'), img_slice)
        cv2.imwrite(os.path.join(lbl_out, f'{i}.png'), lbl_slice.astype(np.uint8))
    print(f'Done: volume-{case_id}')

if __name__ == '__main__':
    files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')]
    print(f'Found {len(files)} volumes')
    with ThreadPoolExecutor(max_workers=2) as ex:
        for f in files:
            ex.submit(process_image, f)
    ex.shutdown(wait=True)
    print('All done!')
