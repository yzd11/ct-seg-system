"""本地预处理 — 单线程版本，错误可见"""
import os, cv2, nibabel, numpy as np

INPUT_IMAGE_DIR = r'H:\volumes'
INPUT_LABEL_DIR = r'H:\segmentations'
OUTPUT_DIR      = r'H:\LiTS'

def window_image(image, window_width=400, window_level=50):
    lo = window_level - window_width // 2
    hi = window_level + window_width // 2
    image = np.clip(image, lo, hi)
    return ((image - lo) / (hi - lo) * 255).astype(np.uint8)

def process_one(image_file):
    label_file = image_file.replace('volume', 'segmentation')
    image_path = os.path.join(INPUT_IMAGE_DIR, image_file)
    label_path = os.path.join(INPUT_LABEL_DIR, label_file)
    if not os.path.exists(label_path):
        return f"SKIP: {image_file} (no label)"

    img_nii = nibabel.load(image_path)
    lbl_nii = nibabel.load(label_path)
    img_data = np.asarray(img_nii.dataobj)  # 避免 float64 内存膨胀
    lbl_data = np.asarray(lbl_nii.dataobj)

    case_num = image_file.split('-')[1].split('.')[0]
    img_out = os.path.join(OUTPUT_DIR, f'volume-{case_num}', 'Image')
    lbl_out = os.path.join(OUTPUT_DIR, f'volume-{case_num}', 'GT')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    count = 0
    for i in range(img_data.shape[2]):
        lbl_slice = lbl_data[:, :, i]
        if not np.any(lbl_slice):
            continue
        img_slice = window_image(img_data[:, :, i])
        lbl_slice = lbl_slice.copy()
        lbl_slice[lbl_slice == 1] = 128
        lbl_slice[lbl_slice == 2] = 255
        cv2.imwrite(os.path.join(img_out, f'{i}.png'), img_slice)
        cv2.imwrite(os.path.join(lbl_out, f'{i}.png'), lbl_slice.astype(np.uint8))
        count += 1
    return f"volume-{case_num}: {count} slices"

if __name__ == '__main__':
    files = sorted(f for f in os.listdir(INPUT_IMAGE_DIR) if f.endswith('.nii') or f.endswith('.nii.gz'))
    print(f'Processing {len(files)} volumes...')
    ok, fail = 0, 0
    for f in files:
        try:
            result = process_one(f)
            print(f'  [{ok+fail+1}/{len(files)}] {result}')
            ok += 1
        except Exception as e:
            print(f'  [{ok+fail+1}/{len(files)}] FAIL {f}: {e}')
            fail += 1
    print(f'\nDone: {ok} OK, {fail} FAIL')
