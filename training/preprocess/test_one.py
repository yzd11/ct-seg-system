"""测试单个 NIfTI 文件的读取和切片"""
import os, cv2, nibabel, numpy as np

f = 'volume-0.nii'
img_path = os.path.join(r'H:\volumes', f)
lbl_path = os.path.join(r'H:\segmentations', f.replace('volume', 'segmentation'))
print(f'Image: {img_path}  exists={os.path.exists(img_path)}')
print(f'Label: {lbl_path}  exists={os.path.exists(lbl_path)}')

img = nibabel.load(img_path)
lbl = nibabel.load(lbl_path)
img_data = img.get_fdata()
lbl_data = lbl.get_fdata()
print(f'Image shape: {img_data.shape}')
print(f'Label shape: {lbl_data.shape}')

# CT 调窗
lo, hi = 50 - 200, 50 + 200
sl = np.clip(img_data[:, :, 0], lo, hi)
sl = ((sl - lo) / (hi - lo) * 255).astype(np.uint8)

out_dir = os.path.join(r'H:\LiTS', 'volume-0', 'Image')
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, '0.png'), sl)
print(f'Test PNG written to {out_dir}')
print(f'File exists: {os.path.exists(os.path.join(out_dir, "0.png"))}')
