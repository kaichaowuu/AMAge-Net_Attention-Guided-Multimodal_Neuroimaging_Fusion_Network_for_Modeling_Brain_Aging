import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom

# === 路径配置 ===
saliency_dir = "saliency_map_oasis"  # 你保存 .npy 显著性图的路径
aal_path = "/mnt/ugreen/E/SPM_AAL/SPM12/aal_for_SPM12/aal_for_SPM12/atlas/AAL.nii"  # AAL模板路径
aal_labels_txt = "/mnt/ugreen/E/SPM_AAL/SPM12/aal_for_SPM12/aal_for_SPM12/ROI_MNI_V4.txt"  # 你的AAL标签文件，包含编号和脑区名
output_csv = "smri_oasis/structural_region_importance.csv"  # 输出文件路径

# === 加载 AAL 模板及标签 ===
aal_nii = nib.load(aal_path)
aal_data = aal_nii.get_fdata().astype(int)
ref_affine = aal_nii.affine  # 用 AAL 的 affine

# 读取 AAL 标签编号与名称
label_df = pd.read_csv(aal_labels_txt, sep="\t", header=None, names=["Abbr", "Name", "Index"])
region_indices = label_df["Index"].values
region_names = label_df["Name"].values

# === 遍历所有显著性图文件并计算每个脑区的平均显著性 ===
results = []

for filename in tqdm(os.listdir(saliency_dir)):
    if filename.endswith(".npy"):
        sal_path = os.path.join(saliency_dir, filename)
        sal_data = np.load(sal_path)  # [D, H, W]

        # 判断是否需要重采样
        if sal_data.shape != aal_data.shape:
            print(f"WARNING: {filename} shape {sal_data.shape} 不匹配AAL模板形状 {aal_data.shape}，进行重采样")
            zoom_factors = [a / b for a, b in zip(aal_data.shape, sal_data.shape)]
            sal_data = zoom(sal_data, zoom_factors, order=1)  # 线性插值重采样

        # 计算每个脑区的重要性平均值
        subject_id = filename.replace(".npy", "")
        region_values = []
        for region_id in region_indices:
            mask = (aal_data == region_id)
            if np.sum(mask) == 0:
                region_values.append(np.nan)
            else:
                region_values.append(sal_data[mask].mean())
        results.append([subject_id] + region_values)

# === 保存为 CSV ===
columns = ["subject_id"] + list(region_names)
df = pd.DataFrame(results, columns=columns)
df.to_csv(output_csv, index=False)
print(f"✅ 成功保存至 {output_csv}")
