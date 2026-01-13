import pandas as pd

# 读取文件：第三列为 Region_Index，第二列为 Region_Name
roi_path = r"E:\SPM12\aal_for_SPM12\aal_for_SPM12\ROI_MNI_V4.txt"
roi_df = pd.read_csv(roi_path, sep=r"\s+", header=None, names=["Abbr", "Region_Name", "Region_Index"])

# 只保留需要的两列
roi_df = roi_df[["Region_Index", "Region_Name"]]

# 转换 Region_Index 为整数
roi_df["Region_Index"] = roi_df["Region_Index"].astype(int)

importance_df = pd.read_csv("smri_important/global_region_importance.csv")
importance_df.rename(columns={"Unnamed: 0": "Region_Index"}, inplace=True)

# 合并脑区名
merged_df = pd.merge(importance_df, roi_df, on="Region_Index", how="left")

# 保存结果
merged_df.to_csv("global_region_importance_named.csv", index=False)
print("✅ 已保存带脑区名称的重要性结果！")
