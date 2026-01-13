# 计算平均脑区重要性
# import pandas as pd
#
# # 读入CSV
# df = pd.read_csv("smri_oasis/structural_region_importance.csv")
#
# # 取前90个脑区列（排除subject_id）
# brain_region_cols = df.columns[1:91]
#
# # 取这90列对应的数据
# df_90 = df[brain_region_cols]
#
# # 计算这90个脑区的平均重要性
# mean_importance_90 = df_90.mean(axis=0)
#
# # 转为DataFrame
# mean_df_90 = mean_importance_90.reset_index()
# mean_df_90.columns = ['Brain_Region', 'Mean_Importance']
#
# # ❌ 不进行排序
# # mean_df_90 = mean_df_90.sort_values(by='Mean_Importance', ascending=False)
#
# # 输出查看
# print(mean_df_90)
#
# # 保存结果
# mean_df_90.to_csv("smri_oasis/mean_structural_region_importance.csv", index=False)
#
# print("✅ 已保存未排序的平均脑区重要性到 smri_male/mean_structural_region_importance.csv")

# 添加脑区编号
# import pandas as pd
#
# # 路径配置
# structural_path = r"smri_oasis/mean_structural_region_importance.csv"
# reference_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func/mean_importance_per_region_with_names.csv"
# output_path = r"smri_oasis/mean_structural_region_importance_with_index.csv"
#
# # 读取两个CSV文件
# df_structural = pd.read_csv(structural_path)
# df_reference = pd.read_csv(reference_path)
#
# # 建立映射关系：Region_name → Region（编号）
# name_to_index = dict(zip(df_reference["Region_name"], df_reference["Region"]))
#
# # 添加新列 Region（编号），放在第一列
# df_structural.insert(0, "Region", df_structural["Brain_Region"].map(name_to_index))
#
# # 保存新文件
# df_structural.to_csv(output_path, index=False)
#
# print("✅ 脑区编号已添加并保存至：", output_path)


# 归一化
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# # 输入输出路
input_path = r"smri_oasis/mean_structural_region_importance_with_index.csv"
output_path = r"smri_oasis/mean_structural_region_importance_normalized_sorted.csv"

# 读取数据
df = pd.read_csv(input_path)

# 归一化 Mean_Importance 列
scaler = MinMaxScaler()
df['Mean_Importance_Normalized'] = scaler.fit_transform(df[['Mean_Importance']])

# 按归一化结果降序排序
df_sorted = df.sort_values(by='Mean_Importance_Normalized', ascending=False)

# 保存
df_sorted.to_csv(output_path, index=False)

print("✅ 归一化并降序排序后的结果已保存至：", output_path)
