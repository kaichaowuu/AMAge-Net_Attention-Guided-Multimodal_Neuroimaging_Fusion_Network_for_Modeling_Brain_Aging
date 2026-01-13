# # 合并文件
# import pandas as pd
# import glob
# import os
#
# # 设置路径
# folder_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func_oasis"
# output_path = os.path.join(folder_path, "merged_output.csv")
#
# # 获取所有csv文件
# csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
#
# # 读取并合并
# df_list = []
# for file in csv_files:
#     df = pd.read_csv(file)
#
#     # 如果subject_id是字符串列表（如 ['CC110087']），就去除[]
#     if df['subject_id'].dtype == object:
#         df['subject_id'] = df['subject_id'].str.strip("[]'\"")  # 去除 [] 和引号
#
#     df_list.append(df)
#
# # 合并所有DataFrame
# df_merged = pd.concat(df_list, ignore_index=True)
#
# # 保存合并后的文件
# df_merged.to_csv(output_path, index=False)
#
# print(f"合并完成，总共合并了 {len(df_list)} 个文件，结果保存在：{output_path}")



# # 计算平均脑区的重要性
# import pandas as pd
#
# # 读取合并好的数据
# df = pd.read_csv("func_oasis/merged_output.csv")
#
# # 去除 subject_id 列
# region_importances = df.drop(columns=["subject_id"])
#
# # 计算每个脑区的平均重要性（按原始列顺序保留）
# mean_importance = region_importances.mean(axis=0)
#
# # 将结果保存为 CSV，不排序
# mean_importance.to_csv("func_oasis/global_region_importance.csv", header=["importance"])
#
# print("✅ 未排序的全局脑区重要性已保存为 func_male/global_region_importance.csv")


# 添加对应脑区的名称
import pandas as pd
#
# # 路径配置
# importance_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func_oasis/global_region_importance.csv"
# region_info_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func/mean_importance_per_region_with_names.csv"
# output_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func_oasis/global_region_importance_with_names.csv"
#
# # 读取重要性和脑区信息
# df_importance = pd.read_csv(importance_path, index_col=0)
# df_region_info = pd.read_csv(region_info_path)
#
# # 将 Region 设置为字符串，确保与 index 匹配
# df_region_info['Region'] = df_region_info['Region'].astype(str)
# df_importance.index = df_importance.index.astype(str)
#
# # 合并：通过 index 和 Region 匹配添加 Region_name
# df_importance = df_importance.reset_index().rename(columns={'index': 'Region'})
# df_merged = pd.merge(df_importance, df_region_info[['Region', 'Region_name']], on='Region', how='left')
#
# # 保存结果
# df_merged.to_csv(output_path, index=False)
#
# print("✅ 添加 Region_name 完成，结果保存至：", output_path)




import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# # 输入和输出路径
input_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func_oasis/global_region_importance_with_names.csv"
output_path = r"/home/zhuowan/code/Age_prediction/important_sf_weight/func_oasis/global_region_importance_normalized_sorted.csv"

# 读取数据
df = pd.read_csv(input_path)

# 归一化 importance 列（Min-Max）
scaler = MinMaxScaler()
df['importance_normalized'] = scaler.fit_transform(df[['importance']])

# 按归一化后的结果降序排列
df_sorted = df.sort_values(by='importance_normalized', ascending=False)

# 保存结果
df_sorted.to_csv(output_path, index=False)

print("✅ 已完成归一化并按降序排序，结果保存在：", output_path)
