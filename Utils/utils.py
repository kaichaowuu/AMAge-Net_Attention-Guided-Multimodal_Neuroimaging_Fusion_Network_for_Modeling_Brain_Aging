import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from config import *
from sklearn.metrics import mean_absolute_error

def load_age_info(csv_path):
    # df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, encoding='gbk')
    df['Subject'] = df['Subject'].astype(str).str.strip()  # 👈 添加这一行
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Age_norm'] = scaler.fit_transform(df[['Age']])
    subject_age_dict = dict(zip(df['Subject'], df['Age_norm']))
    return subject_age_dict,scaler


def normalize_ages(scaler, ages):
    return scaler.transform(ages.reshape(-1, 1)).flatten()

def denormalize_ages(scaler, normalized_ages):
    return scaler.inverse_transform(normalized_ages.reshape(-1, 1)).flatten()
# 新增辅助函数，提取subject_id
def extract_subject_id(path):
    return path.split(os.sep)[-2]

# 新增：根据subject_id返回对应的fMRI路径
def get_fmri_path(subject_id):
    return os.path.join(FMRI_ROOT_OASIS3, subject_id, 'connectome.csv')


# 根据结构路径列表，合并成 (struct_path, fmri_path) 列表
def combine_struct_fmri_paths(struct_paths):
    combined = []
    for sp in struct_paths:
        sid = extract_subject_id(sp)
        fmri_path = get_fmri_path(sid)
        if os.path.exists(fmri_path):
            combined.append((sp, fmri_path))
        else:
            print(f"⚠️ Warning: fMRI path not found for subject {sid}, skip.")
    return combined



# 绘制损失曲线
def plot_loss_curve(train_losses, val_losses, fold_idx, save_dir='/home/zhuowan/code/Age_prediction/loss_plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title(f'Fold {fold_idx + 1} Loss Curve')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'Important_our_oasis_{fold_idx}_loss_curve.png'))
    plt.close()

# 绘制散点图
def plot_scatter(true_ages, pred_ages, fold_idx, r=None, phase="val"):
    save_dir = "/home/zhuowan/code/Age_prediction/figures" if phase == "val" else "test_figures"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))

    # 浅蓝色点 + 高透明度
    plt.scatter(true_ages, pred_ages, alpha=0.5, c='#87CEFA', label='Predictions')

    # y = x 拟合线，颜色柔和的蓝灰色
    plt.plot([min(true_ages), max(true_ages)],
             [min(true_ages), max(true_ages)],
             color='#4B6C8B', linestyle='--', linewidth=2, label='Ideal Fit (y=x)')

    # 计算 MAE（Mean Absolute Error）
    mae = mean_absolute_error(true_ages, pred_ages)

    # 标注 r 和 MAE
    if r is not None:
        textstr = f"r = {r:.2f}\nMAE = {mae:.2f}"
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    # plt.title(f'Fold {fold_idx + 1} - Predicted vs. True Age')
    plt.title(f'Predicted vs True Age')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'Important_our_oasis_{fold_idx}_scatter.png')
    # save_path = os.path.join(save_dir, f'female2_{fold_idx}_scatter.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存散点图到: {save_path}")


