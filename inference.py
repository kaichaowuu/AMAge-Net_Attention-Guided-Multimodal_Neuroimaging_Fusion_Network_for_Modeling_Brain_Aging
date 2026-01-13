import os
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from model import FusionModel
from dataset import MultimodalBrainDataset
from utils import *
from config import *
from tqdm import tqdm

def run_inference_and_save():
    # 加载路径和标签字典
    struct_paths = glob.glob(os.path.join(DATA_DIR_CAMCAN, '*', 'T1.nii.gz'))
    subject_age_dict, scaler = load_age_info(CSV_PATH_CAMCAN)
    all_paths = combine_struct_fmri_paths(struct_paths)

    # K折划分
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []  # 最终结果

    for fold, (_, val_idx) in enumerate(kf.split(all_paths)):
        print(f"\n📂 Inference on Fold {fold + 1}")
        val_paths = [all_paths[i] for i in val_idx]
        val_set = MultimodalBrainDataset(val_paths, subject_age_dict)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        # 加载模型
        model = FusionModel().to(device)
        ckpt_path = f"/home/zhuowan/code/Age_prediction/checkpoints/cross/sf_best_model_fold{fold + 1}.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        preds, labels, subject_ids = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = batch.to(device)
                y = batch.y
                out = model(batch)

                # 收集数据
                preds.extend(out.cpu().numpy().flatten())
                labels.extend(y.cpu().numpy().flatten())
                subject_ids.extend([str(sid) for sid in batch.subject_id])

        # 反归一化
        preds = denormalize_ages(scaler, np.array(preds))
        labels = denormalize_ages(scaler, np.array(labels))

        # 保存本折结果
        df_fold = pd.DataFrame({
            'subject_id': subject_ids,
            'actual_age': labels,
            'predicted_age': preds,
            'fold': fold + 1
        })
        # df_fold.to_csv(f"func_gate/dfold{fold + 1}_results.csv", index=False)
        # print(f"✅ Fold {fold + 1} prediction saved!")

        all_results.append(df_fold)

    # 合并所有折的结果并保存
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv("/home/zhuowan/code/Age_prediction/func_gate/cross_results.csv", index=False)
    print("🎉 All folds prediction results saved to func_gate/our model_results.csv")

if __name__ == '__main__':
    os.makedirs("func_gate", exist_ok=True)
    run_inference_and_save()
