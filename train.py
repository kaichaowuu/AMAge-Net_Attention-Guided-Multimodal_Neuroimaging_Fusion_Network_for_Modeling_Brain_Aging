import os
import glob
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from model import FusionModel
from dataset import MultimodalBrainDataset
from utils import *
from config import *
from copy import deepcopy


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y

        optimizer.zero_grad()
        preds = model(batch)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, scaler, region_labels=None, save_importance_csv=True, fold_idx=None):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    all_func_importances = []
    all_struct_saliency_maps = []
    subject_ids = []

    for batch in loader:
        batch = batch.to(device)

        # 👇 设置结构图像需要梯度
        # batch.img = batch.img.clone().detach().requires_grad_(True)
        batch.img.requires_grad_(True)
        batch.img.retain_grad()

        # 清零可能存在的旧梯度
        if batch.img.grad is not None:
            batch.img.grad = None

        y = batch.y
        out = model(batch)
        loss = criterion(out, y)
        total_loss += loss.item() * len(batch)

        # 👇 关键步骤：启用反向传播以便获得 saliency map
        # loss.backward(retain_graph=True)
        loss.backward()

        preds.extend(out.detach().cpu().numpy().flatten())
        labels.extend(y.detach().cpu().numpy().flatten())

        # 获取功能图的节点注意力（重要性）
        node_attention = model.gcn.get_node_importance()  # [total_nodes]

        for i in range(out.size(0)):
            node_mask = (batch.batch == i)
            importance = node_attention[node_mask].detach().cpu().numpy()  # [90]
            all_func_importances.append(importance)

            # 获取结构图像的显著性图（saliency map）
            if batch.img.grad is not None:
                saliency_map = batch.img.grad[i].detach().cpu().numpy()
                saliency_map = np.abs(saliency_map).squeeze(0)
            else:
                saliency_map = np.zeros(batch.img[i].shape[1:], dtype=np.float32)
            all_struct_saliency_maps.append(saliency_map)

            cur_subject_id = str(batch.subject_id[i])
            subject_ids.append(cur_subject_id)

    # 反归一化预测与标签
    preds = denormalize_ages(scaler, np.array(preds))
    labels = denormalize_ages(scaler, np.array(labels))

    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)
    pcc = pearsonr(labels, preds)[0]
    val_loss = total_loss / len(loader.dataset)

    all_func_importances = np.vstack(all_func_importances)

    # 保存重要性与显著性图
    if save_importance_csv and region_labels is not None and fold_idx is not None:
        df_func = pd.DataFrame(all_func_importances, columns=region_labels)
        df_func.insert(0, 'subject_id', subject_ids)
        df_func.to_csv(f'func_oasis/{fold_idx+1}.csv', index=False)
        print(f"Fold {fold_idx+1} 功能脑区重要性保存完成！")

        for sid, sal_map in zip(subject_ids, all_struct_saliency_maps):
            np.save(f'saliency_map_oasis/fold{fold_idx + 1}_{sid}.npy', sal_map)
        print(f"Fold {fold_idx+1} 结构显著性图保存完成！")

    return val_loss, mae, rmse, r2, pcc, labels, preds, all_func_importances


def run_train():
    # struct_paths = glob.glob(os.path.join(DATA_DIR_CAMCAN, '*', 'T1.nii.gz'))
    # subject_age_dict, scaler = load_age_info(CSV_PATH_CAMCAN)

    struct_paths = glob.glob(os.path.join(DATA_DIR_OASIS3, '*', 'brain.nii.gz'))  # 注意这里是fMRI路径
    subject_age_dict, scaler = load_age_info(CSV_PATH_OASIS3)

    all_paths = combine_struct_fmri_paths(struct_paths)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_labels = []
    all_preds = []
    fold_train_losses = []
    fold_val_losses = []
    all_metrics = {'mae': [], 'rmse': [], 'r2': [], 'pcc': []}

    # 用于保存每个fold预测结果
    fold_results_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_paths)):
        print(f"\n===== Fold {fold + 1} =====")
        train_paths = [all_paths[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]

        train_set = MultimodalBrainDataset(train_paths, subject_age_dict)
        val_set = MultimodalBrainDataset(val_paths, subject_age_dict)

        region_labels = val_set.get_region_labels()

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

        model = FusionModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)

        best_mae = float('inf')
        best_model_state = None
        best_epoch = 0

        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, mae, rmse, r2, pcc, _, _, _ = evaluate(
                model, val_loader, criterion, device, scaler,
                region_labels=region_labels,
                save_importance_csv=False,
                fold_idx=fold
            )
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"MAE: {mae:.4f} | R2: {r2:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1

            save_path = f"/home/zhuowan/code/Age_prediction/checkpoints_oasis/our/important_best_model_fold{fold + 1}.pt"
            torch.save(best_model_state, save_path)

        model.load_state_dict(best_model_state)

        # fold结束，验证时保存节点重要性（所有验证样本）
        val_loss, mae, rmse, r2, pcc, labels, preds, fold_importances = evaluate(
            model, val_loader, criterion, device, scaler,
            region_labels=region_labels,
            save_importance_csv=True,
            fold_idx=fold
        )

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        # 保存该fold的预测结果DataFrame
        df_fold = pd.DataFrame({
            'fold': fold + 1,
            'actual_age': labels,
            'predicted_age': preds
        })
        fold_results_list.append(df_fold)

        all_labels.extend(labels)
        all_preds.extend(preds)

        all_metrics['mae'].append(mae)
        all_metrics['rmse'].append(rmse)
        all_metrics['r2'].append(r2)
        all_metrics['pcc'].append(pcc)

        print(f"✅ Fold {fold + 1} 最佳模型已保存 (Epoch {best_epoch}, MAE={best_mae:.4f})")

        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    plot_scatter(all_labels, all_preds, fold_idx="all_folds", r=pearsonr(all_labels, all_preds)[0], phase="val")

    avg_train_loss = np.mean(fold_train_losses, axis=0)
    avg_val_loss = np.mean(fold_val_losses, axis=0)
    plot_loss_curve(avg_train_loss, avg_val_loss, fold_idx="avg")

    # 保存所有fold预测结果到CSV
    df_all = pd.concat(fold_results_list, ignore_index=True)
    df_all.to_csv("/home/zhuowan/code/Age_prediction/predictions_oasis/important_our_folds_predictions.csv", index=False)
    print("✅ 所有fold的预测结果已保存到 all_folds_predictions.csv")

    print("\n===== 验证集最终结果 =====")
    print(f"平均 MAE     : {np.mean(all_metrics['mae']):.4f} ± {np.std(all_metrics['mae']):.4f}")
    print(f"平均 RMSE    : {np.mean(all_metrics['rmse']):.4f} ± {np.std(all_metrics['rmse']):.4f}")
    print(f"平均 R2      : {np.mean(all_metrics['r2']):.4f} ± {np.std(all_metrics['r2']):.4f}")
    print(f"平均 PCC     : {np.mean(all_metrics['pcc']):.4f} ± {np.std(all_metrics['pcc']):.4f}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    run_train()
