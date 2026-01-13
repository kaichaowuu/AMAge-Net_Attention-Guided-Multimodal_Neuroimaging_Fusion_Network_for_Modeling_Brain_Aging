from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
import os


class MultimodalBrainDataset(Dataset):
    def __init__(self, struct_fmri_paths, subject_age_dict, target_shape=(128, 128, 128)):
        """
        struct_fmri_paths: list of tuples (struct_path, fmri_path)
        subject_age_dict: dict {subject_id: age}
        target_shape: resize shape for MRI volume
        """
        self.struct_fmri_paths = struct_fmri_paths
        self.subject_age_dict = subject_age_dict
        self.target_shape = target_shape

        # 初始化脑区标签为None，将在首次加载时提取
        self.region_labels = None

        # 确保数据集不为空
        if len(struct_fmri_paths) > 0:
            self._initialize_region_labels()

    def __len__(self):
        return len(self.struct_fmri_paths)

    def _initialize_region_labels(self):
        """在首次加载时提取脑区标签"""
        if self.region_labels is None and len(self.struct_fmri_paths) > 0:
            struct_path, fmri_path = self.struct_fmri_paths[0]
            try:
                fc_df = pd.read_csv(fmri_path, index_col=0)
                self.region_labels = fc_df.columns.tolist()
                print(f"脑区标签已提取: 共 {len(self.region_labels)} 个脑区")

                # 验证标签的数量
                if len(self.region_labels) != 90:
                    print(f"警告: 发现 {len(self.region_labels)} 个脑区标签 (应为90)")
            except Exception as e:
                print(f"提取脑区标签时出错: {str(e)}")
                # 使用默认标签作为后备
                self.region_labels = [f"Region_{i}" for i in range(90)]
                print(f"使用默认脑区标签: 共 {len(self.region_labels)} 个脑区")

    def __getitem__(self, idx):
        struct_path, fmri_path = self.struct_fmri_paths[idx]
        subject_id = os.path.basename(os.path.dirname(struct_path))

        # 处理结构MRI
        img = nib.load(struct_path).get_fdata()
        zoom_factors = [t / o for t, o in zip(self.target_shape, img.shape)]
        img = zoom(img, zoom_factors, order=1)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]


        # 读取 fMRI 连接矩阵
        fc_df = pd.read_csv(fmri_path, index_col=0)
        conn = fc_df.values.astype(np.float32)
        conn = torch.tensor(conn)  # [90, 90]

        # 节点特征 - 使用连接矩阵的行作为节点特征
        node_feature = conn.clone()  # 每个节点的特征是该节点与其他节点的连接强度

        # 只保留 > 0 的边
        conn_masked = conn.clone()
        conn_masked[conn_masked <= 0] = 0
        edge_index, edge_attr = dense_to_sparse(conn_masked)

        # 创建批次索引（初始化为 0）
        batch = torch.zeros(90, dtype=torch.long)

        data = Data(
            x=node_feature,
            edge_index=edge_index,
            edge_attr=edge_attr,
            img=img,
            y=torch.tensor([self.subject_age_dict[subject_id]], dtype=torch.float32),
            batch=torch.zeros(90, dtype=torch.long)
        )

        data.subject_id = [subject_id]
        return data

    def get_region_labels(self):
        """获取脑区标签"""
        # 确保标签已初始化
        if self.region_labels is None:
            self._initialize_region_labels()
        return self.region_labels
