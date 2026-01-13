
# AMAge-Net  
**Attention-Guided Multimodal Neuroimaging Fusion Network for Modeling Brain Aging Pattern**  

This repository contains the official implementation of **AMAge-Net**, proposed in:  

> Zhuo Wan, Wanxiang Fu, Javed Hossain, Leonardo L. Gollo*, and Kaichao Wu*
> 
> Attention-Guided Multimodal Neuroimaging Fusion Network for Modeling Brain Aging Pattern (under review).  

---

## 📖 Overview  
Brain age estimation has emerged as a promising biomarker for characterizing individual brain health and detecting deviations from normative aging. AMAge-Net introduces a **novel attention-guided fusion framework** that integrates **structural MRI (sMRI)** and **functional MRI (fMRI)** to more accurately model aging trajectories.  

**Key highlights**  
- **Modality-specific encoders** extract complementary features from sMRI and fMRI.  
- **Attention-guided cross-modal fusion** learns shared representations while preserving modality-specific information.  
- **Interpretability** via saliency analysis, identifying brain regions contributing to aging predictions.  
- **Superior performance** compared to baseline methods on large-scale multimodal datasets.  

---


<img width="3492" height="1884" alt="model" src="https://github.com/user-attachments/assets/8141d5bf-bd1c-4b5f-b87b-93fb4f1667ed" /># AMAge-Net_Attention-Guided-Multimodal_Neuroimaging_Fusion_Network_for_Modeling_Brain_Aging-Pattern

---
## 📁 Repository Structure

```text
.
├── train.py
├── inference.py
├── model.py
├── dataset.py
├── config.py
│
├── Utils/
│   ├── average_saliency.py
│   ├── combine.py
│   ├── f_global.py
│   ├── s_global.py
│   └── utils.py
│
└── Comparison/
    ├── GCN/
    ├── GAT/
    ├── BrainGNN/
    ├── CNN_2D/
    ├── CNN_3D/
    ├── ResNet3D/
    ├── EfficientNet/
    ├── Transformer/
    └── CTransfer/
```
## 🔹 Core scripts

train.py      ----Main training script for model optimization, validation, and checkpoint saving.

inference.py  ----Inference and evaluation script for trained models.

model.py      ---- Definition of the proposed model architecture and forward computation.

dataset.py    ---- Dataset loading and preprocessing utilities.

config.py     ---- Configuration file for hyperparameters, paths, and experiment settings.
