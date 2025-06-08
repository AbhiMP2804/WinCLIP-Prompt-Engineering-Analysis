# 🧠 WinCLIP Prompt Engineering Analysis

A graduate research project exploring **zero-/few-shot anomaly classification and segmentation** using **CLIP (Contrastive Language–Image Pretraining)** and prompt engineering—without the need for any model retraining. This work builds on the **WinCLIP** and **WinCLIP+** frameworks, aiming to enable scalable and instantly deployable defect detection solutions in industrial settings.

## 🚀 Overview

Traditional anomaly detection models rely on large datasets of normal images and intensive retraining. In contrast, **WinCLIP** leverages natural language prompts to detect visual defects using pre-trained vision-language models—allowing **zero-shot** classification and **pixel-level segmentation**.

This project:
- Reconstructs and extends the original WinCLIP architecture.
- Implements **prompt engineering strategies** to boost zero-shot accuracy.
- Combines **ViT-B/32** and **ConvNeXt** models for a hybrid ensemble approach.
- Adapts CLIP to grayscale datasets like **Fashion-MNIST**, simulating domain shifts in real-world applications.

## 🧩 Key Features

- 🔍 **Language-Guided Defect Detection**: Detect "normal" vs. "anomalous" states using compositional natural-language prompts.
- 🖼️ **Zero-Shot Segmentation**: Extracts multi-scale features with CLIP to localize anomalies at the pixel level.
- 🧠 **Few-Shot Enhancement (WinCLIP+)**: Improves performance using just 1–4 reference images of "normal" objects.
- 🧪 **Prompt Engineering**: Evaluated multiple prompt variants for domain adaptation on Fashion-MNIST.
- ⚙️ **Model Ensembling**: Fused CLIP models (ViT-B/32 + ConvNeXt-Base) to improve classification robustness.

## 📊 Results

| Model                  | Top-1 Accuracy | Top-5 Accuracy |
|------------------------|----------------|----------------|
| ViT-B/32 (CLIP)        | 72.99%         | 99.46%         |
| ConvNeXt-Base (CLIP)   | 73.45%         | 99.26%         |
| **Ensemble (Ours)**    | **76.15%**     | **99.60%**     |

> Achieved **91.8% AUROC** for zero-shot classification and **95.2% pixel-AUROC** with few-shot segmentation—without task-specific training.

## 🛠️ Tech Stack

- Python · PyTorch  
- CLIP · OpenCLIP · Vision Transformers (ViT-B/32)  
- ConvNeXt  
- Fashion-MNIST Dataset


## 📌 Future Work

- 🔬 Explore larger backbones (e.g., ViT-L/14, ConvNeXt-Large)
- 🧠 Add confidence-weighted voting in ensemble models
- 📐 Extend to 3D/multimodal inspection and domain-specific vision–language pretraining

## 📄 Reference

Based on the original paper:  
[**WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation**](https://arxiv.org/abs/2310.14530)
