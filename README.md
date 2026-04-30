# FITFoodNet

Official code organization workspace for:

**FITFoodNet: Frequency-Aware Texture Adaptation with Dynamic Ingredient-Aware Modeling for Fine-Grained Food Image Classification**

This repository contains FITFoodNet, the High-Frequency Texture Adapter (HFTA), the Dynamic Ingredient-Aware Head (DIA Head), training and validation/evaluation scripts, robustness evaluation, memory profiling, visualization tools, and DINOv3 baseline/PEFT comparison scripts.

## Repository Status

This directory is being prepared for public release. Local absolute paths, private logs, datasets, and pretrained weights should not be committed.

## Structure

```text
FITFoodNet_OpenSource/
|-- configs/
|-- docs/
|-- examples/
|-- fitfoodnet/
|-- scripts/
|-- tools/
|-- README.md
|-- requirements.txt
|-- environment.yml
`-- .gitignore
```

## Released Components

- FITFoodNet model implementation.
- HFTA module.
- DIA Head and frequency cross-attention.
- Orthogonality loss.
- Dataset loaders for FoodX-251 and VireoFood172.
- FITFoodNet training script.
- DINOv3 linear-probe, full-finetuning, Adapter, and AdaptFormer baseline training script.
- Clean validation/evaluation script.
- Robustness evaluation under Gaussian blur and JPEG compression.
- Parameter and peak-memory profiling script.
- Occlusion and dynamic-query heatmap visualization script.
- Lightweight FoodApp prototype for the application scenario.

## Important Training Convention

The `batch_size` option denotes the total physical batch size passed to the PyTorch `DataLoader` before `DataParallel` splits the mini-batch across GPUs. In the paper setting, FITFoodNet is trained with a total physical batch size of 64 on three NVIDIA GeForce RTX 2080 Ti GPUs. It does not mean 64 images per GPU.

## Evaluation Split

The reported FoodX-251 and VireoFood172 metrics are computed on fixed validation/evaluation splits prepared from the official datasets. They should not be described as hidden test-set results.

## Main Results

| Method | Epochs | Resolution | FoodX-251 Acc. (%) | FoodX-251 Pre. (%) | FoodX-251 F1 (%) | VireoFood172 Acc. (%) | VireoFood172 Pre. (%) | VireoFood172 F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 50 | 224 x 224 | 72.19 | 72.15 | 72.78 | 83.95 | 84.00 | 83.91 |
| ResNet-101 | 50 | 224 x 224 | 73.17 | 73.16 | 74.07 | 85.03 | 85.09 | 84.87 |
| TResNet-L | 50 | 224 x 224 | 74.75 | 74.72 | 75.56 | 91.08 | 91.02 | 90.88 |
| TResNet-XL | 50 | 224 x 224 | 75.02 | 75.16 | 75.33 | 92.15 | 91.83 | 91.91 |
| EfficientNet-b7 | 50 | 224 x 224 | 73.48 | 73.39 | 74.08 | 90.02 | 90.19 | 90.03 |
| ConvNeXt-B | 50 | 224 x 224 | 77.93 | 77.82 | 78.43 | 91.53 | 91.58 | 91.55 |
| ConvNeXt-L | 50 | 224 x 224 | 78.11 | 77.94 | 78.56 | 91.94 | 92.03 | 91.95 |
| ViT-B | 50 | 224 x 224 | 77.39 | 77.26 | 78.01 | 91.15 | 91.07 | 91.12 |
| ViT-L | 50 | 224 x 224 | 79.63 | 79.42 | 79.94 | 91.33 | 91.46 | 91.41 |
| SwinT-B | 50 | 224 x 224 | 78.63 | 78.61 | 79.22 | 91.24 | 91.27 | 91.25 |
| SwinT-L | 50 | 224 x 224 | 79.65 | 79.59 | 80.21 | 92.07 | 92.24 | 92.15 |
| SwinV2-B | 50 | 224 x 224 | 77.42 | 77.39 | 78.02 | 91.76 | 91.82 | 91.81 |
| SwinV2-L | 50 | 224 x 224 | 78.28 | 78.24 | 78.78 | 92.84 | 92.71 | 92.75 |
| DeiT-S | 50 | 224 x 224 | 73.01 | 72.93 | 73.41 | 91.46 | 91.39 | 91.45 |
| DeiT-B | 50 | 224 x 224 | 75.84 | 75.86 | 76.58 | 92.05 | 91.98 | 92.01 |
| DeiTv2-B | 50 | 224 x 224 | 75.01 | 75.32 | 75.93 | 92.37 | 92.32 | 92.35 |
| DeiTv2-L | 50 | 224 x 224 | 77.65 | 77.56 | 78.27 | 93.17 | 93.31 | 93.25 |
| Twins-B | 50 | 224 x 224 | 75.73 | 75.65 | 76.36 | 92.33 | 92.31 | 92.31 |
| Twins-L | 50 | 224 x 224 | 76.14 | 76.02 | 76.71 | 93.02 | 93.14 | 93.09 |
| Swin-ACST | 50 | 224 x 224 | 82.28 | 82.21 | 82.76 | - | - | - |
| AlsmViT-L | 50 | 224 x 224 | - | - | - | 94.29 | 94.29 | 94.25 |
| **FITFoodNet (Ours)** | **50** | **224 x 224** | **84.65** | **83.92** | **83.62** | **95.38** | **95.41** | **95.44** |

FITFoodNet achieves 84.65% accuracy on FoodX-251 and 95.38% accuracy on VireoFood172 under the paper setting.

## Quick Start

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate fitfoodnet
```

Alternatively, install dependencies with pip:

```bash
pip install -r requirements.txt
```

Update the paths in `configs/foodx251_fitfoodnet.yaml` and `configs/vireo172_fitfoodnet.yaml`, especially:

- `data_root`
- `train_dir`, `val_dir`, or list files
- `image_dirs`
- `dinov3_repo`
- `dinov3_weight`

Train FITFoodNet:

```bash
bash scripts/train_foodx251.sh
bash scripts/train_vireo172.sh
```

Run Table 2 style baseline/adaptation comparisons:

```bash
bash scripts/train_linear_probe_foodx251.sh
bash scripts/train_linear_probe_vireo172.sh
bash scripts/train_full_finetune_foodx251.sh
bash scripts/train_full_finetune_vireo172.sh
bash scripts/train_adaptformer_foodx251.sh
bash scripts/train_adaptformer_vireo172.sh
```

For full fine-tuning, the paper reports the best validation setting among tested learning rates. Re-run `tools/train_baseline.py` with different `--lr` values and select the best validation result.

## Robustness and Visualization Settings

The robustness script evaluates clean, Gaussian-blur, and JPEG-compressed validation/evaluation images. The default perturbation settings match the paper: Gaussian blur uses kernel size 7 and sigma 1.5, while JPEG compression uses quality 30.

The qualitative visualization script uses occlusion patch size 24, stride 8, and brightness factor 0.60. The top-2 query heatmaps are selected according to the query weights produced by the Query Score branch.

## FoodApp Prototype

A lightweight application prototype is provided in `examples/foodapp_prototype/`. It demonstrates how FITFoodNet can be connected to a prompt builder and an optional LLM interface for mobile-oriented dietary feedback. This prototype is not required for reproducing the benchmark results.

## Data and Pretrained Weights

FoodX-251, VireoFood172, and DINOv3 pretrained weights are not redistributed in this repository. Users should download them from their official sources and configure local paths through command-line arguments or config files.

## Citation

The BibTeX entry will be added after publication or final preprint release.
