# FITFoodNet

Official implementation for the manuscript submitted to **The Visual Computer**:

**Frequency-Aware Adaptive Modeling for Fine-Grained Food Visual Recognition**

This repository is directly associated with the above manuscript. If you use this code, benchmark protocol, or reproduced results, please cite the related manuscript. A formal BibTeX entry and archival DOI will be updated after the final preprint/publication or Zenodo release is available.

## What This Repository Provides

- FITFoodNet implementation with a frozen DINOv3 backbone.
- High-Frequency Texture Adapter (HFTA).
- Dynamic Ingredient-Aware Head (DIA Head).
- Frequency cross-attention and query-orthogonality loss.
- Training scripts for FoodX-251 and VireoFood172.
- Clean validation/evaluation scripts.
- Robustness evaluation under Gaussian blur and JPEG compression.
- Parameter, peak-memory, latency, and throughput profiling tools.
- Occlusion-based feature sensitivity and dynamic-query heatmap visualization.
- DINOv3 linear probe, full fine-tuning, Adapter, and AdaptFormer comparison scripts.
- A lightweight FoodApp prototype used only to demonstrate the application scenario.

## Repository Structure

```text
FITFoodNet/
|-- configs/                    # Dataset/model/training configs
|-- docs/                       # Environment, data, and reproduction notes
|-- examples/foodapp_prototype/ # Optional application prototype
|-- fitfoodnet/                 # Model implementation
|-- scripts/                    # Reproducible shell entrypoints
|-- tools/                      # Training, evaluation, profiling, visualization
|-- CITATION.cff                # Citation metadata
|-- environment.yml             # Conda environment
|-- requirements.txt            # Pip requirements
`-- README.md
```

## Environment

The reported experiments were run with:

- Python 3.10
- PyTorch 2.5.1+cu121
- CUDA runtime used by PyTorch: 12.1
- GPUs: 3 x NVIDIA GeForce RTX 2080 Ti
- Input resolution: 224 x 224
- Automatic mixed precision enabled

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate fitfoodnet
```

Or install dependencies with pip:

```bash
pip install -r requirements.txt
```

More environment details are provided in `docs/environment.md`.

## Data Preparation

This repository does not redistribute FoodX-251, VireoFood172, dataset images, or DINOv3 pretrained weights. Please download them from the official sources and follow their licenses or academic access agreements.

### FoodX-251

Expected structure:

```text
FoodX251/
|-- train/
|-- val/
|-- test_set/
|-- train_labels.csv
|-- val_labels.csv
`-- class_list.txt
```

The CSV files should contain:

```csv
img_name,label
example.jpg,0
```

### VireoFood172

ImageFolder layout:

```text
VireoFood172/
|-- train/
|   |-- class_000/
|   `-- ...
`-- val/
    |-- class_000/
    `-- ...
```

TXT-list layout is also supported:

```text
train_list.txt
val_list.txt
```

Each line should contain:

```text
relative_image_path label
```

Detailed notes are in `docs/data_preparation.md`.

## Configure Local Paths

Before running experiments, edit:

```text
configs/foodx251_fitfoodnet.yaml
configs/vireo172_fitfoodnet.yaml
```

Set the following fields:

- `data_root`
- `train_dir`, `val_dir`, or list/CSV files
- `image_dirs`
- `dinov3_repo`
- `dinov3_source`
- `dinov3_weight`
- `output_dir`

`dinov3_weight` should point to the official DINOv3 pretrained checkpoint downloaded by the user.

## Training FITFoodNet

FoodX-251:

```bash
bash scripts/train_foodx251.sh
```

VireoFood172:

```bash
bash scripts/train_vireo172.sh
```

Important convention: `batch_size` is the **total physical batch size** passed to the PyTorch `DataLoader` before `DataParallel` splits it across GPUs. In the paper setting, the total physical batch size is 64 on three RTX 2080 Ti GPUs. It is not 64 images per GPU.

## Clean Validation Evaluation

FoodX-251:

```bash
bash scripts/evaluate_foodx251.sh
```

VireoFood172:

```bash
bash scripts/evaluate_vireo172.sh
```

Evaluate a custom checkpoint:

```bash
python tools/evaluate.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --checkpoint /path/to/fitfoodnet_best.pth \
  --output_json outputs/vireo172_fitfoodnet/eval.json
```

The paper reports metrics on official validation/evaluation splits, not hidden test sets. Reported metrics are accuracy, macro-precision, macro-recall, and macro-F1.

## Baseline and PEFT Comparisons

Run DINOv3-L comparison methods:

```bash
bash scripts/train_linear_probe_foodx251.sh
bash scripts/train_linear_probe_vireo172.sh
bash scripts/train_full_finetune_foodx251.sh
bash scripts/train_full_finetune_vireo172.sh
bash scripts/train_adaptformer_foodx251.sh
bash scripts/train_adaptformer_vireo172.sh
```

For full fine-tuning, the paper reports the best validation result among the tested learning rates under the same training budget. To reproduce this protocol, run `tools/train_baseline.py` with candidate learning rates and select the best validation checkpoint.

## Robustness Evaluation

The robustness experiment uses:

- Gaussian blur: kernel size 7, sigma 1.5
- JPEG compression: quality factor 30

Run:

```bash
bash scripts/evaluate_robustness_foodx251.sh
bash scripts/evaluate_robustness_vireo172.sh
```

For the w/o HFTA counterpart:

```bash
python tools/evaluate_blur_jpeg.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --no-use_hfta \
  --checkpoint /path/to/wo_hfta_checkpoint.pth
```

## Parameter, Memory, Latency, and FPS Profiling

Peak training memory and parameter counts:

```bash
bash scripts/profile_memory_vireo172.sh
```

Inference latency and throughput:

```bash
bash scripts/profile_inference_vireo172.sh
```

Equivalent direct command:

```bash
python tools/profile_inference.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --model_kind fitfoodnet \
  --checkpoint /path/to/fitfoodnet_best.pth \
  --batch_size 1 \
  --throughput_batch_size 64 \
  --warmup_iters 50 \
  --measure_iters 200 \
  --output_json outputs/vireo172_fitfoodnet/profile_inference.json
```

The profiling script uses random tensors with the same input size and excludes data loading and image preprocessing, matching the paper's inference-speed reporting convention.

## Occlusion and Dynamic-Query Heatmap Visualization

Generate qualitative visualizations:

```bash
bash scripts/visualize_queries_vireo172.sh
```

Default settings match the manuscript:

- occlusion patch size: 24
- stride: 8
- occluded-region brightness factor: 0.60
- top-k query heatmaps: 2

The generated panels include:

- original image
- occlusion-based feature sensitivity heatmap
- top-1 dynamic-query attention heatmap
- top-2 dynamic-query attention heatmap

## Main Reproduced Results

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

## Checkpoints and Pretrained Weights

This repository does not include large model weights in Git history.

- DINOv3 pretrained weights must be downloaded from the official DINOv3 source.
- Dataset images are not redistributed.
- Trained FITFoodNet checkpoints can be reproduced by the training commands above.
- If model-weight redistribution is permitted by all relevant licenses and storage policies, archival checkpoint links will be provided through GitHub Releases or Zenodo.

## Additional Documentation

- `docs/environment.md`: server and dependency details.
- `docs/data_preparation.md`: dataset layout.
- `docs/reproduction.md`: detailed reproduction notes.
- `docs/files_needed_from_author.md`: private files that must not be committed.

## FoodApp Prototype

`examples/foodapp_prototype/` contains a lightweight prototype showing how FITFoodNet predictions can be connected to a prompt builder and optional large language model interface. This prototype is not required to reproduce benchmark results.

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{xiao2026fitfoodnet,
  title   = {Frequency-Aware Adaptive Modeling for Fine-Grained Food Visual Recognition},
  author  = {Xiao, Zhiyong and Li, Zihao and Deng, Zhaohong},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

Please also cite the official datasets and DINOv3 backbone according to their original licenses and citation instructions.
