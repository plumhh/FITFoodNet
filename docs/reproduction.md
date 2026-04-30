# Reproduction Notes

## Main Results

Run the training scripts for each dataset after updating paths in the YAML config files:

```bash
bash scripts/train_foodx251.sh
bash scripts/train_vireo172.sh
```

The `batch_size` field denotes the total physical batch size passed to the
PyTorch `DataLoader` before `DataParallel` splits the mini-batch across GPUs.
For the paper setting, the total physical batch size is 64 on three NVIDIA
GeForce RTX 2080 Ti GPUs. It is not 64 per GPU.

## Adaptation Strategy Baselines

Table 2 can be reproduced with `tools/train_baseline.py`, which supports:

- `linear_probe`
- `full_finetune`
- `adapter`
- `adaptformer`

Run the prepared scripts after updating the same dataset and DINOv3 paths used
by the main FITFoodNet configs:

```bash
bash scripts/train_linear_probe_foodx251.sh
bash scripts/train_linear_probe_vireo172.sh
bash scripts/train_full_finetune_foodx251.sh
bash scripts/train_full_finetune_vireo172.sh
bash scripts/train_adaptformer_foodx251.sh
bash scripts/train_adaptformer_vireo172.sh
```

The baseline scripts also use a total physical batch size of 64. For full
fine-tuning, the paper reports the best validation setting among tested
learning rates. To reproduce this protocol, rerun the same command with
candidate values such as:

```bash
python tools/train_baseline.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --baseline_method full_finetune \
  --lr 1.0e-5 \
  --exp_name full_finetune_foodx251_lr1e-5
```

Then report the checkpoint with the best validation accuracy under the same
training budget, data split, input resolution, and metric computation.

## Evaluation Metrics

The paper reports metrics on the fixed validation/evaluation splits of
FoodX-251 and VireoFood172. These results should not be described as hidden
test-set results.

After training, evaluate a checkpoint with:

```bash
bash scripts/evaluate_foodx251.sh
bash scripts/evaluate_vireo172.sh
```

If the checkpoint is stored elsewhere, pass it explicitly:

```bash
python tools/evaluate.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --checkpoint /path/to/fitfoodnet_best.pth
```

The reported metrics are:

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1

Class-wise zero denominators are handled with `zero_division=0`.

## Robustness Evaluation

Table 5 uses:

- Gaussian blur: kernel size 7, sigma 1.5
- JPEG compression: quality factor 30

Run robustness evaluation with:

```bash
bash scripts/evaluate_robustness_foodx251.sh
bash scripts/evaluate_robustness_vireo172.sh
```

For the `w/o HFTA` counterpart in Table 5, train a checkpoint with HFTA disabled
and evaluate it with the same script plus `--no-use_hfta`:

```bash
python tools/evaluate_blur_jpeg.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --no-use_hfta \
  --checkpoint /path/to/wo_hfta_checkpoint.pth
```

This keeps the current DIA Head unchanged and disables only the HFTA insertion.

## Memory Profiling

Peak training memory should be measured on a single NVIDIA GeForce RTX 2080 Ti with:

- input resolution 224 x 224
- batch size 32
- automatic mixed precision enabled
- one training iteration including forward, backward, and optimizer update

Run:

```bash
bash scripts/profile_memory_vireo172.sh
```

## Qualitative Visualization

The visualization script generates the panel used for qualitative analysis:

- original image
- occlusion-based sensitivity map
- top-1 dynamic-query heatmap
- top-2 dynamic-query heatmap

Default occlusion settings follow the paper: patch size 24, stride 8, and
brightness factor 0.60. Top queries are selected by descending query weights
after the Query Score branch and softmax normalization.

Run:

```bash
bash scripts/visualize_queries_vireo172.sh
```
