#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --baseline_method linear_probe \
  --exp_name linear_probe_vireo172 \
  --output_dir outputs/vireo172_linear_probe \
  --lr 5.0e-5 \
  --batch_size 64
