#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --baseline_method adapter \
  --exp_name adapter_vireo172 \
  --output_dir outputs/vireo172_adapter \
  --lr 5.0e-4 \
  --batch_size 64
