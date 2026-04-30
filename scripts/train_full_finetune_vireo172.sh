#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --baseline_method full_finetune \
  --exp_name full_finetune_vireo172 \
  --output_dir outputs/vireo172_full_finetune \
  --lr 5.0e-5 \
  --batch_size 64
