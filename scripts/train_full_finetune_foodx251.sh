#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --baseline_method full_finetune \
  --exp_name full_finetune_foodx251 \
  --output_dir outputs/foodx251_full_finetune \
  --lr 5.0e-5 \
  --batch_size 64
