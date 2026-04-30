#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --baseline_method adaptformer \
  --exp_name adaptformer_foodx251 \
  --output_dir outputs/foodx251_adaptformer \
  --lr 5.0e-4 \
  --batch_size 64
