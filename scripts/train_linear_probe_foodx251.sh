#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --baseline_method linear_probe \
  --exp_name linear_probe_foodx251 \
  --output_dir outputs/foodx251_linear_probe \
  --lr 2.0e-3 \
  --batch_size 64
