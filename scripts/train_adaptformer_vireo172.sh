#!/usr/bin/env bash
python tools/train_baseline.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --baseline_method adaptformer \
  --exp_name adaptformer_vireo172 \
  --output_dir outputs/vireo172_adaptformer \
  --lr 5.0e-4 \
  --batch_size 64
