#!/usr/bin/env bash
python tools/profile_params_memory.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --batch_size 32 \
  --output_json outputs/vireo172_fitfoodnet/profile_memory.json

