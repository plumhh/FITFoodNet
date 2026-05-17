#!/usr/bin/env bash
python tools/profile_inference.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --model_kind fitfoodnet \
  --checkpoint /path/to/fitfoodnet_best.pth \
  --batch_size 1 \
  --throughput_batch_size 64 \
  --warmup_iters 50 \
  --measure_iters 200 \
  --output_json outputs/vireo172_fitfoodnet/profile_inference.json
