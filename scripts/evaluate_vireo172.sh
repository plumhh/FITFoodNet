#!/usr/bin/env bash
python tools/evaluate.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --checkpoint outputs/vireo172_fitfoodnet/fitfoodnet_best.pth \
  --output_json outputs/vireo172_fitfoodnet/eval_metrics.json
