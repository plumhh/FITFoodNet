#!/usr/bin/env bash
python tools/evaluate_blur_jpeg.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --checkpoint outputs/foodx251_fitfoodnet/fitfoodnet_best.pth \
  --output_json outputs/foodx251_fitfoodnet/robustness_metrics.json

