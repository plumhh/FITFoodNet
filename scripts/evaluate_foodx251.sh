#!/usr/bin/env bash
python tools/evaluate.py \
  --config configs/foodx251_fitfoodnet.yaml \
  --checkpoint outputs/foodx251_fitfoodnet/fitfoodnet_best.pth \
  --output_json outputs/foodx251_fitfoodnet/eval_metrics.json
