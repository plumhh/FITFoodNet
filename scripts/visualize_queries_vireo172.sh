#!/usr/bin/env bash
python tools/visualize_queries.py \
  --config configs/vireo172_fitfoodnet.yaml \
  --checkpoint outputs/vireo172_fitfoodnet/fitfoodnet_best.pth \
  --class_json outputs/vireo172_fitfoodnet/class_indices.json \
  --sample_index 0 \
  --max_samples 1 \
  --output_root outputs/vireo172_fitfoodnet/query_visualization

