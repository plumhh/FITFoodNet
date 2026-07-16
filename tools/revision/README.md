# Revision experiment utilities

These scripts support the additional analyses reported in the revised FITFoodNet manuscript:

- `export_foodx251_failure_images.py`: exports representative FoodX-251 failure cases.
- `measure_lightweight_metrics_peft_fixed.py`: profiles parameters, memory, latency, throughput, and FLOPs for PEFT baselines.
- `test_foodx_to_uec_shared.py`: evaluates a FoodX-251 model on the shared FoodX-251/UEC-Food256 label space.
- `test_vireo_to_chinesefoodnet_shared.py`: evaluates a VireoFood172 model on the shared VireoFood172/ChineseFoodNet label space.
- `visualize_hfta_frequency_vireo_fixed.py`: visualizes HFTA frequency-band modulation.

The scripts preserve the experiment-time defaults used by the authors. Configure the path constants in the failure-case exporter, and use the command-line arguments supported by the other utilities, when running in another environment.
