# Environment

The released dependency files are cleaned from the author's server environment.
Local wheel paths from the training machine are intentionally removed.

## Server Used for Reported Experiments

- PyTorch: 2.5.1+cu121
- PyTorch CUDA runtime: 12.1
- CUDA available in PyTorch: true
- NVIDIA driver: 535.113.01
- Driver-reported CUDA version: 12.2
- GPUs: 3 x NVIDIA GeForce RTX 2080 Ti, 11264 MiB each

The exported wheel tags indicate Python 3.10.

## Create Environment

```bash
conda env create -f environment.yml
conda activate fitfoodnet
```

Or install into an existing environment:

```bash
pip install -r requirements.txt
```

If the CUDA wheel index is temporarily unavailable, install PyTorch manually
from the official PyTorch wheel index for CUDA 12.1 and then install the
remaining dependencies.
