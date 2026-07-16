import argparse
import csv
import json
import os
import sys
import time
import shutil
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False


# =========================================================
# Dynamic import / utilities
# =========================================================

def dynamic_import_module(py_path: Path):
    py_path = py_path.resolve()
    if not py_path.exists():
        raise FileNotFoundError(f"Training script does not exist: {py_path}")

    module_name = py_path.stem + "_dynamic_profile"
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import training script: {py_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_run_dir(output_root: Path, exp_name: str, model_name: str, peft: str):
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = exp_name.strip() or f"{model_name}_{peft}"
    run_dir = output_root / f"{now_str}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100.0 * trainable_params / max(total_params, 1)
    return total_params, trainable_params, trainable_ratio


# =========================================================
# Arguments
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        "Measure PEFT lightweight metrics: train peak memory, latency, FPS, params"
    )

    parser.add_argument("--train_script", type=Path, required=True, help="Path to the training script")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to the trained best.pth checkpoint")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/amax/4t/lzh/DINOv3/lightweight_results_peft"),
    )
    parser.add_argument("--exp_name", type=str, default="")

    # ------------------------------------------------------------------
    # Dataset arguments: compatible with both old scripts and new PEFT script
    # ------------------------------------------------------------------
    parser.add_argument("--dataset", type=str, default="vireo", choices=["foodx", "vireo"])
    parser.add_argument("--dataset_type", type=str, default="txt", choices=["csv", "txt", "imagefolder"])
    parser.add_argument("--data_root", type=Path, default=Path("/home/amax/4t/lzh/data/VireoFood172"))

    parser.add_argument("--foodx_root", type=str, default="/home/amax/4t/lzh/data/FoodX251")
    parser.add_argument("--vireo_root", type=str, default="/home/amax/4t/lzh/data/VireoFood172")

    parser.add_argument("--train_csv", type=Path, default=Path("/home/amax/4t/lzh/data/FoodX251/train_labels.csv"))
    parser.add_argument("--val_csv", type=Path, default=Path("/home/amax/4t/lzh/data/FoodX251/val_labels.csv"))
    parser.add_argument("--class_list", type=Path, default=Path("/home/amax/4t/lzh/data/FoodX251/class_list.txt"))
    parser.add_argument("--train_txt", type=Path, default=Path("/home/amax/4t/lzh/data/VireoFood172/train_list.txt"))
    parser.add_argument("--val_txt", type=Path, default=Path("/home/amax/4t/lzh/data/VireoFood172/val_list.txt"))

    parser.add_argument(
        "--image_dirs",
        nargs="+",
        default=[
            "/home/amax/4t/lzh/data/FoodX251/train",
            "/home/amax/4t/lzh/data/FoodX251/val",
            "/home/amax/4t/lzh/data/FoodX251/test_set",
            "/home/amax/4t/lzh/data/FoodX251",
        ],
    )
    parser.add_argument("--train_dir", type=Path, default=Path("/home/amax/4t/lzh/data/FoodX251/train"))
    parser.add_argument("--val_dir", type=Path, default=Path("/home/amax/4t/lzh/data/FoodX251/val"))

    # ------------------------------------------------------------------
    # Model / PEFT arguments
    # ------------------------------------------------------------------
    parser.add_argument("--model_name", type=str, default="dinov3_l", choices=["dinov3_s", "dinov3_b", "dinov3_l"])
    parser.add_argument("--method", type=str, default="adapter", choices=["linear_probe", "full_finetune", "adapter", "adaptformer"])
    parser.add_argument("--peft", type=str, default="lora", choices=["linear", "lora", "ssf", "vpt_deep"])

    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used to measure peak training memory")
    parser.add_argument("--infer_batch_size", type=int, default=64, help="Batch size used to measure throughput/FPS")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=-1, help="Known class count; otherwise inferred from the dataset")

    parser.add_argument("--insert_layers", nargs="+", type=int, default=[5, 11, 17, 23])
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_scale", type=float, default=1e-4)
    parser.add_argument("--disable_adapter_ln", action="store_true")
    parser.add_argument("--head_hidden_dim", type=int, default=1024)
    parser.add_argument("--head_dropout", type=float, default=0.4)
    parser.add_argument("--allow_mlp_in_linear_probe", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")  # compatibility alias for training scripts

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # VPT arguments: keep both names for compatibility
    parser.add_argument("--vpt_prompt_len", type=int, default=10)
    parser.add_argument("--prompt_len", type=int, default=10)

    # PEFT optimizer compatibility
    parser.add_argument("--peft_lr", type=float, default=-1.0)

    # DINOv3 backbone paths: keep both old and new argument names
    parser.add_argument("--dinov3_repo", type=str, default="/home/amax/.cache/torch/hub/facebookresearch_dinov3_main")
    parser.add_argument("--dinov3_weight", type=str, default="/home/amax/4t/lzh/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--local_repo_path", type=str, default="/home/amax/.cache/torch/hub/facebookresearch_dinov3_main")
    parser.add_argument("--github_repo", type=str, default="facebookresearch/dinov3")
    parser.add_argument("--use_github_for_small", action="store_true")
    parser.add_argument("--weight_s", type=str, default="")
    parser.add_argument("--weight_b", type=str, default="/home/amax/4t/lzh/DINOv3/dinov3_vitb16_pretrain.pth")
    parser.add_argument("--weight_l", type=str, default="/home/amax/4t/lzh/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

    # Measurement arguments
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP for latency, FPS, and training-memory measurements")
    parser.add_argument("--measure_train_peak_memory", action="store_true")
    parser.add_argument("--measure_latency", action="store_true")
    parser.add_argument("--measure_infer_peak_memory", action="store_true")
    parser.add_argument("--measure_flops", action="store_true")

    return parser.parse_args()


def normalize_args_for_dataset(args):
    """Fill dataset-specific paths so both old scripts and PEFT scripts can read them."""
    if args.dataset == "vireo":
        args.dataset_type = "txt"
        args.data_root = Path(args.vireo_root)
        args.train_txt = Path(args.vireo_root) / "train_list.txt"
        args.val_txt = Path(args.vireo_root) / "val_list.txt"
    elif args.dataset == "foodx":
        args.dataset_type = "csv"
        args.data_root = Path(args.foodx_root)
        args.train_csv = Path(args.foodx_root) / "train_labels.csv"
        args.val_csv = Path(args.foodx_root) / "val_labels.csv"
        args.class_list = Path(args.foodx_root) / "class_list.txt"
        args.image_dirs = [
            str(Path(args.foodx_root) / "train"),
            str(Path(args.foodx_root) / "val"),
            str(Path(args.foodx_root) / "test_set"),
            str(Path(args.foodx_root)),
        ]
        args.train_dir = Path(args.foodx_root) / "train"
        args.val_dir = Path(args.foodx_root) / "val"

    # Keep aliases synchronized.
    args.local_repo_path = args.dinov3_repo
    args.weight_l = args.dinov3_weight
    if not hasattr(args, "output_root"):
        args.output_root = Path("/home/amax/4t/lzh/DINOv3/lightweight_results_peft")

    return args


# =========================================================
# Data / model compatibility
# =========================================================

def resolve_num_classes_and_train_loader(args, module):
    if args.num_classes > 0 and not args.measure_train_peak_memory:
        return args.num_classes, None

    if not hasattr(module, "create_datasets"):
        if args.num_classes <= 0:
            raise AttributeError(
                "The training script has no create_datasets function; provide --num_classes"
            )
        return args.num_classes, None

    train_dataset, _val_dataset, idx_to_class = module.create_datasets(args)
    num_classes = len(idx_to_class)

    train_loader = None
    if args.measure_train_peak_memory:
        nw = min([os.cpu_count() or 1, args.batch_size if args.batch_size > 1 else 0, args.num_workers])
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return num_classes, train_loader


def load_model(args, module, num_classes: int, device: torch.device):
    # Compatible with your current PEFT training script:
    #   class DINOv3PEFTClassifier(nn.Module):
    #       def __init__(self, num_classes: int, args)
    # Also keeps compatibility with earlier PEFTCompareClassifier / DINOv3Classifier scripts.
    if hasattr(module, "DINOv3PEFTClassifier"):
        print("[Model] Using DINOv3PEFTClassifier from training script")
        model = module.DINOv3PEFTClassifier(num_classes=num_classes, args=args)
    elif hasattr(module, "PEFTCompareClassifier"):
        print("[Model] Using PEFTCompareClassifier from training script")
        model = module.PEFTCompareClassifier(num_classes=num_classes, args=args)
    elif hasattr(module, "DINOv3Classifier"):
        print("[Model] Using DINOv3Classifier from training script")
        # old AdaptFormer / adapter scripts
        try:
            model = module.DINOv3Classifier(
                num_classes=num_classes,
                model_type=args.model_name,
                method=args.method,
                adapter_dim=args.adapter_dim,
                insert_layers=args.insert_layers,
                adapter_dropout=args.adapter_dropout,
                adapter_scale=args.adapter_scale,
            )
        except TypeError:
            try:
                model = module.DINOv3Classifier(num_classes=num_classes, args=args)
            except TypeError:
                model = module.DINOv3Classifier(num_classes=num_classes, model_type=args.model_name)
    else:
        class_names = [name for name in dir(module) if name.lower().endswith("classifier") or "classifier" in name.lower()]
        raise AttributeError(
            "No supported model class was found in the training script. Expected: "
            "DINOv3PEFTClassifier / PEFTCompareClassifier / DINOv3Classifier. "
            f"Candidate classifier classes: {class_names}"
        )

    if not args.weight_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.weight_path}")

    state_dict = safe_torch_load(args.weight_path)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[nk] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)}")
        print("       first missing:", missing[:10])
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)}")
        print("       first unexpected:", unexpected[:10])
    print(f"[OK] Loaded weights from: {args.weight_path}")

    return model.to(device)


# =========================================================
# Measurement
# =========================================================

def measure_model_size_mb(weight_path: Path):
    return os.path.getsize(weight_path) / (1024 * 1024)


@torch.no_grad()
def measure_latency_and_throughput(model, args, device):
    model.eval()

    # latency: batch size 1
    x1 = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    for _ in range(args.warmup):
        if args.use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _ = model(x1)
        else:
            _ = model(x1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(args.runs):
        if args.use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _ = model(x1)
        else:
            _ = model(x1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0 / max(args.runs, 1)

    # FPS/throughput: infer_batch_size, normally 64 according to your manuscript
    xb = torch.randn(args.infer_batch_size, 3, args.img_size, args.img_size, device=device)
    for _ in range(args.warmup):
        if args.use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _ = model(xb)
        else:
            _ = model(xb)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(args.runs):
        if args.use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                _ = model(xb)
        else:
            _ = model(xb)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    throughput = args.runs * args.infer_batch_size / max(t1 - t0, 1e-12)

    return latency_ms, throughput


@torch.no_grad()
def measure_infer_peak_memory(model, args, device):
    if device.type != "cuda":
        return None

    model.eval()
    x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if args.use_amp:
        with torch.amp.autocast("cuda"):
            _ = model(x)
    else:
        _ = model(x)

    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def build_optimizer_for_memory(model: nn.Module, args, module):
    """Use the training script optimizer if available; otherwise fall back to Adam."""
    if hasattr(module, "build_optimizer"):
        try:
            return module.build_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)
        except TypeError:
            pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)


def measure_train_peak_memory(model, train_loader, args, device, module):
    if device.type != "cuda" or train_loader is None:
        return None

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer_for_memory(model, args, module)
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    images, labels = next(iter(train_loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    autocast_enabled = args.use_amp and device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=autocast_enabled):
        try:
            outputs = model(images, return_attn=True)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                logits, aux = outputs
                cls_loss = criterion(logits, labels)

                ortho_loss = None
                if hasattr(module, "compute_orthogonal_loss"):
                    try:
                        if isinstance(aux, dict) and "attn_weights" in aux:
                            ortho_loss = module.compute_orthogonal_loss(aux["attn_weights"])
                        else:
                            ortho_loss = module.compute_orthogonal_loss(aux)
                    except Exception:
                        ortho_loss = None

                loss = cls_loss + 0.1 * ortho_loss if ortho_loss is not None else cls_loss
            else:
                logits = outputs
                loss = criterion(logits, labels)
        except TypeError:
            logits = model(images)
            loss = criterion(logits, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def measure_flops_and_macs(model, args, device):
    if not THOP_AVAILABLE:
        return None, None, "thop_not_installed"

    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    try:
        macs, _params = profile(model, inputs=(dummy,), verbose=False)
        flops = macs * 2
        return flops, macs, "approx"
    except Exception as e:
        return None, None, f"failed: {e}"


# =========================================================
# Save
# =========================================================

def save_results(run_dir: Path, results: dict, args):
    json_path = run_dir / "lightweight_metrics.json"
    csv_path = run_dir / "lightweight_metrics.csv"
    txt_path = run_dir / "summary.txt"
    cmd_path = run_dir / "run_command.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results.items():
            writer.writerow([k, v])

    with txt_path.open("w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    with cmd_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    try:
        shutil.copy2(args.train_script, run_dir / args.train_script.name)
    except Exception:
        pass

    return json_path, csv_path, txt_path, cmd_path


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    args = normalize_args_for_dataset(args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(args.output_root, args.exp_name, args.model_name, args.peft)
    print(f"[Output Dir] {run_dir}")

    module = dynamic_import_module(args.train_script)

    if hasattr(module, "set_seed"):
        module.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    num_classes, train_loader = resolve_num_classes_and_train_loader(args, module)
    print(f"Dataset: {args.dataset}")
    print(f"PEFT: {args.peft}")
    print(f"Num classes: {num_classes}")

    model = load_model(args, module, num_classes, device)

    total_params, trainable_params, trainable_ratio = count_parameters(model)
    model_size_mb = measure_model_size_mb(args.weight_path)

    flops = macs = flops_note = None
    if args.measure_flops:
        flops, macs, flops_note = measure_flops_and_macs(model, args, device)

    latency_ms = throughput = None
    if args.measure_latency:
        latency_ms, throughput = measure_latency_and_throughput(model, args, device)

    infer_peak_mem_gb = None
    if args.measure_infer_peak_memory:
        infer_peak_mem_gb = measure_infer_peak_memory(model, args, device)

    train_peak_mem_gb = None
    if args.measure_train_peak_memory:
        train_peak_mem_gb = measure_train_peak_memory(model, train_loader, args, device, module)

    results = {
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_type": args.dataset_type,
        "model_name": args.model_name,
        "method": args.method,
        "peft": args.peft,
        "weight_path": str(args.weight_path),
        "train_script": str(args.train_script),
        "img_size": args.img_size,
        "train_batch_size": args.batch_size,
        "infer_batch_size": args.infer_batch_size,
        "use_amp": args.use_amp,
        "warmup": args.warmup,
        "runs": args.runs,
        "total_params": total_params,
        "total_params_M": round(total_params / 1e6, 4),
        "trainable_params": trainable_params,
        "trainable_params_M": round(trainable_params / 1e6, 4),
        "trainable_ratio_percent": round(trainable_ratio, 4),
        "model_size_MB": round(model_size_mb, 4),
        "macs_G": None if macs is None else round(macs / 1e9, 4),
        "flops_G": None if flops is None else round(flops / 1e9, 4),
        "flops_note": flops_note,
        "latency_ms_per_image": None if latency_ms is None else round(latency_ms, 4),
        "throughput_images_per_s": None if throughput is None else round(throughput, 4),
        "infer_peak_memory_GB": None if infer_peak_mem_gb is None else round(infer_peak_mem_gb, 4),
        "train_peak_memory_GB": None if train_peak_mem_gb is None else round(train_peak_mem_gb, 4),
        "train_peak_memory_MiB": None if train_peak_mem_gb is None else round(train_peak_mem_gb * 1024.0, 2),
    }

    print("\n========== Results ==========")
    for k, v in results.items():
        print(f"{k}: {v}")

    json_path, csv_path, txt_path, cmd_path = save_results(run_dir, results, args)

    print("\n========== Saved ==========")
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")
    print(f"TXT  : {txt_path}")
    print(f"CMD  : {cmd_path}")


if __name__ == "__main__":
    main()
