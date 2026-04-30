"""Profile parameter counts and peak training memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitfoodnet.datasets import create_datasets  # noqa: E402
from fitfoodnet.losses import compute_orthogonal_loss, ortho_weight  # noqa: E402
from fitfoodnet.model import FITFoodNet  # noqa: E402


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section in ("dataset", "model", "training", "paths"):
        values = config.get(section, {})
        if isinstance(values, dict):
            flat.update(values)
    return flat


def load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return flatten_config(config)


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Profile FITFoodNet parameter and memory cost")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset_type", type=str, default=defaults.get("type", "imagefolder"), choices=["imagefolder", "txt", "csv"])
    parser.add_argument("--data_root", type=Path, default=defaults.get("data_root", None))
    parser.add_argument("--train_dir", type=Path, default=defaults.get("train_dir", None))
    parser.add_argument("--val_dir", type=Path, default=defaults.get("val_dir", None))
    parser.add_argument("--train_txt", type=Path, default=defaults.get("train_txt", None))
    parser.add_argument("--val_txt", type=Path, default=defaults.get("val_txt", None))
    parser.add_argument("--train_csv", type=Path, default=defaults.get("train_csv", None))
    parser.add_argument("--val_csv", type=Path, default=defaults.get("val_csv", None))
    parser.add_argument("--class_list", type=Path, default=defaults.get("class_list", None))
    parser.add_argument("--image_dirs", nargs="+", default=defaults.get("image_dirs", None))

    parser.add_argument("--model_name", type=str, default=defaults.get("backbone", "dinov3_l"), choices=["dinov3_s", "dinov3_b", "dinov3_l"])
    parser.add_argument("--img_size", type=int, default=defaults.get("img_size", 224))
    parser.add_argument("--num_queries", type=int, default=defaults.get("num_queries", 8))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--hfta_bottleneck", type=int, default=defaults.get("hfta_bottleneck", 64))
    parser.add_argument("--use_hfta", action=argparse.BooleanOptionalAction, default=defaults.get("use_hfta", True))
    parser.add_argument("--dinov3_repo", type=str, default=defaults.get("dinov3_repo", "facebookresearch/dinov3"))
    parser.add_argument("--dinov3_source", type=str, default=defaults.get("dinov3_source", "github"), choices=["github", "local"])
    parser.add_argument("--dinov3_weight", type=Path, default=defaults.get("dinov3_weight", None))

    # Table 2 uses single-GPU profiling with batch size 32 and AMP.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 8))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 5e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_ortho_max", type=float, default=defaults.get("lambda_ortho_max", 0.03))
    parser.add_argument("--lambda_ortho_warmup_epochs", type=int, default=defaults.get("lambda_ortho_warmup_epochs", 10))
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    defaults = load_config_defaults(config_args.config)
    return build_parser(defaults).parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.data_root is not None:
        data_root = Path(args.data_root)
        if args.train_dir is None:
            args.train_dir = data_root / "train"
        if args.val_dir is None:
            args.val_dir = data_root / "val"
        if args.train_txt is None:
            args.train_txt = data_root / "train_list.txt"
        if args.val_txt is None:
            args.val_txt = data_root / "val_list.txt"
        if args.train_csv is None:
            args.train_csv = data_root / "train_labels.csv"
        if args.val_csv is None:
            args.val_csv = data_root / "val_labels.csv"
        if args.class_list is None:
            args.class_list = data_root / "class_list.txt"
        if args.image_dirs is None:
            args.image_dirs = [str(data_root / "train"), str(data_root / "val"), str(data_root)]
    return args


def main() -> None:
    args = complete_paths(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for peak-memory profiling.")

    device = torch.device("cuda:0")
    train_dataset, _, idx_to_class = create_datasets(args)
    num_classes = len(idx_to_class)
    num_workers = min(os.cpu_count() or 1, args.batch_size if args.batch_size > 1 else 0, args.num_workers)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = FITFoodNet(
        num_classes=num_classes,
        model_type=args.model_name,
        img_size=args.img_size,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        hfta_bottleneck=args.hfta_bottleneck,
        use_hfta=args.use_hfta,
        dinov3_repo=args.dinov3_repo,
        dinov3_source=args.dinov3_source,
        dinov3_weight=args.dinov3_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=args.amp):
        logits, aux = model(images, return_attn=True)
        cls_loss = criterion(logits, labels)
        loss_ortho = compute_orthogonal_loss(aux["attn_weights"])
        loss = cls_loss + ortho_weight(0, args.lambda_ortho_max, args.lambda_ortho_warmup_epochs) * loss_ortho
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    result = {
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "amp": args.amp,
        "use_hfta": args.use_hfta,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "trainable_ratio_percent": trainable_params / total_params * 100,
        "train_peak_memory_mib": peak_memory_mib,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

