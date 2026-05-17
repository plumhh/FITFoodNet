"""Profile single-image latency and batch throughput.

The measurement uses random tensors with the same input resolution as the
paper setting and excludes data loading and image preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitfoodnet.baselines import DINOv3BaselineClassifier  # noqa: E402
from fitfoodnet.datasets import create_datasets  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Profile inference latency and throughput")
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
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--model_kind", type=str, default="fitfoodnet", choices=["fitfoodnet", "baseline"])
    parser.add_argument("--baseline_method", type=str, default="linear_probe", choices=["linear_probe", "full_finetune", "adapter", "adaptformer"])
    parser.add_argument("--model_name", type=str, default=defaults.get("backbone", "dinov3_l"), choices=["dinov3_s", "dinov3_b", "dinov3_l"])
    parser.add_argument("--img_size", type=int, default=defaults.get("img_size", 224))
    parser.add_argument("--num_queries", type=int, default=defaults.get("num_queries", 8))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--hfta_bottleneck", type=int, default=defaults.get("hfta_bottleneck", 64))
    parser.add_argument("--use_hfta", action=argparse.BooleanOptionalAction, default=defaults.get("use_hfta", True))
    parser.add_argument("--adapter_dim", type=int, default=defaults.get("adapter_dim", 64))
    parser.add_argument("--insert_layers", nargs="+", type=int, default=defaults.get("insert_layers", [5, 11, 17, 23]))
    parser.add_argument("--adapter_dropout", type=float, default=defaults.get("adapter_dropout", 0.0))
    parser.add_argument("--adapter_scale", type=float, default=defaults.get("adapter_scale", 1e-4))
    parser.add_argument("--dinov3_repo", type=str, default=defaults.get("dinov3_repo", "facebookresearch/dinov3"))
    parser.add_argument("--dinov3_source", type=str, default=defaults.get("dinov3_source", "github"), choices=["github", "local"])
    parser.add_argument("--dinov3_weight", type=Path, default=defaults.get("dinov3_weight", None))

    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--throughput_batch_size", type=int, default=64)
    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--measure_iters", type=int, default=200)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.get("amp", True))
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


def infer_num_classes(args: argparse.Namespace) -> int:
    if args.num_classes is not None:
        return args.num_classes
    try:
        _, _, idx_to_class = create_datasets(args)
        return len(idx_to_class)
    except Exception as exc:
        raise RuntimeError("Could not infer num_classes from dataset. Pass --num_classes explicitly.") from exc


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def build_model(args: argparse.Namespace, num_classes: int, device: torch.device) -> torch.nn.Module:
    if args.model_kind == "fitfoodnet":
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
        )
    else:
        model = DINOv3BaselineClassifier(
            num_classes=num_classes,
            method=args.baseline_method,
            model_type=args.model_name,
            adapter_dim=args.adapter_dim,
            insert_layers=args.insert_layers,
            adapter_dropout=args.adapter_dropout,
            adapter_scale=args.adapter_scale,
            dinov3_repo=args.dinov3_repo,
            dinov3_source=args.dinov3_source,
            dinov3_weight=args.dinov3_weight,
        )

    model = model.to(device).eval()
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, device)
    return model


@torch.no_grad()
def measure(model: torch.nn.Module, batch_size: int, img_size: int, warmup_iters: int, measure_iters: int, amp: bool, device: torch.device) -> dict[str, float]:
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    amp_enabled = amp and device.type == "cuda"

    for _ in range(warmup_iters):
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(measure_iters):
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    images = batch_size * measure_iters
    return {
        "batch_size": float(batch_size),
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": elapsed / measure_iters * 1000.0,
        "latency_ms_per_image": elapsed / images * 1000.0,
        "fps": images / elapsed,
    }


def main() -> None:
    args = complete_paths(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = infer_num_classes(args)
    model = build_model(args, num_classes, device)

    single = measure(model, args.batch_size, args.img_size, args.warmup_iters, args.measure_iters, args.amp, device)
    throughput = measure(model, args.throughput_batch_size, args.img_size, args.warmup_iters, args.measure_iters, args.amp, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "model_kind": args.model_kind,
        "baseline_method": args.baseline_method if args.model_kind == "baseline" else None,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "amp": args.amp and device.type == "cuda",
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "single_batch": single,
        "throughput_batch": throughput,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
