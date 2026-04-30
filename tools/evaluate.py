"""Evaluate FITFoodNet on the validation/evaluation split.

The paper reports metrics on fixed validation/evaluation splits for FoodX-251
and VireoFood172, not on hidden test sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser = argparse.ArgumentParser(description="Evaluate FITFoodNet")
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

    # batch_size is the total physical batch size before DataParallel split.
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 8))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.get("amp", True))

    parser.add_argument("--checkpoint", type=Path, default=defaults.get("checkpoint", None), required=defaults.get("checkpoint", None) is None)
    parser.add_argument("--output_json", type=Path, default=defaults.get("output_json", None))
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


def load_checkpoint(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True):
    model.eval()
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    autocast_enabled = amp and device.type == "cuda"

    val_bar = tqdm(loader, file=sys.stdout)
    for images, labels in val_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(images)

        preds = torch.max(outputs, dim=1)[1]
        correct += torch.eq(preds, labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / max(total, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    return acc, precision, recall, f1


def main() -> None:
    args = complete_paths(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_dataset, idx_to_class = create_datasets(args)
    num_classes = len(idx_to_class)
    num_workers = min(os.cpu_count() or 1, args.batch_size if args.batch_size > 1 else 0, args.num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
    load_checkpoint(model, args.checkpoint, device)

    acc, precision, recall, f1 = evaluate(model, val_loader, device, amp=args.amp)
    metrics = {
        "split": "validation/evaluation",
        "num_classes": num_classes,
        "num_samples": len(val_dataset),
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "accuracy_percent": acc * 100,
        "macro_precision_percent": precision * 100,
        "macro_recall_percent": recall * 100,
        "macro_f1_percent": f1 * 100,
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
