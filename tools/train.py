"""Train FITFoodNet on FoodX-251 or VireoFood172."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitfoodnet.datasets import create_datasets  # noqa: E402
from fitfoodnet.losses import compute_orthogonal_loss, ortho_weight  # noqa: E402
from fitfoodnet.model import FITFoodNet  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert nested YAML config fields into argparse defaults."""

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
    parser = argparse.ArgumentParser(description="Train FITFoodNet")
    parser.add_argument("--config", type=str, default=None)

    # Dataset.
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

    # Model.
    parser.add_argument("--model_name", type=str, default=defaults.get("backbone", "dinov3_l"), choices=["dinov3_s", "dinov3_b", "dinov3_l"])
    parser.add_argument("--img_size", type=int, default=defaults.get("img_size", 224))
    parser.add_argument("--num_queries", type=int, default=defaults.get("num_queries", 8))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--hfta_bottleneck", type=int, default=defaults.get("hfta_bottleneck", 64))
    parser.add_argument("--use_hfta", action=argparse.BooleanOptionalAction, default=defaults.get("use_hfta", True))
    parser.add_argument("--dinov3_repo", type=str, default=defaults.get("dinov3_repo", "facebookresearch/dinov3"))
    parser.add_argument("--dinov3_source", type=str, default=defaults.get("dinov3_source", "github"), choices=["github", "local"])
    parser.add_argument("--dinov3_weight", type=Path, default=defaults.get("dinov3_weight", None))

    # Training. batch_size is the total physical batch size before DataParallel split.
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 50))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 5e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 8))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.get("amp", True))
    parser.add_argument("--grad_clip_norm", type=float, default=defaults.get("grad_clip_norm", 1.0))
    parser.add_argument("--lambda_ortho_max", type=float, default=defaults.get("lambda_ortho_max", 0.03))
    parser.add_argument("--lambda_ortho_warmup_epochs", type=int, default=defaults.get("lambda_ortho_warmup_epochs", 10))

    # Outputs.
    parser.add_argument("--output_dir", type=Path, default=defaults.get("output_dir", "outputs/fitfoodnet"))
    parser.add_argument("--exp_name", type=str, default=defaults.get("exp_name", "fitfoodnet"))
    parser.add_argument("--save_path", type=Path, default=defaults.get("save_path", None))
    parser.add_argument("--class_json", type=Path, default=defaults.get("class_json", None))
    parser.add_argument("--metrics_csv", type=Path, default=defaults.get("metrics_csv", None))
    parser.add_argument("--summary_json", type=Path, default=defaults.get("summary_json", None))
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

    output_dir = Path(args.output_dir)
    if args.save_path is None:
        args.save_path = output_dir / "fitfoodnet_best.pth"
    if args.class_json is None:
        args.class_json = output_dir / "class_indices.json"
    if args.metrics_csv is None:
        args.metrics_csv = output_dir / "metrics.csv"
    if args.summary_json is None:
        args.summary_json = output_dir / "summary.json"
    return args


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    autocast_enabled = args.amp and device.type == "cuda"

    train_bar = tqdm(loader, file=sys.stdout)
    for images, labels in train_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits, aux = model(images, return_attn=True)
            cls_loss = criterion(logits, labels)
            loss_ortho = compute_orthogonal_loss(aux["attn_weights"])
            weight = ortho_weight(
                epoch=epoch,
                max_weight=args.lambda_ortho_max,
                warmup_epochs=args.lambda_ortho_warmup_epochs,
            )
            loss = cls_loss + weight * loss_ortho

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = torch.max(logits, dim=1)[1]
        correct += torch.eq(preds, labels).sum().item()
        total += labels.size(0)
        train_bar.desc = f"Train Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}"

    return running_loss / max(len(loader), 1), correct / max(total, 1)


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
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"Using {device} device.")
    if gpu_count > 0:
        print(f"Detected {gpu_count} GPU(s): {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
    print(f"Total physical batch size: {args.batch_size}")

    train_dataset, val_dataset, idx_to_class = create_datasets(args)
    num_classes = len(idx_to_class)
    print(f"Dataset loaded: {len(train_dataset)} train images, {len(val_dataset)} val images.")
    print(f"Num classes: {num_classes}")

    args.class_json.parent.mkdir(parents=True, exist_ok=True)
    with args.class_json.open("w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=4, ensure_ascii=False)

    num_workers = min(os.cpu_count() or 1, args.batch_size if args.batch_size > 1 else 0, args.num_workers)
    print(f"Using {num_workers} dataloader workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
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
    )

    if gpu_count > 1:
        print(f"Enabling DataParallel across {gpu_count} GPU(s)")
        model = nn.DataParallel(model)
    model = model.to(device)

    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print(f"Trainable ratio: {(trainable_params / total_params) * 100:.2f} %")

    criterion = nn.CrossEntropyLoss()
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_precision", "val_recall", "val_f1", "best_acc"])

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and torch.cuda.is_available()))
    best_acc = 0.0
    best_f1 = 0.0
    total_peak_memory = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            scaler=scaler,
            args=args,
        )

        if epoch == 0 and torch.cuda.is_available():
            total_peak_memory = 0.0
            for i in range(gpu_count):
                peak_mem_gb = torch.cuda.max_memory_allocated(i) / (1024**3)
                print(f"GPU {i} peak VRAM: {peak_mem_gb:.2f} GB")
                total_peak_memory += peak_mem_gb
            print(f"Total peak VRAM usage: {total_peak_memory:.2f} GB")

        scheduler.step()

        val_acc, val_precision, val_recall, val_f1 = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            amp=args.amp,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, args.save_path)
            print(f"Best model updated: {args.save_path} (Val Acc = {best_acc:.4f})")

        with args.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_precision:.6f}",
                    f"{val_recall:.6f}",
                    f"{val_f1:.6f}",
                    f"{best_acc:.6f}",
                ]
            )

    summary = {
        "exp_name": args.exp_name,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "img_size": args.img_size,
        "num_queries": args.num_queries,
        "num_heads": args.num_heads,
        "hfta_bottleneck": args.hfta_bottleneck,
        "lambda_ortho_max": args.lambda_ortho_max,
        "lambda_ortho_warmup_epochs": args.lambda_ortho_warmup_epochs,
        "total_params_m": round(total_params / 1e6, 4),
        "trainable_params_m": round(trainable_params / 1e6, 4),
        "trainable_ratio_percent": round(trainable_params / total_params * 100, 4),
        "best_val_acc": round(best_acc, 6),
        "best_val_f1": round(best_f1, 6),
        "peak_memory_gb_sum": round(total_peak_memory, 4),
        "save_path": str(args.save_path),
        "metrics_csv": str(args.metrics_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Training completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
