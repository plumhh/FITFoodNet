#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-dataset shared-label evaluation:
VireoFood172-trained FITFoodNet -> ChineseFoodNet shared subset.

The model is NOT trained or fine-tuned on ChineseFoodNet.
Evaluation is restricted to the shared label space defined by a mapping CSV.
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_name(s: str) -> str:
    s = str(s).lower()
    s = s.replace("saut茅ed", "sauteed").replace("saut茅", "saute")
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def import_model_def(model_def_path: Path):
    spec = importlib.util.spec_from_file_location("model_def", str(model_def_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model definition from: {model_def_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_def"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "DINOv3Classifier"):
        raise AttributeError(f"{model_def_path} does not define DINOv3Classifier")
    return module.DINOv3Classifier


def clean_state_dict(state):
    # Support raw state_dict, {'state_dict': ...}, {'model': ...}, etc.
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict.")

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


def load_checkpoint(model, checkpoint_path: Path):
    # Load on CPU first to avoid CUDA OOM during deserialization.
    state = torch.load(str(checkpoint_path), map_location="cpu")
    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if len(missing) > 20 or len(unexpected) > 20:
        print("WARNING: Many missing/unexpected keys. Make sure --model_def_path matches the checkpoint training code.")
        print("First missing:", missing[:10])
        print("First unexpected:", unexpected[:10])


def read_mapping(mapping_csv: Path) -> List[Dict]:
    rows = []
    with mapping_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rows.append({
                "shared_idx": i,
                "vireo_model_idx": int(r["vireo_model_idx"]),
                "vireo_folder_id": int(r["vireo_folder_id"]),
                "vireo_class": r["vireo_class"],
                "chinesefoodnet_idx": int(r["chinesefoodnet_idx"]),
                "chinesefoodnet_cn": r.get("chinesefoodnet_cn", ""),
                "chinesefoodnet_en": r.get("chinesefoodnet_en", ""),
            })
    if not rows:
        raise RuntimeError(f"Empty mapping CSV: {mapping_csv}")
    return rows


def resolve_class_dir(root: Path, row: Dict, all_dirs_norm: Dict[str, Path]) -> Path:
    idx = row["chinesefoodnet_idx"]
    cn = row["chinesefoodnet_cn"]
    en = row["chinesefoodnet_en"]

    candidates = [
        str(idx),
        f"{idx:03d}",
        f"{idx:04d}",
        cn,
        en,
        en.lower(),
        en.replace(" ", "_"),
        en.lower().replace(" ", "_"),
        en.replace(" ", "-"),
        en.lower().replace(" ", "-"),
    ]

    for c in candidates:
        if c:
            p = root / c
            if p.exists() and p.is_dir():
                return p

    # normalized fallback against actual directory names
    norm_candidates = [norm_name(str(idx)), norm_name(cn), norm_name(en)]
    for nc in norm_candidates:
        if nc in all_dirs_norm:
            return all_dirs_norm[nc]

    return None


class SharedChineseFoodNetDataset(Dataset):
    def __init__(self, root: Path, mapping_rows: List[Dict], transform=None):
        self.root = root
        self.mapping_rows = mapping_rows
        self.transform = transform
        self.samples: List[Tuple[Path, int, Dict]] = []
        self.status = []

        all_dirs = [p for p in root.iterdir() if p.is_dir()]
        all_dirs_norm = {norm_name(p.name): p for p in all_dirs}

        for row in mapping_rows:
            class_dir = resolve_class_dir(root, row, all_dirs_norm)
            if class_dir is None:
                self.status.append((row, "NOT_FOUND", 0, ""))
                continue
            imgs = sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
            self.status.append((row, "OK", len(imgs), str(class_dir)))
            for img in imgs:
                self.samples.append((img, row["shared_idx"], row))

        if not self.samples:
            raise RuntimeError(f"No images loaded from {root}. Check directory names and mapping CSV.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, shared_label, row = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path} | {e}")
        if self.transform is not None:
            image = self.transform(image)
        return image, shared_label, str(path), row["chinesefoodnet_idx"], row["chinesefoodnet_en"]


def evaluate(model, loader, device, source_ids: List[int], shared_names: List[str], output_dir: Path):
    model.eval()
    source_ids_t = torch.tensor(source_ids, dtype=torch.long, device=device)

    y_true, y_pred = [], []
    pred_records = []

    with torch.no_grad():
        for images, labels, paths, target_ids, target_names in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                shared_logits = logits.index_select(dim=1, index=source_ids_t)
            preds = torch.argmax(shared_logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            probs = torch.softmax(shared_logits.float(), dim=1).cpu().numpy()
            pred_np = preds.cpu().numpy()
            label_np = labels.cpu().numpy()
            for i in range(len(paths)):
                pred_idx = int(pred_np[i])
                true_idx = int(label_np[i])
                pred_records.append({
                    "image_path": paths[i],
                    "true_shared_idx": true_idx,
                    "true_class": shared_names[true_idx],
                    "pred_shared_idx": pred_idx,
                    "pred_class": shared_names[pred_idx],
                    "confidence": float(probs[i, pred_idx]),
                    "correct": int(pred_idx == true_idx),
                })

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "num_classes": int(len(shared_names)),
        "num_images": int(len(y_true)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vireo_to_chinesefoodnet_shared_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    report = classification_report(y_true, y_pred, target_names=shared_names, zero_division=0, output_dict=True)
    with (output_dir / "vireo_to_chinesefoodnet_shared_class_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls_name in shared_names:
            row = report.get(cls_name, {})
            writer.writerow([cls_name, row.get("precision", 0), row.get("recall", 0), row.get("f1-score", 0), row.get("support", 0)])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(shared_names))))
    with (output_dir / "vireo_to_chinesefoodnet_shared_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + shared_names)
        for i, row in enumerate(cm):
            writer.writerow([shared_names[i]] + row.tolist())

    with (output_dir / "vireo_to_chinesefoodnet_shared_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_records[0].keys()))
        writer.writeheader()
        writer.writerows(pred_records)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser("VireoFood172 -> ChineseFoodNet shared-label evaluation")
    parser.add_argument("--model_def_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chinese_root", type=Path, required=True, help="Path to ChineseFoodNet split, e.g., .../ChineseFoodNet/test")
    parser.add_argument("--mapping_csv", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="dinov3_l")
    parser.add_argument("--num_classes", type=int, default=172)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, default=Path("./cross_dataset_vireo_to_chinesefoodnet"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mapping_rows = read_mapping(args.mapping_csv)
    print(f"Loaded mapping: {len(mapping_rows)} shared classes")
    for r in mapping_rows[:10]:
        print(f"  shared {r['shared_idx']:02d}: Vireo[{r['vireo_model_idx']}] {r['vireo_class']} <-> ChineseFoodNet[{r['chinesefoodnet_idx']}] {r['chinesefoodnet_en']}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SharedChineseFoodNetDataset(args.chinese_root, mapping_rows, transform=transform)
    print(f"Loaded ChineseFoodNet shared images: {len(dataset)}")
    print("Per-class directory status:")
    for row, status, count, path in dataset.status:
        print(f"  CF {row['chinesefoodnet_idx']:3d} | {row['chinesefoodnet_en'][:40]:40s} | {status:9s} | {count:5d} | {path}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    DINOv3Classifier = import_model_def(args.model_def_path)
    model = DINOv3Classifier(num_classes=args.num_classes, model_type=args.model_name)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)

    source_ids = [r["vireo_model_idx"] for r in mapping_rows]
    shared_names = [r["vireo_class"] for r in mapping_rows]
    metrics = evaluate(model, loader, device, source_ids, shared_names, args.output_dir)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))
    print("Saved outputs to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
