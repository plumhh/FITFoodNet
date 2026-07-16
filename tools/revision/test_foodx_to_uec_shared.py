"""
Cross-dataset robustness test: FoodX-251 trained FITFoodNet -> UEC-Food256 shared-category subset.

This script does NOT train or fine-tune on UEC-Food256.
It loads a FoodX-251 checkpoint and evaluates it on UEC shared classes defined by foodx_uec_final_mapping.csv.

Expected UEC directory example:
  UEC-Food256_Ready/val/1/*.jpg
  UEC-Food256_Ready/val/2/*.jpg
  ...

Run example:
  python test_foodx_to_uec_shared.py \
    --model_def_path /home/amax/4t/lzh/your_train_script.py \
    --checkpoint /home/amax/4t/lzh/your_project/models/train_FrequencydomainAttention_newIACA_best.pth \
    --uec_root /home/amax/4t/lzh/data/UEC-Food256_Ready/val \
    --mapping_csv /home/amax/4t/lzh/foodx_uec_final_mapping.csv \
    --batch_size 64 \
    --output_dir ./cross_dataset_foodx_to_uec
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model_class(model_def_path: Path):
    if not model_def_path.exists():
        raise FileNotFoundError(f"model_def_path not found: {model_def_path}")
    spec = importlib.util.spec_from_file_location("foodx_model_def", str(model_def_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "DINOv3Classifier"):
        raise AttributeError("DINOv3Classifier was not found in model_def_path")
    return module.DINOv3Classifier


def load_state_dict_safely(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # handle DataParallel checkpoints if needed
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print("First missing keys:", missing[:10])
    if unexpected:
        print("First unexpected keys:", unexpected[:10])


def read_mapping(mapping_csv: Path) -> List[Dict]:
    if not mapping_csv.exists():
        raise FileNotFoundError(f"mapping_csv not found: {mapping_csv}")
    rows = []
    with mapping_csv.open("r", encoding="utf-utf-8-sig".replace("utf-utf", "utf"), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "shared_idx": int(row["shared_idx"]),
                "foodx_model_idx": int(row["foodx_model_idx"]),
                "foodx_class": row["foodx_class"],
                "uec_id": str(row["uec_id"]),
                "uec_class": row["uec_class"],
            })
    rows = sorted(rows, key=lambda r: r["shared_idx"])
    assert [r["shared_idx"] for r in rows] == list(range(len(rows))), "shared_idx must be continuous from 0"
    return rows


def normalize_dir_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("&", "and")


class UECSharedDataset(Dataset):
    def __init__(self, uec_root: Path, mapping_rows: List[Dict], transform=None):
        self.uec_root = uec_root
        self.mapping_rows = mapping_rows
        self.transform = transform
        self.samples: List[Tuple[Path, int, str, str]] = []
        self.class_dir_status = []

        if not self.uec_root.exists():
            raise FileNotFoundError(f"UEC root not found: {self.uec_root}")

        for r in mapping_rows:
            shared_idx = r["shared_idx"]
            uec_id = str(r["uec_id"])
            uec_class = r["uec_class"]

            # Most likely structure: val/<uec_id>/image.jpg
            candidates = [
                self.uec_root / uec_id,
                self.uec_root / uec_class,
                self.uec_root / normalize_dir_name(uec_class),
                self.uec_root / uec_class.replace(" ", "_"),
            ]

            class_dir = None
            for c in candidates:
                if c.exists() and c.is_dir():
                    class_dir = c
                    break

            if class_dir is None:
                self.class_dir_status.append((uec_id, uec_class, "NOT_FOUND", 0, ""))
                continue

            imgs = sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
            self.class_dir_status.append((uec_id, uec_class, "OK", len(imgs), str(class_dir)))

            for p in imgs:
                self.samples.append((p, shared_idx, uec_id, uec_class))

        if not self.samples:
            raise RuntimeError("No images found for the shared UEC classes. Check uec_root and mapping_csv.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, shared_idx, uec_id, uec_class = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, shared_idx, str(path), uec_id, uec_class


@torch.no_grad()
def evaluate_shared(model, loader, device, source_indices: List[int], class_names: List[str], output_dir: Path, use_amp: bool):
    model.eval()
    source_indices_tensor = torch.tensor(source_indices, dtype=torch.long, device=device)

    all_labels = []
    all_preds = []
    pred_rows = []

    for images, labels, paths, uec_ids, uec_classes in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
            shared_logits = logits.index_select(dim=1, index=source_indices_tensor)
            preds = shared_logits.argmax(dim=1)
            confs = shared_logits.softmax(dim=1).max(dim=1).values

        labels_cpu = labels.cpu().tolist()
        preds_cpu = preds.cpu().tolist()
        confs_cpu = confs.cpu().tolist()

        all_labels.extend(labels_cpu)
        all_preds.extend(preds_cpu)

        for path, y, pred, conf, uec_id, uec_class in zip(paths, labels_cpu, preds_cpu, confs_cpu, uec_ids, uec_classes):
            pred_rows.append({
                "image_path": path,
                "uec_id": uec_id,
                "uec_class": uec_class,
                "true_shared_idx": y,
                "true_foodx_class": class_names[y],
                "pred_shared_idx": pred,
                "pred_foodx_class": class_names[pred],
                "confidence": f"{conf:.6f}",
                "correct": int(y == pred),
            })

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_csv = output_dir / "foodx_to_uec_shared_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    # Save per-class report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_csv = output_dir / "foodx_to_uec_shared_class_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls_name in class_names:
            r = report[cls_name]
            writer.writerow([cls_name, r["precision"], r["recall"], r["f1-score"], r["support"]])

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm_csv = output_dir / "foodx_to_uec_shared_confusion_matrix.csv"
    with cm_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for cls_name, row in zip(class_names, cm):
            writer.writerow([cls_name] + row.tolist())

    metrics = {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "num_classes": len(class_names),
        "num_images": len(all_labels),
    }
    metrics_json = output_dir / "foodx_to_uec_shared_metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n========== FoodX-251 -> UEC-Food256 Shared-Class Evaluation ==========")
    print(f"Images: {len(all_labels)}")
    print(f"Shared classes: {len(class_names)}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro-Precision: {precision * 100:.2f}%")
    print(f"Macro-Recall: {recall * 100:.2f}%")
    print(f"Macro-F1: {f1 * 100:.2f}%")
    print("Saved outputs to:", output_dir.resolve())

    return metrics


def parse_args():
    p = argparse.ArgumentParser("FoodX-trained FITFoodNet direct evaluation on UEC shared classes")
    p.add_argument("--model_def_path", type=Path, required=True, help="Path to the training .py containing DINOv3Classifier")
    p.add_argument("--checkpoint", type=Path, required=True, help="FoodX-251 trained checkpoint .pth")
    p.add_argument("--uec_root", type=Path, required=True, help="UEC-Food256 target test root, e.g. UEC-Food256_Ready/val")
    p.add_argument("--mapping_csv", type=Path, required=True, help="foodx_uec_final_mapping.csv")
    p.add_argument("--model_name", type=str, default="dinov3_l", choices=["dinov3_s", "dinov3_b", "dinov3_l"])
    p.add_argument("--num_classes", type=int, default=251, help="FoodX source classifier classes")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", type=Path, default=Path("./cross_dataset_foodx_to_uec"))
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mapping_rows = read_mapping(args.mapping_csv)
    source_indices = [r["foodx_model_idx"] for r in mapping_rows]
    class_names = [r["foodx_class"] for r in mapping_rows]

    print(f"Loaded mapping: {len(mapping_rows)} shared classes")
    print("First mappings:")
    for r in mapping_rows[:10]:
        print(f"  shared {r['shared_idx']:02d}: FoodX[{r['foodx_model_idx']}] {r['foodx_class']}  <->  UEC[{r['uec_id']}] {r['uec_class']}")

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = UECSharedDataset(args.uec_root, mapping_rows, transform=val_transform)
    print(f"Loaded UEC shared images: {len(dataset)}")
    print("Per-class directory status:")
    for uec_id, uec_class, status, n, class_dir in dataset.class_dir_status:
        print(f"  UEC {uec_id:>3} | {uec_class:<35} | {status:<9} | {n:>5} | {class_dir}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    DINOv3Classifier = load_model_class(args.model_def_path)
    model = DINOv3Classifier(num_classes=args.num_classes, model_type=args.model_name)
    load_state_dict_safely(model, args.checkpoint, device)
    model = model.to(device)

    evaluate_shared(
        model=model,
        loader=loader,
        device=device,
        source_indices=source_indices,
        class_names=class_names,
        output_dir=args.output_dir,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
