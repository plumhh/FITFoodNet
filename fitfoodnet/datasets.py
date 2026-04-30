"""Dataset utilities for FoodX-251 and VireoFood172."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CsvImageDataset(Dataset):
    """Dataset that reads image names and labels from a CSV file."""

    def __init__(
        self,
        csv_file: str | Path,
        image_dirs: list[str | Path],
        transform=None,
        img_key: str = "img_name",
        label_key: str = "label",
        max_retry: int = 10,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.image_dirs = [Path(p) for p in image_dirs]
        self.transform = transform
        self.max_retry = max_retry
        self.samples: list[tuple[str, int]] = []

        with self.csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row[img_key].strip()
                label = int(row[label_key])
                self.samples.append((img_name, label))

        if not self.samples:
            raise RuntimeError(f"CSV file is empty: {self.csv_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, img_name: str) -> Path:
        for image_dir in self.image_dirs:
            candidate = image_dir / img_name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find image '{img_name}' in {[str(p) for p in self.image_dirs]}"
        )

    def __getitem__(self, idx: int, retry_count: int = 0):
        if retry_count > self.max_retry:
            raise RuntimeError(f"Too many failed image reads. Last index: {idx}")

        img_name, label = self.samples[idx]
        try:
            image = Image.open(self._resolve_image_path(img_name)).convert("RGB")
        except Exception:
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx, retry_count + 1)

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TxtImageDataset(Dataset):
    """Dataset that reads image names and labels from a TXT list file."""

    def __init__(
        self,
        txt_file: str | Path,
        image_dirs: list[str | Path],
        transform=None,
        max_retry: int = 10,
    ) -> None:
        self.txt_file = Path(txt_file)
        self.image_dirs = [Path(p) for p in image_dirs]
        self.transform = transform
        self.max_retry = max_retry
        self.samples: list[tuple[str, int]] = []

        with self.txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], int(parts[1])))

        if not self.samples:
            raise RuntimeError(f"TXT file is empty or malformed: {self.txt_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, img_name: str) -> Path:
        for image_dir in self.image_dirs:
            candidate = image_dir / img_name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find image '{img_name}' in {[str(p) for p in self.image_dirs]}"
        )

    def __getitem__(self, idx: int, retry_count: int = 0):
        if retry_count > self.max_retry:
            raise RuntimeError(f"Too many failed image reads. Last index: {idx}")

        img_name, label = self.samples[idx]
        try:
            image = Image.open(self._resolve_image_path(img_name)).convert("RGB")
        except Exception:
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx, retry_count + 1)

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


class JPEGCompression:
    """Apply lossy JPEG compression at the PIL-image stage."""

    def __init__(self, quality: int = 30) -> None:
        if not (1 <= quality <= 95):
            raise ValueError("jpeg_quality must be in [1, 95]")
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_eval_transform(
    img_size: int,
    perturbation: str = "clean",
    blur_kernel: int = 7,
    blur_sigma: float = 1.5,
    jpeg_quality: int = 30,
):
    """Build validation/evaluation transform with optional perturbation."""

    if perturbation not in {"clean", "blur", "jpeg"}:
        raise ValueError("perturbation must be one of: clean, blur, jpeg")
    if blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be odd")

    ops = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
    ]
    if perturbation == "blur":
        ops.append(transforms.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma))
    elif perturbation == "jpeg":
        ops.append(JPEGCompression(quality=jpeg_quality))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def parse_class_list(class_list_path: str | Path | None) -> dict[int, str]:
    if class_list_path is None:
        return {}

    path = Path(class_list_path)
    if not path.exists():
        return {}

    class_map: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(" ", 1)
            class_map[int(idx)] = name
    return class_map


def maybe_convert_one_based_labels(dataset: Dataset) -> None:
    if not hasattr(dataset, "samples"):
        return
    labels = sorted({label for _, label in dataset.samples})
    if labels and labels[0] == 1 and labels[-1] == len(labels):
        dataset.samples = [(img, label - 1) for img, label in dataset.samples]


def create_datasets(args: Any):
    train_transform, val_transform = build_transforms(args.img_size)

    if args.dataset_type == "imagefolder":
        train_dataset = datasets.ImageFolder(root=str(args.train_dir), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(args.val_dir), transform=val_transform)
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        return train_dataset, val_dataset, idx_to_class

    image_dirs = [Path(p) for p in args.image_dirs]

    if args.dataset_type == "csv":
        train_dataset = CsvImageDataset(args.train_csv, image_dirs, transform=train_transform)
        val_dataset = CsvImageDataset(args.val_csv, image_dirs, transform=val_transform)
        class_map = parse_class_list(getattr(args, "class_list", None))
        if class_map:
            return train_dataset, val_dataset, class_map
    elif args.dataset_type == "txt":
        train_dataset = TxtImageDataset(args.train_txt, image_dirs, transform=train_transform)
        val_dataset = TxtImageDataset(args.val_txt, image_dirs, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    maybe_convert_one_based_labels(train_dataset)
    maybe_convert_one_based_labels(val_dataset)
    labels = sorted({label for _, label in train_dataset.samples})
    idx_to_class = {idx: str(idx) for idx in labels}
    return train_dataset, val_dataset, idx_to_class


def create_eval_dataset(args: Any, transform=None):
    """Create only the validation/evaluation dataset."""

    if transform is None:
        transform = build_eval_transform(args.img_size)

    if args.dataset_type == "imagefolder":
        return datasets.ImageFolder(root=str(args.val_dir), transform=transform)

    image_dirs = [Path(p) for p in args.image_dirs]
    if args.dataset_type == "csv":
        return CsvImageDataset(args.val_csv, image_dirs, transform=transform)
    if args.dataset_type == "txt":
        dataset = TxtImageDataset(args.val_txt, image_dirs, transform=transform)
        maybe_convert_one_based_labels(dataset)
        return dataset
    raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")
