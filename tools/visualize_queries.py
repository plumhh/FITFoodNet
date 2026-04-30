"""Visualize occlusion maps and top-k dynamic-query heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageEnhance
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitfoodnet.datasets import create_eval_dataset  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Visualize FITFoodNet query heatmaps")
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

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--class_json", type=Path, default=defaults.get("class_json", None))
    parser.add_argument("--image_path", type=Path, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--output_root", type=Path, required=True)

    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--occ_size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--brightness", type=float, default=0.60)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--smooth_kernel", type=int, default=15)
    parser.add_argument("--smooth_sigma", type=float, default=4.0)
    parser.add_argument("--low_thresh", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--pct", type=float, default=98.0)
    parser.add_argument("--seed", type=int, default=42)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_class_names(path: Path | None, num_classes: int) -> dict[str, str]:
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {str(i): str(i) for i in range(num_classes)}


def normalize_map(x: np.ndarray, gamma: float = 0.90, pct: float = 98.0, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0)
    if float(x.max()) <= eps:
        return np.zeros_like(x, dtype=np.float32)
    hi = np.percentile(x, pct)
    if hi <= eps:
        hi = float(x.max())
    x = np.clip(x / (hi + eps), 0, 1)
    x = np.power(x, gamma)
    return x.astype(np.float32)


def simple_colormap(norm_map: np.ndarray) -> np.ndarray:
    x = np.clip(norm_map.astype(np.float32), 0, 1)
    r = np.clip(1.8 * x - 0.3, 0, 1)
    g = np.clip(1.8 * (1 - np.abs(x - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - x), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def smooth_map_np(x: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size <= 1:
        return x.astype(np.float32)
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    t = torch.tensor(x, dtype=torch.float32).view(1, 1, *x.shape)
    return F.conv2d(t, kernel, padding=kernel_size // 2).squeeze().numpy().astype(np.float32)


def postprocess_map(
    x: np.ndarray,
    out_hw: tuple[int, int],
    smooth_kernel: int,
    smooth_sigma: float,
    low_thresh: float,
    gamma: float,
    pct: float,
) -> np.ndarray:
    x = smooth_map_np(x, smooth_kernel, smooth_sigma)
    t = torch.tensor(x, dtype=torch.float32).view(1, 1, *x.shape)
    t = F.interpolate(t, size=out_hw, mode="bicubic", align_corners=False)
    x = normalize_map(t.squeeze().numpy(), gamma=gamma, pct=pct)
    x = np.maximum(x - low_thresh, 0)
    if x.max() > 1e-8:
        x = x / x.max()
    return x.astype(np.float32)


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float, brightness: float) -> Image.Image:
    pil_img = pil_img.convert("RGB")
    height, width = pil_img.size[1], pil_img.size[0]
    if heatmap.shape != (height, width):
        heatmap = np.array(
            Image.fromarray((normalize_map(heatmap) * 255).astype(np.uint8)).resize((width, height), resample=Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
    heatmap = normalize_map(heatmap)
    heat_rgb = simple_colormap(heatmap)
    img_np = np.array(ImageEnhance.Brightness(pil_img).enhance(brightness)).astype(np.float32)
    alpha_map = (heatmap[..., None] * alpha).astype(np.float32)
    out = img_np * (1.0 - alpha_map) + heat_rgb.astype(np.float32) * alpha_map
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def add_title(img: Image.Image, title: str, pad_h: int = 28) -> Image.Image:
    img = img.convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + pad_h), color=(255, 255, 255))
    canvas.paste(img, (0, pad_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    return canvas


def make_row(images: list[Image.Image], titles: list[str]) -> Image.Image:
    titled = [add_title(image, title) for image, title in zip(images, titles)]
    cell_w = max(image.width for image in titled)
    cell_h = max(image.height for image in titled)
    panel = Image.new("RGB", (len(titled) * cell_w, cell_h), color=(255, 255, 255))
    for i, image in enumerate(titled):
        x = i * cell_w + (cell_w - image.width) // 2
        y = (cell_h - image.height) // 2
        panel.paste(image, (x, y))
    return panel


def make_preprocess(img_size: int):
    pil_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(img_size)])
    tensor_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return pil_tf, tensor_tf


def dim_normalized_region(region: torch.Tensor, brightness: float) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=region.device, dtype=region.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=region.device, dtype=region.dtype).view(1, 3, 1, 1)
    pixel = region * std + mean
    pixel = torch.clamp(pixel * brightness, 0, 1)
    return (pixel - mean) / std


@torch.no_grad()
def compute_occlusion_map(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    occ_size: int,
    stride: int,
    brightness: float,
) -> np.ndarray:
    _, _, height, width = x.shape
    base_score = float(model(x)[0, target_class].item())
    heat = torch.zeros((height, width), dtype=torch.float32, device=x.device)
    count = torch.zeros((height, width), dtype=torch.float32, device=x.device)

    ys = list(range(0, max(height - occ_size + 1, 1), stride))
    xs = list(range(0, max(width - occ_size + 1, 1), stride))
    if ys[-1] != height - occ_size:
        ys.append(height - occ_size)
    if xs[-1] != width - occ_size:
        xs.append(width - occ_size)

    for y in ys:
        for x0 in xs:
            x_occ = x.clone()
            region = x_occ[:, :, y : y + occ_size, x0 : x0 + occ_size]
            x_occ[:, :, y : y + occ_size, x0 : x0 + occ_size] = dim_normalized_region(region, brightness)
            occ_score = float(model(x_occ)[0, target_class].item())
            drop = max(0.0, base_score - occ_score)
            heat[y : y + occ_size, x0 : x0 + occ_size] += drop
            count[y : y + occ_size, x0 : x0 + occ_size] += 1.0

    return (heat / (count + 1e-8)).detach().cpu().numpy()


@torch.no_grad()
def get_query_maps(model: torch.nn.Module, x: torch.Tensor) -> dict[str, Any]:
    logits, aux = model(x, return_attn=True)
    pred = int(logits.argmax(dim=1).item())
    pred_prob = float(torch.softmax(logits, dim=-1)[0, pred].item())
    attn = aux["attn_weights"][0].detach().float().cpu()
    query_weights = aux["query_weights"][0].detach().float().cpu()
    side = int(math.sqrt(attn.shape[1]))
    if side * side != attn.shape[1]:
        raise RuntimeError(f"Patch token count is not square: {attn.shape[1]}")
    order = torch.argsort(query_weights, descending=True).tolist()
    return {
        "pred": pred,
        "pred_prob": pred_prob,
        "attn": attn,
        "query_weights": query_weights,
        "query_order": order,
        "side": side,
    }


def resolve_dataset_image(dataset, sample_index: int) -> tuple[Path, int]:
    sample = dataset.samples[sample_index]
    img_ref, label = sample[0], int(sample[1])
    if isinstance(img_ref, str) and Path(img_ref).exists():
        return Path(img_ref), label
    if hasattr(dataset, "_resolve_image_path"):
        return dataset._resolve_image_path(img_ref), label
    return Path(img_ref), label


def main() -> None:
    args = complete_paths(parse_args())
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = create_eval_dataset(args, transform=None)
    num_classes = len(eval_dataset.classes) if hasattr(eval_dataset, "classes") else len({label for _, label in eval_dataset.samples})
    class_names = load_class_names(args.class_json, num_classes)

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
    model.eval()

    pil_tf, tensor_tf = make_preprocess(args.img_size)
    rows = []

    if args.image_path is not None:
        selected = [(args.image_path, -1)]
    else:
        selected = []
        end = min(args.sample_index + args.max_samples, len(eval_dataset))
        for idx in range(args.sample_index, end):
            selected.append(resolve_dataset_image(eval_dataset, idx))

    for index, (image_path, label) in enumerate(selected, start=1):
        pil_raw = Image.open(image_path).convert("RGB")
        pil_crop = pil_tf(pil_raw)
        x = tensor_tf(pil_raw).unsqueeze(0).to(device)

        info = get_query_maps(model, x)
        pred = info["pred"]
        pred_name = class_names.get(str(pred), str(pred))
        occ_raw = compute_occlusion_map(
            model,
            x,
            target_class=pred,
            occ_size=args.occ_size,
            stride=args.stride,
            brightness=args.brightness,
        )
        occ_map = postprocess_map(
            occ_raw,
            out_hw=(pil_crop.size[1], pil_crop.size[0]),
            smooth_kernel=args.smooth_kernel,
            smooth_sigma=args.smooth_sigma,
            low_thresh=args.low_thresh,
            gamma=args.gamma,
            pct=args.pct,
        )
        occ_overlay = overlay_heatmap(pil_crop, occ_map, alpha=args.alpha, brightness=args.brightness)

        query_images = []
        query_titles = []
        for rank, query_idx in enumerate(info["query_order"][: args.topk], start=1):
            raw_map = info["attn"][query_idx].view(info["side"], info["side"]).numpy()
            query_map = postprocess_map(
                raw_map,
                out_hw=(pil_crop.size[1], pil_crop.size[0]),
                smooth_kernel=args.smooth_kernel,
                smooth_sigma=args.smooth_sigma,
                low_thresh=args.low_thresh,
                gamma=args.gamma,
                pct=args.pct,
            )
            query_images.append(overlay_heatmap(pil_crop, query_map, alpha=args.alpha, brightness=args.brightness))
            weight = float(info["query_weights"][query_idx])
            query_titles.append(f"Top-{rank} query | q={query_idx}, w={weight:.3f}")

        panel_images = [pil_crop, occ_overlay] + query_images
        panel_titles = [f"Original | pred={pred_name}", f"Occlusion | {args.occ_size}/{args.stride}"] + query_titles
        panel = make_row(panel_images, panel_titles)

        safe_name = image_path.stem.replace(" ", "_")
        out_dir = args.output_root / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)
        panel.save(out_dir / "query_occlusion_panel.jpg")
        pil_crop.save(out_dir / "original.jpg")
        occ_overlay.save(out_dir / "occlusion_overlay.jpg")

        rows.append(
            {
                "image": str(image_path),
                "label": label,
                "pred": pred,
                "pred_name": pred_name,
                "pred_prob": info["pred_prob"],
                "query_order": info["query_order"],
                "query_weights": info["query_weights"].tolist(),
                "panel": str(out_dir / "query_occlusion_panel.jpg"),
            }
        )
        print(f"[{index}/{len(selected)}] saved {out_dir / 'query_occlusion_panel.jpg'}")

    with (args.output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "pred", "pred_name", "pred_prob", "panel"])
        for row in rows:
            writer.writerow([row["image"], row["label"], row["pred"], row["pred_name"], f"{row['pred_prob']:.6f}", row["panel"]])


if __name__ == "__main__":
    main()

