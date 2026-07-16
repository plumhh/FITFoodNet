import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms


PROJECT_ROOT = Path("/home/amax/4t/lzh/DINOv3")
DEFAULT_DATA_ROOT = Path("/home/amax/4t/lzh/data/VireoFood172")
DEFAULT_DINOV3_REPO = Path("/home/amax/.cache/torch/hub/facebookresearch_dinov3_main")
DEFAULT_DINOV3_WEIGHT = PROJECT_ROOT / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "hfta_frequency_vis_fixed"


# ============================================================
# Dataset helpers
# ============================================================
def read_txt_list(txt_file: Path) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with txt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                samples.append((parts[0], int(parts[1])))
    if not samples:
        raise RuntimeError(f"TXT file is empty or malformed: {txt_file}")
    return samples


def resolve_image_path(img_name: str, image_dirs: List[Path]) -> Path:
    p = Path(img_name)
    if p.exists():
        return p
    for image_dir in image_dirs:
        candidate = image_dir / img_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find image '{img_name}' in: {[str(x) for x in image_dirs]}"
    )


def choose_image(args) -> Tuple[Path, Optional[int]]:
    if args.image is not None:
        return args.image, None

    if args.dataset_type == "imagefolder":
        val_dir = args.data_root / "val"
        ds = datasets.ImageFolder(root=str(val_dir))
        if not ds.samples:
            raise RuntimeError(f"No images found under {val_dir}")
        index = max(0, min(args.sample_index, len(ds.samples) - 1))
        image_path, label = ds.samples[index]
        return Path(image_path), int(label)

    txt_val = args.data_root / "val_list.txt"
    samples = read_txt_list(txt_val)
    index = max(0, min(args.sample_index, len(samples) - 1))
    img_name, label = samples[index]
    image_dirs = [
        args.data_root,
        args.data_root / "ready_chinese_food",
        args.data_root / "images",
        args.data_root / "val",
        args.data_root / "train",
    ]
    return resolve_image_path(img_name, image_dirs), int(label)


def infer_num_classes_from_checkpoint(ckpt: Dict[str, torch.Tensor]) -> Optional[int]:
    for k in ["head.classifier.4.weight", "module.head.classifier.4.weight"]:
        if k in ckpt:
            return int(ckpt[k].shape[0])
    return None


def auto_find_checkpoint(project_root: Path) -> Path:
    candidates = [
        project_root / "models" / "train_FITFoodNet_Vireo172_best.pth",
        project_root / "improve-query-HFTA" / "models" / "train_FITFoodNet_Vireo172_best.pth",
        project_root / "A_revise" / "models" / "train_FITFoodNet_Vireo172_best.pth",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = list(project_root.rglob("train_FITFoodNet_Vireo172_best.pth"))
    if matches:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0]

    raise FileNotFoundError("Could not find train_FITFoodNet_Vireo172_best.pth. Use --checkpoint.")


# ============================================================
# Model: matches the uploaded old VireoFood172 training code
# ============================================================
class DynamicFrequencyTextureAdapter(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim=64, hw_shape=(14, 14)):
        super().__init__()
        self.hw_shape = hw_shape
        self.bottleneck_dim = bottleneck_dim

        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()

        freq_elements = hw_shape[0] * (hw_shape[1] // 2 + 1) * 2 * bottleneck_dim
        self.mask_generator = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, freq_elements)
        )

        nn.init.zeros_(self.mask_generator[-1].weight)
        nn.init.zeros_(self.mask_generator[-1].bias)

        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.zeros(1))

        self.capture_spectrum = False
        self.last_fft_before_full = None
        self.last_fft_after_full = None
        self.last_delta_full = None
        self.last_mask_deviation = None
        self.last_scale_value = None

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.hw_shape
        orig_dtype = x.dtype

        x_down = self.act(self.down_proj(x))
        global_context = x_down.mean(dim=1)

        raw_mask = self.mask_generator(global_context)
        dynamic_mask = torch.tanh(raw_mask).to(torch.float32)
        dynamic_mask = dynamic_mask.view(B, -1, H, W // 2 + 1, 2)

        x_spatial = x_down.transpose(1, 2).view(B, -1, H, W).to(torch.float32)
        fft_x_r = torch.fft.rfft2(x_spatial, norm="ortho")

        weight_complex = torch.view_as_complex(dynamic_mask) + 1.0
        fft_x_enhanced_r = fft_x_r * weight_complex
        x_spatial_enhanced = torch.fft.irfft2(fft_x_enhanced_r, s=(H, W), norm="ortho")

        if self.capture_spectrum:
            with torch.no_grad():
                # Full 14x14 spectra are much easier to display than raw rFFT 14x8.
                fft_before_full = torch.fft.fftshift(
                    torch.fft.fft2(x_spatial, norm="ortho"), dim=(-2, -1)
                )
                fft_after_full = torch.fft.fftshift(
                    torch.fft.fft2(x_spatial_enhanced.float(), norm="ortho"), dim=(-2, -1)
                )

                before = torch.abs(fft_before_full).detach().cpu()
                after = torch.abs(fft_after_full).detach().cpu()
                delta = torch.abs(after - before)

                # This is the learned modulation itself: |M(f)-1|.
                mask_dev = torch.abs(weight_complex - 1.0).detach().cpu()

                self.last_fft_before_full = before
                self.last_fft_after_full = after
                self.last_delta_full = delta
                self.last_mask_deviation = mask_dev
                self.last_scale_value = float(self.scale.detach().cpu().item())

        x_seq = x_spatial_enhanced.to(orig_dtype).flatten(2).transpose(1, 2)
        out = self.up_proj(x_seq)
        return x + out * self.scale


class BlockWithFreqAdapter(nn.Module):
    def __init__(self, original_block, adapter):
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def _process_single_tensor(self, t):
        H, W = self.adapter.hw_shape
        num_patches = H * W
        other_tokens = t[:, :-num_patches, :]
        patch_tokens = t[:, -num_patches:, :]
        enhanced_patch_tokens = self.adapter(patch_tokens)
        return torch.cat([other_tokens, enhanced_patch_tokens], dim=1)

    def forward(self, x, *args, **kwargs):
        out = self.original_block(x, *args, **kwargs)
        if isinstance(out, (list, tuple)):
            return type(out)([self._process_single_tensor(t) for t in out])
        return self._process_single_tensor(out)


class FrequencyCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        Q_f = torch.fft.rfft(Q.float(), dim=-1)
        K_f = torch.fft.rfft(K.float(), dim=-1)
        V_f = torch.fft.rfft(V.float(), dim=-1)

        Q_mag = torch.nn.functional.normalize(torch.abs(Q_f), p=2, dim=-1)
        K_mag = torch.nn.functional.normalize(torch.abs(K_f), p=2, dim=-1)

        attn = torch.matmul(Q_mag, K_mag.transpose(-2, -1))
        attn = attn / self.temperature.clamp(min=1e-3)
        attn = torch.softmax(attn.float(), dim=-1).to(Q_mag.dtype)

        out_f = attn.to(V_f.dtype) @ V_f
        out = torch.fft.irfft(out_f, n=self.head_dim, dim=-1)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out), attn.mean(dim=1)


class IngredientAwareHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_queries=8, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.cross_attn = FrequencyCrossAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

        fusion_dim = (num_queries + 1) * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, cls_token, patch_tokens, return_attn=False):
        B = patch_tokens.shape[0]
        q = self.query_embed.expand(B, -1, -1)
        attn_output, attn_weights = self.cross_attn(q, patch_tokens, patch_tokens)
        local_features = self.norm(q + attn_output)
        local_features = local_features.reshape(B, -1)
        logits = self.classifier(torch.cat([cls_token, local_features], dim=1))
        if return_attn:
            return logits, attn_weights
        return logits


class DINOv3Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        print("========== Loading old Vireo FITFoodNet Architecture ==========")
        self.backbone = torch.hub.load(
            str(DEFAULT_DINOV3_REPO),
            "dinov3_vitl16",
            source="local",
            pretrained=False
        )

        print(f"Loading DINOv3-L weights from: {DEFAULT_DINOV3_WEIGHT}")
        try:
            state_dict = torch.load(DEFAULT_DINOV3_WEIGHT, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(DEFAULT_DINOV3_WEIGHT, map_location="cpu")
        self.backbone.load_state_dict(state_dict)

        embed_dim = 1024
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.insert_layers = [5, 11, 17, 23]
        for layer_idx in self.insert_layers:
            original_block = self.backbone.blocks[layer_idx]
            adapter = DynamicFrequencyTextureAdapter(
                embed_dim=embed_dim,
                bottleneck_dim=64,
                hw_shape=(14, 14)
            )
            self.backbone.blocks[layer_idx] = BlockWithFreqAdapter(original_block, adapter)
            print(f"Inserted HFTA at Block {layer_idx}")

        self.head = IngredientAwareHead(embed_dim=embed_dim, num_classes=num_classes, num_queries=8)

    def forward(self, x, return_attn=False):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        return self.head(cls_token, patch_tokens, return_attn=return_attn)

    def get_adapter(self, layer_idx: int):
        return self.backbone.blocks[layer_idx].adapter


# ============================================================
# Loading and plotting
# ============================================================
def load_checkpoint(model: nn.Module, checkpoint_path: Path, device) -> Tuple[int, int]:
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    cleaned = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print("First missing keys:", missing[:20])
    if unexpected:
        print("First unexpected keys:", unexpected[:20])

    # Important diagnostic: if these are missing, HFTA is not loaded.
    missing_hfta = [k for k in missing if "adapter" in k]
    unexpected_hfta = [k for k in unexpected if "adapter" in k]
    print(f"Missing HFTA keys: {len(missing_hfta)}")
    print(f"Unexpected HFTA keys: {len(unexpected_hfta)}")
    return len(missing_hfta), len(unexpected_hfta)


def prepare_input(image_path: Path):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    raw = Image.open(image_path).convert("RGB")
    return raw, tfm(raw).unsqueeze(0)


def robust_norm(arr: np.ndarray, low=1.0, high=99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.log1p(arr)
    lo, hi = np.percentile(arr, [low, high])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr


def to_map(x: torch.Tensor) -> np.ndarray:
    return robust_norm(x[0].mean(dim=0).detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description="Fixed HFTA frequency visualization.")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset_type", type=str, default="imagefolder", choices=["imagefolder", "txt"])
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--layer_idx", type=int, default=23, choices=[5, 11, 17, 23])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="hfta_frequency_fixed")
    parser.add_argument("--all_layers", action="store_true", help="Capture and save all HFTA layers.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint if args.checkpoint is not None else auto_find_checkpoint(args.project_root)
    print(f"Using checkpoint: {checkpoint}")

    raw_ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        ckpt_for_shape = raw_ckpt["state_dict"]
    elif isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        ckpt_for_shape = raw_ckpt["model"]
    else:
        ckpt_for_shape = raw_ckpt

    num_classes = infer_num_classes_from_checkpoint(ckpt_for_shape) or 172
    print(f"Inferred num_classes: {num_classes}")

    image_path, label = choose_image(args)
    print(f"Using image: {image_path}")
    print(f"Label: {label}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3Classifier(num_classes=num_classes)
    missing_hfta, unexpected_hfta = load_checkpoint(model, checkpoint, device)
    model = model.to(device)
    model.eval()

    layers = [5, 11, 17, 23] if args.all_layers else [args.layer_idx]
    for li in layers:
        model.get_adapter(li).capture_spectrum = True

    raw_img, x = prepare_input(image_path)
    x = x.to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            conf, pred = torch.max(prob, dim=1)

    pred_idx = int(pred.item())
    conf_value = float(conf.item())

    # If all_layers, save a compact layer diagnostic figure.
    if args.all_layers:
        fig, axes = plt.subplots(len(layers), 4, figsize=(12, 10), dpi=300)
        if len(layers) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, li in enumerate(layers):
            adapter = model.get_adapter(li)
            before = adapter.last_fft_before_full
            after = adapter.last_fft_after_full
            delta = adapter.last_delta_full
            mask_dev = adapter.last_mask_deviation
            if before is None:
                raise RuntimeError(f"Layer {li} spectrum was not captured.")

            before_img = to_map(before)
            after_img = to_map(after)
            delta_img = to_map(delta)
            mask_img = robust_norm(mask_dev[0].mean(dim=0).numpy())

            axes[row, 0].imshow(before_img, cmap="viridis", interpolation="bicubic")
            axes[row, 0].set_title(f"L{li} before", fontsize=9)
            axes[row, 1].imshow(after_img, cmap="viridis", interpolation="bicubic")
            axes[row, 1].set_title(f"L{li} after", fontsize=9)
            axes[row, 2].imshow(delta_img, cmap="magma", interpolation="bicubic")
            axes[row, 2].set_title(f"L{li} |after-before|", fontsize=9)
            axes[row, 3].imshow(mask_img, cmap="magma", interpolation="bicubic")
            axes[row, 3].set_title(f"L{li} |mask-1|", fontsize=9)

            for col in range(4):
                axes[row, col].axis("off")

            print(
                f"Layer {li}: scale={adapter.last_scale_value:.6f}, "
                f"delta_mean={float(delta.mean()):.8f}, delta_max={float(delta.max()):.8f}, "
                f"mask_dev_mean={float(mask_dev.mean()):.8f}, mask_dev_max={float(mask_dev.max()):.8f}"
            )

        fig.suptitle(
            f"HFTA frequency diagnostics | pred={pred_idx}, conf={conf_value:.2f}",
            fontsize=11,
        )
        plt.tight_layout()
        png_path = args.output_dir / f"{args.prefix}_all_layers_sample{args.sample_index}.png"
        pdf_path = args.output_dir / f"{args.prefix}_all_layers_sample{args.sample_index}.pdf"
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

    # Save the selected layer as paper-style figure.
    adapter = model.get_adapter(args.layer_idx)
    before = adapter.last_fft_before_full
    after = adapter.last_fft_after_full
    delta = adapter.last_delta_full
    mask_dev = adapter.last_mask_deviation
    if before is None:
        raise RuntimeError("Failed to capture selected layer spectrum.")

    before_img = to_map(before)
    after_img = to_map(after)
    delta_img = to_map(delta)
    mask_img = robust_norm(mask_dev[0].mean(dim=0).numpy())

    print(
        f"Selected layer {args.layer_idx}: scale={adapter.last_scale_value:.6f}, "
        f"delta_mean={float(delta.mean()):.8f}, delta_max={float(delta.max()):.8f}, "
        f"mask_dev_mean={float(mask_dev.mean()):.8f}, mask_dev_max={float(mask_dev.max()):.8f}"
    )

    if missing_hfta > 0:
        print("\nWARNING: Some HFTA keys were missing during checkpoint loading.")
        print("The frequency difference may be near zero because the adapter is not loaded correctly.\n")

    if float(mask_dev.max()) < 1e-6 or float(delta.max()) < 1e-6:
        print("\nWARNING: HFTA modulation is almost zero for this checkpoint/layer.")
        print("Try --all_layers or check that the checkpoint is the trained FITFoodNet checkpoint.\n")

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=300)

    axes[0].imshow(raw_img)
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(before_img, cmap="viridis", interpolation="bicubic")
    axes[1].set_title("Before HFTA\nfull FFT magnitude", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(after_img, cmap="viridis", interpolation="bicubic")
    axes[2].set_title("After HFTA\nfull FFT magnitude", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(delta_img, cmap="magma", interpolation="bicubic")
    axes[3].set_title("Spectrum change\n|After - Before|", fontsize=10)
    axes[3].axis("off")

    axes[4].imshow(mask_img, cmap="magma", interpolation="bicubic")
    axes[4].set_title("Learned modulation\n|Mask - 1|", fontsize=10)
    axes[4].axis("off")

    fig.suptitle(
        f"HFTA frequency-domain visualization | layer {args.layer_idx} | "
        f"pred={pred_idx}, conf={conf_value:.2f}",
        fontsize=11,
    )
    plt.tight_layout()

    png_path = args.output_dir / f"{args.prefix}_layer{args.layer_idx}_sample{args.sample_index}.png"
    pdf_path = args.output_dir / f"{args.prefix}_layer{args.layer_idx}_sample{args.sample_index}.pdf"
    npz_path = args.output_dir / f"{args.prefix}_layer{args.layer_idx}_sample{args.sample_index}.npz"

    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        npz_path,
        before=before_img,
        after=after_img,
        delta=delta_img,
        mask=mask_img,
        image_path=str(image_path),
        label=-1 if label is None else label,
        pred_idx=pred_idx,
        confidence=conf_value,
        layer_idx=args.layer_idx,
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved NPZ: {npz_path}")


if __name__ == "__main__":
    main()
