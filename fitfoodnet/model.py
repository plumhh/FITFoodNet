"""FITFoodNet model definition."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .dia_head import DynamicIngredientAwareHead
from .hfta import BlockWithFreqAdapter, DynamicFrequencyTextureAdapter


_HUB_NAMES = {
    "dinov3_s": "dinov3_vits16",
    "dinov3_b": "dinov3_vitb16",
    "dinov3_l": "dinov3_vitl16",
}

_EMBED_DIMS = {
    "dinov3_s": 384,
    "dinov3_b": 768,
    "dinov3_l": 1024,
}


def load_dinov3_backbone(
    model_type: str,
    dinov3_repo: str = "facebookresearch/dinov3",
    dinov3_source: str = "github",
    dinov3_weight: str | Path | None = None,
) -> tuple[nn.Module, int]:
    """Load a DINOv3 backbone through torch.hub and optional local weights."""

    if model_type not in _HUB_NAMES:
        raise ValueError(f"Unsupported DINOv3 model type: {model_type}")
    if dinov3_source not in {"github", "local"}:
        raise ValueError("dinov3_source must be 'github' or 'local'")

    backbone = torch.hub.load(
        dinov3_repo,
        _HUB_NAMES[model_type],
        source=dinov3_source,
        pretrained=False,
    )

    if dinov3_weight:
        try:
            state_dict = torch.load(Path(dinov3_weight), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(Path(dinov3_weight), map_location="cpu")
        backbone.load_state_dict(state_dict)

    return backbone, _EMBED_DIMS[model_type]


class FITFoodNet(nn.Module):
    """FITFoodNet with frozen DINOv3, HFTA modules, and DIA Head."""

    def __init__(
        self,
        num_classes: int,
        model_type: str = "dinov3_l",
        img_size: int = 224,
        num_queries: int = 8,
        num_heads: int = 8,
        hfta_bottleneck: int = 64,
        use_hfta: bool = True,
        dinov3_repo: str = "facebookresearch/dinov3",
        dinov3_source: str = "github",
        dinov3_weight: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.backbone, embed_dim = load_dinov3_backbone(
            model_type=model_type,
            dinov3_repo=dinov3_repo,
            dinov3_source=dinov3_source,
            dinov3_weight=dinov3_weight,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        if use_hfta:
            hw_shape = (img_size // 16, img_size // 16)
            candidate_layers = [5, 11, 17, 23]
            max_idx = len(self.backbone.blocks) - 1
            insert_layers = [idx for idx in candidate_layers if idx <= max_idx]

            for layer_idx in insert_layers:
                original_block = self.backbone.blocks[layer_idx]
                adapter = DynamicFrequencyTextureAdapter(
                    embed_dim=embed_dim,
                    bottleneck_dim=hfta_bottleneck,
                    hw_shape=hw_shape,
                )
                self.backbone.blocks[layer_idx] = BlockWithFreqAdapter(original_block, adapter)

        self.head = DynamicIngredientAwareHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        return self.head(cls_token, patch_tokens, return_attn=return_attn)


# Backward-compatible alias for older local scripts.
DINOv3Classifier = FITFoodNet
