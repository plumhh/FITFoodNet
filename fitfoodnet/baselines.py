"""DINOv3 baseline and PEFT comparison models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .model import load_dinov3_backbone


class ResidualAdapter(nn.Module):
    """Post-block bottleneck adapter: y = x + A(x)."""

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.full((1,), init_scale))

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.down_proj(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        return x + self.scale * out


class AdaptFormerBranch(nn.Module):
    """Parallel bottleneck branch added to the original block output."""

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.full((1,), init_scale))

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.down_proj(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        return self.scale * out


class BlockWithAdapter(nn.Module):
    """Wrap a frozen transformer block with a residual adapter."""

    def __init__(self, original_block: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def _apply_adapter(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.adapter(tensor)

    def forward(self, x, *args, **kwargs):
        out = self.original_block(x, *args, **kwargs)
        if isinstance(out, (tuple, list)):
            return type(out)([self._apply_adapter(tensor) for tensor in out])
        return self._apply_adapter(out)


class BlockWithAdaptFormer(nn.Module):
    """Wrap a frozen transformer block with an AdaptFormer-style branch."""

    def __init__(self, original_block: nn.Module, branch: nn.Module) -> None:
        super().__init__()
        self.original_block = original_block
        self.branch = branch

    def _merge(self, x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        return x_out + self.branch(x_in)

    def forward(self, x, *args, **kwargs):
        out = self.original_block(x, *args, **kwargs)
        if isinstance(out, (tuple, list)):
            if not isinstance(x, (tuple, list)):
                raise TypeError("Tuple/list block output requires tuple/list block input.")
            return type(out)([self._merge(xi, xo) for xi, xo in zip(x, out)])
        return self._merge(x, out)


def build_mlp_head(embed_dim: int, num_classes: int) -> nn.Sequential:
    """MLP classification head used by linear-probe and PEFT baselines."""

    return nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


class DINOv3BaselineClassifier(nn.Module):
    """DINOv3 baseline model for linear probe, full fine-tuning, and PEFT."""

    SUPPORTED_METHODS = {"linear_probe", "full_finetune", "adapter", "adaptformer"}

    def __init__(
        self,
        num_classes: int,
        method: str = "linear_probe",
        model_type: str = "dinov3_l",
        adapter_dim: int = 64,
        insert_layers: Sequence[int] | None = None,
        adapter_dropout: float = 0.0,
        adapter_scale: float = 1e-4,
        dinov3_repo: str = "facebookresearch/dinov3",
        dinov3_source: str = "github",
        dinov3_weight: str | Path | None = None,
    ) -> None:
        super().__init__()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported baseline method: {method}")

        self.method = method
        self.backbone, embed_dim = load_dinov3_backbone(
            model_type=model_type,
            dinov3_repo=dinov3_repo,
            dinov3_source=dinov3_source,
            dinov3_weight=dinov3_weight,
        )

        train_backbone = method == "full_finetune"
        for param in self.backbone.parameters():
            param.requires_grad = train_backbone

        if method in {"adapter", "adaptformer"}:
            self._insert_peft_modules(
                method=method,
                embed_dim=embed_dim,
                adapter_dim=adapter_dim,
                insert_layers=insert_layers,
                adapter_dropout=adapter_dropout,
                adapter_scale=adapter_scale,
            )

        self.head = build_mlp_head(embed_dim=embed_dim, num_classes=num_classes)

    def _insert_peft_modules(
        self,
        method: str,
        embed_dim: int,
        adapter_dim: int,
        insert_layers: Sequence[int] | None,
        adapter_dropout: float,
        adapter_scale: float,
    ) -> None:
        if insert_layers is None:
            insert_layers = [5, 11, 17, 23]

        max_idx = len(self.backbone.blocks) - 1
        valid_layers = [idx for idx in insert_layers if idx <= max_idx]

        for layer_idx in valid_layers:
            original_block = self.backbone.blocks[layer_idx]
            if method == "adapter":
                adapter = ResidualAdapter(
                    embed_dim=embed_dim,
                    bottleneck_dim=adapter_dim,
                    dropout=adapter_dropout,
                    init_scale=adapter_scale,
                )
                self.backbone.blocks[layer_idx] = BlockWithAdapter(original_block, adapter)
            else:
                branch = AdaptFormerBranch(
                    embed_dim=embed_dim,
                    bottleneck_dim=adapter_dim,
                    dropout=adapter_dropout,
                    init_scale=adapter_scale,
                )
                self.backbone.blocks[layer_idx] = BlockWithAdaptFormer(original_block, branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.head(features)
