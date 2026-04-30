"""High-Frequency Texture Adapter (HFTA)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicFrequencyTextureAdapter(nn.Module):
    """Frequency-aware texture adapter for ViT patch tokens.

    The adapter projects patch tokens to a bottleneck space, modulates their
    two-dimensional frequency representation with an input-conditioned complex
    mask, and adds the result back through a learnable residual scale.
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 64,
        hw_shape: tuple[int, int] = (14, 14),
    ) -> None:
        super().__init__()
        self.hw_shape = hw_shape
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()

        freq_elements = hw_shape[0] * (hw_shape[1] // 2 + 1) * 2 * bottleneck_dim
        self.mask_generator = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, freq_elements),
        )

        nn.init.zeros_(self.mask_generator[-1].weight)
        nn.init.zeros_(self.mask_generator[-1].bias)

        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = x.shape
        height, width = self.hw_shape
        orig_dtype = x.dtype

        x_down = self.act(self.down_proj(x))
        global_context = x_down.mean(dim=1)

        raw_mask = self.mask_generator(global_context)
        dynamic_mask = torch.tanh(raw_mask).to(torch.float32)
        dynamic_mask = dynamic_mask.view(bsz, -1, height, width // 2 + 1, 2)

        x_spatial = x_down.transpose(1, 2).view(bsz, -1, height, width).to(torch.float32)
        fft_x = torch.fft.rfft2(x_spatial, norm="ortho")

        weight_complex = torch.view_as_complex(dynamic_mask) + 1.0
        fft_x_enhanced = fft_x * weight_complex
        x_spatial_enhanced = torch.fft.irfft2(
            fft_x_enhanced,
            s=(height, width),
            norm="ortho",
        )

        x_seq = x_spatial_enhanced.to(orig_dtype).flatten(2).transpose(1, 2)
        out = self.up_proj(x_seq)
        return x + out * self.scale


class BlockWithFreqAdapter(nn.Module):
    """Wrap a transformer block and apply HFTA to the output patch tokens."""

    def __init__(self, original_block: nn.Module, adapter: DynamicFrequencyTextureAdapter) -> None:
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def _process_single_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        height, width = self.adapter.hw_shape
        num_patches = height * width
        other_tokens = tokens[:, :-num_patches, :]
        patch_tokens = tokens[:, -num_patches:, :]
        enhanced_patch_tokens = self.adapter(patch_tokens)
        return torch.cat([other_tokens, enhanced_patch_tokens], dim=1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.original_block(x, *args, **kwargs)
        if isinstance(out, (list, tuple)):
            return type(out)([self._process_single_tensor(t) for t in out])
        return self._process_single_tensor(out)

