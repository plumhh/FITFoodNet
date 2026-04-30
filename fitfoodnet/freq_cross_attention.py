"""Frequency cross-attention used by the DIA Head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyCrossAttention(nn.Module):
    """Cross-attention computed with frequency-magnitude matching.

    The one-dimensional FFT is applied along the per-head feature dimension.
    Attention weights are computed from L2-normalized frequency magnitudes and
    are then used to aggregate the frequency-domain value features.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_queries, channels = query.shape
        num_keys = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, num_keys, self.num_heads, self.head_dim).transpose(1, 2)

        q_f = torch.fft.rfft(q.float(), dim=-1)
        k_f = torch.fft.rfft(k.float(), dim=-1)
        v_f = torch.fft.rfft(v.float(), dim=-1)

        q_mag = F.normalize(torch.abs(q_f), p=2, dim=-1)
        k_mag = F.normalize(torch.abs(k_f), p=2, dim=-1)

        attn = torch.matmul(q_mag, k_mag.transpose(-2, -1))
        attn = attn / self.temperature.clamp(min=1e-3)
        attn = torch.softmax(attn.float(), dim=-1).to(q_mag.dtype)

        out_f = attn.to(v_f.dtype) @ v_f
        out = torch.fft.irfft(out_f, n=self.head_dim, dim=-1)
        out = out.transpose(1, 2).reshape(bsz, num_queries, channels)

        # Head-averaged query-to-token attention for visualization/loss.
        attn_out = attn.mean(dim=1)
        return self.out_proj(out), attn_out

