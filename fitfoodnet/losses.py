"""Loss functions for FITFoodNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_orthogonal_loss(attn_weights: torch.Tensor | list | tuple) -> torch.Tensor:
    """Compute orthogonality loss on query-to-token attention.

    Args:
        attn_weights: Attention tensor with shape [B, Nq, N] or
            [B, H, Nq, N]. A list/tuple input uses the first element.
    """

    if isinstance(attn_weights, (list, tuple)):
        attn_weights = attn_weights[0]

    if attn_weights.dim() == 4:
        attn_weights = attn_weights.mean(dim=1)

    attn_weights = attn_weights.to(torch.float32)
    attn = F.normalize(attn_weights, p=2, dim=-1)
    sim = torch.matmul(attn, attn.transpose(-1, -2))
    identity = torch.eye(sim.size(-1), device=sim.device)
    return ((sim - identity) ** 2).mean()


def ortho_weight(epoch: int, max_weight: float = 0.03, warmup_epochs: int = 10) -> float:
    """Linear warm-up schedule for the orthogonality loss weight."""

    if warmup_epochs <= 0:
        return max_weight
    return min(max_weight, max_weight * epoch / warmup_epochs)

