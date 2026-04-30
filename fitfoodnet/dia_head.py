"""Dynamic Ingredient-Aware Head (DIA Head)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .freq_cross_attention import FrequencyCrossAttention


class DynamicIngredientAwareHead(nn.Module):
    """DIA Head with image-conditioned dynamic queries.

    No explicit ingredient labels are required. The head learns
    ingredient-related local discriminative patterns from image-level category
    supervision.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_queries: int = 8,
        num_heads: int = 8,
        hidden_dim: int = 1024,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        self.base_query = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        self.query_delta_gen = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_queries * embed_dim),
        )
        self.query_gate_gen = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_queries),
        )

        nn.init.zeros_(self.query_delta_gen[-1].weight)
        nn.init.zeros_(self.query_delta_gen[-1].bias)
        nn.init.zeros_(self.query_gate_gen[-1].weight)
        nn.init.zeros_(self.query_gate_gen[-1].bias)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = FrequencyCrossAttention(embed_dim, num_heads)
        self.local_norm = nn.LayerNorm(embed_dim)

        self.query_score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.last_vis: dict[str, Any] = {}

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        return_attn: bool = False,
    ):
        bsz, _, channels = patch_tokens.shape

        global_patch = patch_tokens.mean(dim=1)
        context = torch.cat([cls_token, global_patch], dim=-1)

        delta_q = self.query_delta_gen(context).view(bsz, self.num_queries, channels)
        gate = torch.sigmoid(self.query_gate_gen(context)).unsqueeze(-1)

        base_q = self.base_query.expand(bsz, -1, -1)
        query = self.query_norm(base_q + gate * delta_q)

        attn_output, attn_weights = self.cross_attn(query, patch_tokens, patch_tokens)
        local_features = self.local_norm(query + attn_output)

        query_logits = self.query_score(local_features).squeeze(-1)
        query_weights = torch.softmax(query_logits, dim=-1)

        weighted_local = torch.sum(local_features * query_weights.unsqueeze(-1), dim=1)
        max_local = local_features.max(dim=1).values

        fusion_features = torch.cat([cls_token, weighted_local, max_local], dim=-1)
        logits = self.classifier(fusion_features)

        with torch.no_grad():
            self.last_vis = {
                "query_weights": query_weights.detach(),
                "attn_weights": attn_weights.detach(),
                "dynamic_queries": query.detach(),
                "local_features": local_features.detach(),
            }

        if return_attn:
            aux = {
                "attn_weights": attn_weights,
                "query_weights": query_weights,
                "dynamic_queries": query,
                "local_features": local_features,
            }
            return logits, aux

        return logits


# Backward-compatible alias for older local scripts.
IngredientAwareHead = DynamicIngredientAwareHead

