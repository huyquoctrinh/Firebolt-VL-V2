import torch
import torch.nn as nn
from typing import Optional, Tuple


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.temperature = temperature
        self.importance = nn.Linear(d_model, 1)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = z.shape
        k = min(self.top_k, N)
        scores = self.importance(z).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        scores = scores / max(self.temperature, 1e-8)
        topv, topi = torch.topk(scores, k, dim=-1)
        indices = topi
        z_top = torch.gather(z, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        return z_top, indices
