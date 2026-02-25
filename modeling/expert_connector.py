# Expert Connector: nhiều Expert, router chọn tổ hợp -> Z_I (B, T*k, D)
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Một expert: FFN với hidden_dim."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class ExpertConnector(nn.Module):
    """
    Expert Connector: Z_I (B, k, D) -> qua router chọn experts -> (B, k, D).
    MoE-style: router(B, k, num_experts) -> weighted sum of expert outputs.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, k, D)
        Returns: (B, k, D)
        """
        _, _, _ = z.shape
        router_logits = self.router(z)
        top_vals, top_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_vals, dim=-1).to(dtype=router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter(-1, top_idx, top_weights)
        out = torch.zeros_like(z)
        for i, expert in enumerate(self.experts):
            w = router_weights[..., i : i + 1]
            out = out + w * expert(z)
        return out
