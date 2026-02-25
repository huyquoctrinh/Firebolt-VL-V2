# SSM on visual stream: S4 / S4D / self_attn (optional import from Viper-LM)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelfAttnSSM(nn.Module):
    """Self-attention temporal mixing (B, L, D) -> (B, L, D)."""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        x = self.norm(x)
        x_attn, _ = self.attn(x, x, x)
        x = residual + self.dropout(x_attn)
        x = x + self.ff(self.norm(x))
        return x, None


class SSMWrapper(nn.Module):
    """Wrap Viper-LM S4/S4D: (B, L, D) in/out; S4D expects (B, D, L)."""
    def __init__(self, inner: nn.Module, transposed: bool = True):
        super().__init__()
        self.inner = inner
        self.transposed = transposed

    def forward(self, x: torch.Tensor, state=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.transposed:
            x = x.transpose(1, 2)
        out, st = self.inner(x, **({} if state is None else {"state": state}))
        if self.transposed:
            out = out.transpose(1, 2)
        return out, st


def build_ssm(
    ssm_type: str,
    d_model: int,
    d_state: int = 64,
    dropout: float = 0.1,
    viper_lm_path: Optional[str] = None,
) -> nn.Module:
    if ssm_type == "self_attn":
        return SelfAttnSSM(d_model, num_heads=8, dropout=dropout)
    if ssm_type == "s4d":
        try:
            path = viper_lm_path or _get_viper_lm_path()
            if path:
                import sys
                if path not in sys.path:
                    sys.path.insert(0, path)
                from modeling.ssm.s4d import S4D
                inner = S4D(d_model, d_state=d_state, dropout=dropout, transposed=True)
                return SSMWrapper(inner, transposed=True)
        except Exception:
            pass
        return SelfAttnSSM(d_model, num_heads=8, dropout=dropout)
    if ssm_type == "s4":
        try:
            path = viper_lm_path or _get_viper_lm_path()
            if path:
                import sys
                if path not in sys.path:
                    sys.path.insert(0, path)
                from modeling.ssm.s4 import S4Block
                return S4Block(d_model, l_max=512, dropout=dropout, transposed=False)
        except Exception:
            pass
        return SelfAttnSSM(d_model, num_heads=8, dropout=dropout)
    return SelfAttnSSM(d_model, num_heads=8, dropout=dropout)


def _get_viper_lm_path():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    viper = root.parent / "Viper-LM"
    return str(viper) if viper.exists() else None
