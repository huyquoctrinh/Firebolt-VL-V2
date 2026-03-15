import torch
import torch.nn as nn


class FeatureDropout(nn.Module):
    """Dropout along the feature dimension on visual tokens."""

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x: [B, N, D] — drop entire feature channels
        mask = torch.ones(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        mask = nn.functional.dropout(mask, p=self.p, training=True)
        return x * mask


class MemoryModule(nn.Module):
    """LayerNorm + Linear + GELU + Linear residual block to refine visual tokens."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.proj(self.norm(x))


class GuidingGate(nn.Module):
    """Text-conditioned per-token relevance gate."""

    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, vision_dim)
        self.gate = nn.Linear(vision_dim * 2, 1)

    def forward(self, z_visual, z_text):
        # z_visual: [B, N_v, vision_dim]
        # z_text:   [B, N_t, text_dim]
        z_text_proj = self.text_proj(z_text)          # [B, N_t, vision_dim]
        z_text_pool = z_text_proj.mean(dim=1, keepdim=True)  # [B, 1, vision_dim]
        z_text_expanded = z_text_pool.expand_as(z_visual)     # [B, N_v, vision_dim]
        combined = torch.cat([z_visual, z_text_expanded], dim=-1)  # [B, N_v, 2*vision_dim]
        return torch.sigmoid(self.gate(combined))  # [B, N_v, 1]


class ExpertMLP(nn.Module):
    """2-layer MLP expert: vision_dim -> hidden_size."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MoERouter(nn.Module):
    """Combines GuidingGate + expert dispatch with top-k routing."""

    def __init__(self, vision_dim, text_dim, output_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.guiding_gate = GuidingGate(vision_dim, text_dim)
        self.router = nn.Linear(vision_dim, num_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(vision_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, z_visual, z_text):
        # Compute text-conditioned relevance
        relevance = self.guiding_gate(z_visual, z_text)  # [B, N, 1]
        z_gated = z_visual * relevance  # [B, N, vision_dim]

        # Router logits and top-k selection
        router_logits = self.router(z_gated)  # [B, N, num_experts]
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, N, top_k]
        top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [B, N, top_k]

        # Run all experts and gather top-k outputs
        # expert_outputs: list of [B, N, output_dim]
        expert_outputs = torch.stack([expert(z_gated) for expert in self.experts], dim=2)  # [B, N, num_experts, output_dim]

        # Gather top-k expert outputs
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1])  # [B, N, top_k, output_dim]
        selected_outputs = torch.gather(expert_outputs, 2, top_k_indices_expanded)  # [B, N, top_k, output_dim]

        # Weighted sum
        output = (selected_outputs * top_k_weights.unsqueeze(-1)).sum(dim=2)  # [B, N, output_dim]
        return output


class MoEVisionLanguageConnector(nn.Module):
    """MoE-based vision-language connector.

    Replaces the default MLP projector with a Mixture-of-Experts architecture
    that uses text context to guide visual token processing.
    """

    requires_text_context = True

    def __init__(self, config):
        super().__init__()
        vision_dim = config.mm_hidden_size       # 1024
        text_dim = config.hidden_size             # 4096
        num_experts = getattr(config, 'moe_num_experts', 4)
        top_k = getattr(config, 'moe_top_k', 2)
        feat_dropout = getattr(config, 'moe_feature_dropout', 0.1)

        self.feature_dropout = FeatureDropout(p=feat_dropout)
        self.memory = MemoryModule(vision_dim)
        self.router = MoERouter(
            vision_dim=vision_dim,
            text_dim=text_dim,
            output_dim=text_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, x, z_text=None):
        """
        Args:
            x: visual features [B, N, vision_dim]
            z_text: text embeddings [B, N_t, text_dim]
        Returns:
            projected features [B, N, text_dim]
        """
        if z_text is None:
            raise ValueError("MoEVisionLanguageConnector requires z_text but received None")
        x = self.feature_dropout(x)
        x = self.memory(x)
        x = self.router(x, z_text)
        return x
