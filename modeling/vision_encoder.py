import torch
import torch.nn as nn
from typing import Optional


def _load_siglip_encoder(ckpt_path: str, dtype=torch.float16, device=None):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(ckpt_path, dtype=dtype, low_cpu_mem_usage=True)
    if hasattr(model, "vision_model"):
        model = model.vision_model
    if device is not None:
        model = model.to(device)
    return model


def _extract_patch_features(backbone: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return patch tokens as (B, N_patches, D) for SigLIP-like backbones."""
    out = None

    # Preferred path for multi-modal wrappers (e.g. SigLIP model with vision tower).
    if hasattr(backbone, "vision_model"):
        try:
            out = backbone.vision_model(pixel_values=pixel_values, return_dict=True)
        except TypeError:
            out = backbone.vision_model(pixel_values)
    if out is None:
        try:
            out = backbone(pixel_values=pixel_values, return_dict=True)
        except TypeError:
            try:
                out = backbone(pixel_values)
            except Exception:
                out = None

    feats = None
    if out is not None:
        if hasattr(out, "vision_model_output") and getattr(out.vision_model_output, "last_hidden_state", None) is not None:
            feats = out.vision_model_output.last_hidden_state
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            feats = out.last_hidden_state
        elif isinstance(out, dict):
            if "vision_model_output" in out and hasattr(out["vision_model_output"], "last_hidden_state"):
                feats = out["vision_model_output"].last_hidden_state
            elif "last_hidden_state" in out:
                feats = out["last_hidden_state"]
            elif "image_embeds" in out:
                feats = out["image_embeds"]
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            feats = out[0]

    # Fallback when model only exposes pooled image features.
    if feats is None:
        if hasattr(backbone, "get_image_features"):
            feats = backbone.get_image_features(pixel_values)
        else:
            raise RuntimeError("Cannot extract vision features from backbone output.")

    if feats.dim() == 2:
        feats = feats.unsqueeze(1)
    if feats.dim() != 3:
        raise RuntimeError(f"Expected 3D patch features, got shape {tuple(feats.shape)}.")
    return feats


class VisionEncoderVJEPA(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        encoder_type: str = "siglip",
        tile_size: int = 128,
        overlap: int = 2,
        add_global: bool = True,
        global_size: int = 256,
        freeze: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.freeze = freeze
        self._dtype = dtype or torch.float32
        self._device = device
        # Always use a single-image vision backbone; no grid/tiled extraction.
        self.encoder = _load_siglip_encoder(ckpt_path, self._dtype, device)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze):
            return _extract_patch_features(self.encoder, pixel_values)
