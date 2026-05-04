"""
AutoVision encoder plugin registry.

Each block registers a foundation-model encoder into the JIT selection pool.
Encoders are activated only when their dependencies are importable and
the hardware budget covers their VRAM footprint.

New encoders are automatically ranked by capacity and selected by the
JIT VRAM profiler — no other changes needed.

Plugin contract
---------------
  factory       : zero-arg callable → frozen, eval-mode nn.Module
                  The module must have a ``get_output_dim() -> int`` method.
  output_dim    : int — dimensionality of the encoder's output tensor
  capacity      : int — approximate parameter count (used for VRAM ranking)
  dummy_input_fn: (batch_size, device) -> Tensor  for dry-run profiling
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── CLIP ViT-B/16 (OpenAI, 2021) ──────────────────────────────────────────
# 86M params, 512-dim output.  Pre-trained on 400M image-text pairs.
# CLIP normalization: mean=[0.481, 0.458, 0.408]  std=[0.269, 0.261, 0.276]

try:
    import torch as _torch
    from automl.jit_encoder_selector import register_vision_encoder as _rvenc

    def _make_clip_vit_b16():
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai"
            )
        except ImportError:
            from transformers import CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            model = model.vision_model

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        class _CLIPWrapper(_torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.normalize_mode = "clip"
                self.normalize_mean = [0.481, 0.458, 0.408]
                self.normalize_std = [0.269, 0.261, 0.276]

            def forward(self, x, return_all_tokens: bool = False):
                if hasattr(self.m, "encode_image"):
                    if return_all_tokens:
                        # Patch tokens via ViT forward with all tokens
                        out = self.m.visual.transformer(
                            self.m.visual.conv1(x).flatten(2).transpose(1, 2)
                            if hasattr(self.m, "visual") else x
                        )
                        return out if isinstance(out, _torch.Tensor) else out[0]
                    return self.m.encode_image(x)
                return self.m(pixel_values=x).pooler_output

            def get_output_dim(self):
                return 512

        return _CLIPWrapper(model)

    _rvenc(
        name="CLIP-ViT-B/16",
        factory=_make_clip_vit_b16,
        output_dim=512,
        capacity=86_000_000,
        dummy_input_fn=lambda bs, dev: _torch.zeros(bs, 3, 224, 224, device=dev),
    )
    logger.info("Plugin registered: CLIP-ViT-B/16")
except Exception as _clip_exc:
    logger.debug("CLIP-ViT-B/16 plugin not loaded: %s", _clip_exc)


# ── DINOv2 ViT-B/14 (Meta, 2023) ──────────────────────────────────────────
# 86M params, 768-dim output.  Self-supervised, superior dense features.

try:
    import torch as _torch
    from automl.jit_encoder_selector import register_vision_encoder as _rvenc

    def _make_dinov2_vit_b14():
        try:
            model = _torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", verbose=False
            )
        except Exception:
            from transformers import AutoModel
            model = AutoModel.from_pretrained("facebook/dinov2-base")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        class _DinoWrapper(_torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, return_all_tokens: bool = False):
                if hasattr(self.m, "forward_features"):
                    feats = self.m.forward_features(x)
                    if return_all_tokens and isinstance(feats, dict):
                        return feats.get("x_norm_patchtokens", feats.get("x", x))
                    if isinstance(feats, dict):
                        return feats.get("x_norm_clstoken", feats.get("x", x)[:, 0])
                    return feats[:, 0]
                if hasattr(self.m, "last_hidden_state"):
                    out = self.m(pixel_values=x)
                    if return_all_tokens:
                        return out.last_hidden_state[:, 1:]
                    return out.last_hidden_state[:, 0]
                return self.m(x)

            def get_output_dim(self):
                return 768

        return _DinoWrapper(model)

    _rvenc(
        name="DINOv2-ViT-B/14",
        factory=_make_dinov2_vit_b14,
        output_dim=768,
        capacity=86_000_000,
        dummy_input_fn=lambda bs, dev: _torch.zeros(bs, 3, 224, 224, device=dev),
    )
    logger.info("Plugin registered: DINOv2-ViT-B/14")
except Exception as _dino_exc:
    logger.debug("DINOv2-ViT-B/14 plugin not loaded: %s", _dino_exc)


# ── SigLIP ViT-B/16 (Google, 2023) ────────────────────────────────────────
# 93M params, 768-dim output.  Sigmoid loss contrastive — better calibrated
# probabilities than CLIP's softmax loss.
# Requires: pip install transformers

try:
    import torch as _torch
    from automl.jit_encoder_selector import register_vision_encoder as _rvenc
    from transformers import AutoModel as _AutoModel

    def _make_siglip_vit_b16():
        model = _AutoModel.from_pretrained("google/siglip-base-patch16-224")
        vision = model.vision_model
        vision.eval()
        for p in vision.parameters():
            p.requires_grad = False

        class _SigLIPWrapper(_torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, return_all_tokens: bool = False):
                # SigLIP-ViT-B/16 position embedding has exactly 196 positions
                # (14×14 patches from 224×224 / 16px). Resize if the ImagePreprocessor
                # produced a different resolution so position IDs always match.
                if x.shape[-1] != 224 or x.shape[-2] != 224:
                    import torch.nn.functional as _F
                    x = _F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                out = self.m(pixel_values=x)
                if return_all_tokens:
                    # SigLIP has NO CLS token — last_hidden_state is all 196 patches.
                    # [:, 1:] would incorrectly drop patch 0; return the full sequence.
                    return out.last_hidden_state          # (N, 196, 768)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    return out.pooler_output              # (N, 768) — mean of all patches
                return out.last_hidden_state.mean(dim=1) # (N, 768) fallback

            def get_output_dim(self):
                return 768

        return _SigLIPWrapper(vision)

    _rvenc(
        name="SigLIP-ViT-B/16",
        factory=_make_siglip_vit_b16,
        output_dim=768,
        capacity=93_000_000,
        dummy_input_fn=lambda bs, dev: _torch.zeros(bs, 3, 224, 224, device=dev),
    )
    logger.info("Plugin registered: SigLIP-ViT-B/16")
except Exception as _siglip_exc:
    logger.debug("SigLIP-ViT-B/16 plugin not loaded: %s", _siglip_exc)


# ── Mistral-7B-Instruct (4-bit quantized) ─────────────────────────────────
# Requires: pip install transformers bitsandbytes accelerate  + 6+ GB VRAM

# def _make_mistral_7b_4bit(): ...
# Uncomment when GPU memory ≥ 6 GB and bitsandbytes installed.


# ── Sentence-Transformers all-mpnet-base-v2 ───────────────────────────────
# 110M params, 768-dim.  Best semantic similarity encoder as of 2023.
# Requires: pip install sentence-transformers

try:
    import torch as _torch
    from automl.jit_encoder_selector import register_text_encoder as _rtenc
    from sentence_transformers import SentenceTransformer as _ST

    def _make_mpnet():
        st = _ST("all-mpnet-base-v2")
        st.eval()
        for p in st.parameters():
            p.requires_grad = False

        class _STWrapper(_torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids):
                emb = self.m({"input_ids": input_ids,
                              "attention_mask": (input_ids > 0).long()})
                return emb["sentence_embedding"]

            def get_output_dim(self):
                return 768

        return _STWrapper(st)

    _rtenc(
        name="all-mpnet-base-v2",
        factory=_make_mpnet,
        output_dim=768,
        capacity=110_000_000,
        dummy_input_fn=lambda bs, dev: _torch.zeros(bs, 128, dtype=_torch.long, device=dev),
    )
    logger.info("Plugin registered: all-mpnet-base-v2")
except Exception as _mpnet_exc:
    logger.debug("all-mpnet-base-v2 plugin not loaded: %s", _mpnet_exc)
