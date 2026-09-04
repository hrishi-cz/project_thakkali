"""
modelss/adapters/lora.py

Low-Rank Adaptation (LoRA) for frozen encoder fine-tuning.

Usage
-----
>>> from modelss.adapters.lora import apply_lora, lora_parameters
>>> encoder = TextEncoder("bert-base-uncased", freeze_backbone=True)
>>> apply_lora(encoder, r=8, alpha=16)
>>> trainable = list(lora_parameters(encoder))  # only A/B matrices

References
----------
Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models."
ICLR 2022.  https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import logging
import re as _re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch  # noqa: F401 — needed for type hint in property

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default target module name fragments for attention layers
_DEFAULT_TARGETS: Tuple[str, ...] = (
    "query", "value",          # BERT / RoBERTa / DeBERTa
    "q_proj", "v_proj",        # LLaMA / Mistral / GPT-style
    "q", "v",                  # ViT attention
    "to_q", "to_v",            # DiT / diffusion transformers
    "out_proj",                # nn.MultiheadAttention output projection
    "linear1",                 # PyTorch TransformerEncoderLayer FFN layer 1
)

# Ordered regex patterns covering HuggingFace transformer block naming conventions:
#   BERT/DeBERTa/RoBERTa/XLM/DINOv2 : encoder.layer.N
#   DistilBERT                        : transformer.layer.N
#   CLIP ViT (HuggingFace)            : vision_model.encoder.layers.N
#   CLIP ViT (raw OpenAI)             : transformer.resblocks.N
#   GPT-2 / GPT-J                     : transformer.h.N
#   Swin / generic ViT blocks         : layers.N  or  blocks.N
_BLOCK_PATTERNS: Tuple[str, ...] = (
    r'(?:^|\.)(?:encoder\.layer|encoder\.layers|transformer\.layer'
    r'|vision_model\.encoder\.layers)\.(\d+)\.',
    r'(?:^|\.)transformer\.resblocks\.(\d+)\.',
    r'(?:^|\.)transformer\.h\.(\d+)\.',
    r'(?:^|\.)(?:layers|blocks)\.(\d+)\.',
)


def _block_idx_from_path(full_path: str) -> int:
    """Extract transformer block index from a dotted module path. Returns -1 if not found."""
    for pat in _BLOCK_PATTERNS:
        m = _re.search(pat, full_path)
        if m:
            return int(m.group(1))
    return -1


def _collect_lora_candidates(
    module: nn.Module,
    target_modules: Sequence[str],
    prefix: str = "",
) -> List[Tuple[nn.Module, str, str, nn.Linear]]:
    """
    DFS collect (parent, attr_name, full_path, child_linear) for all matching nn.Linear layers.

    Only layers whose *immediate* attribute name contains one of the target_modules
    strings are collected — consistent with the original apply_lora behaviour.
    """
    result: List[Tuple[nn.Module, str, str, nn.Linear]] = []
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
            result.append((module, name, full, child))
        else:
            result.extend(_collect_lora_candidates(child, target_modules, prefix=full))
    return result


class LoRALinear(nn.Module):
    """
    Drop-in replacement for a frozen ``nn.Linear`` with LoRA adaptation.

    The adapted weight is:
        W' = W + (B @ A) * (alpha / r)

    where:
        A ∈ ℝ^{r × d_in}  — initialized with Kaiming-uniform noise
        B ∈ ℝ^{d_out × r} — initialized to zero (ensures W'=W at start)

    Only A and B are updated during training; W is frozen.

    Parameters
    ----------
    linear : nn.Linear
        The frozen linear layer to wrap.  Its weights remain unchanged.
    r : int
        Rank of the low-rank decomposition.
    alpha : float
        Scaling factor.  The effective scale is ``alpha / r``.
        Using ``alpha = 2*r`` is a common default.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        # Freeze backbone immediately — LoRALinear always owns the frozen layer
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)
        self.linear = linear
        self.r = r
        self.scale = float(alpha) / float(r)

        # LoRA A: Kaiming-uniform init (like default nn.Linear)
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # LoRA B: zero init — so at start ΔW = 0 (no perturbation)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    # ── Proxy attributes so PyTorch internals (e.g. nn.MultiheadAttention)
    # can still access .weight and .bias on the LoRALinear wrapper ──────────

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight = frozen base + LoRA delta (detached for read-only access)."""
        return self.linear.weight + (self.lora_B @ self.lora_A) * self.scale

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # ΔW·x = B @ (A @ x^T) = (x @ A^T) @ B^T  — avoids large intermediate
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + delta

    def extra_repr(self) -> str:
        d_out, d_in = self.linear.weight.shape
        return f"d_in={d_in}, d_out={d_out}, r={self.r}, scale={self.scale:.3f}"


def apply_lora(
    module: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Sequence[str] = _DEFAULT_TARGETS,
    last_n_layers: int = 0,
) -> nn.Module:
    """
    Inject LoRA adapters into matching ``nn.Linear`` layers in *module*.

    Parameters
    ----------
    module : nn.Module
        Any PyTorch module (typically a frozen encoder).
    r : int
        LoRA rank.
    alpha : float
        LoRA scaling.
    target_modules : sequence of str
        Name fragments to match.  Defaults to common attention projection names.
    last_n_layers : int
        When > 0, restrict LoRA injection to the last N transformer blocks only.
        Block indices are detected from the module path via ``_BLOCK_PATTERNS``.
        Falls back to the last ``last_n_layers * 4`` candidates by DFS order when
        no block indices can be parsed (unusual architectures).
        When 0 (default), all matching layers across the full module are patched.

    Returns
    -------
    nn.Module
        The same *module* with LoRA adapters injected in-place.
    """
    candidates = _collect_lora_candidates(module, target_modules)

    if not candidates:
        # CNN-family encoders (ResNet, EfficientNet, ConvNeXt) have no Q/V attention
        # projections — LoRA is not applicable; return unchanged.
        logger.info(
            "apply_lora: no target layers found in %s — LoRA not applicable (CNN/no-attention encoder)",
            type(module).__name__,
        )
        return module

    if last_n_layers > 0:
        block_indices = [_block_idx_from_path(c[2]) for c in candidates]
        valid_blocks = [b for b in block_indices if b >= 0]

        if valid_blocks:
            max_block = max(valid_blocks)
            threshold = max_block - last_n_layers + 1
            selected_blocks = sorted(set(b for b in valid_blocks if b >= threshold))
            filtered = [
                c for c, b in zip(candidates, block_indices)
                if b < 0 or b >= threshold
            ]
            logger.info(
                "apply_lora: LoRA injected into blocks %s (%d layers total)",
                selected_blocks, len(filtered),
            )
        else:
            # No block indices found (unusual architecture) — positional fallback:
            # take the last last_n_layers * 4 candidates in DFS order.
            n_take = min(len(candidates), last_n_layers * 4)
            filtered = candidates[-n_take:]
            logger.info(
                "apply_lora: positional fallback — last %d of %d candidates "
                "(no block indices found in module paths)",
                len(filtered), len(candidates),
            )
    else:
        filtered = candidates

    n_replaced = 0
    for parent, attr_name, _full, child in filtered:
        child.weight.requires_grad_(False)
        if child.bias is not None:
            child.bias.requires_grad_(False)
        setattr(parent, attr_name, LoRALinear(child, r=r, alpha=alpha))
        n_replaced += 1

    if n_replaced:
        logger.debug("apply_lora: replaced %d Linear layers in %s", n_replaced, type(module).__name__)
    return module


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only LoRA A and B parameters (not backbone weights)."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a lightweight state dict containing only LoRA A/B tensors.
    Suitable for checkpointing adapters without saving the full backbone.
    """
    state: Dict[str, torch.Tensor] = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            state[f"{name}.lora_A"] = m.lora_A.data.clone()
            state[f"{name}.lora_B"] = m.lora_B.data.clone()
    return state


def load_lora_state_dict(module: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load a LoRA-only state dict back into a module that already has LoRA applied."""
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state:
                m.lora_A.data.copy_(state[key_a])
            if key_b in state:
                m.lora_B.data.copy_(state[key_b])
