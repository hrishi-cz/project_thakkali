"""
MultimodalInferenceEngine – load Phase-7 artifacts and run batch inference + XAI.

Artifacts consumed
------------------
models/registry/{model_id}/
├── artifacts/
│   ├── model_weights.pth           – _MultimodalHead state dict
│   ├── tabular_scaler.joblib       – fitted TabularPreprocessor
│   ├── tabular_encoder_state.pth   – trained GRN/MLP tabular encoder (optional)
│   ├── text_tokenizer/             – HuggingFace tokenizer (optional)
│   ├── text_encoder_state.pth      – frozen TextEncoder weights (optional)
│   ├── image_encoder_state.pth     – frozen ImageEncoder weights (optional)
│   ├── encoder_config.json         – encoder model names + settings (optional)
│   ├── probability_calibrator.joblib – post-hoc probability calibration (optional)
│   └── schema.json                 – GlobalSchema from Phase 2
└── metadata.json                   – full provenance (config, artifact_paths, …)

Design notes
------------
* The fusion head (_MultimodalHead) is always loaded from model_weights.pth.
* Frozen encoders (BERT for text, ResNet50 for image) are loaded from saved
  state dicts when available, allowing real multimodal inference.
* ``predict_batch`` runs under ``torch.no_grad()``.
* ``generate_explanations`` enables gradients only while Captum is active.
* Text token attributions use real BERT word embeddings when the text encoder
  is available, falling back to an approximate random-embedding method.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.paths import MODEL_REGISTRY_DIR

logger = logging.getLogger(__name__)


def _canonical_fusion_strategy(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "concat": "concatenation",
        "concatenationfusion": "concatenation",
        "attentionfusion": "attention",
        "unifiedlatentfusion": "ula",
        "unified_latent": "ula",
        "unified_latent_alignment": "ula",
        "omnimodal": "ula",
        "gatedfusion": "gated",
        "gated_fusion": "gated",
    }
    return aliases.get(raw, raw or "concatenation")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MultimodalInferenceEngine:
    """
    Load Phase-7 model artifacts and run batch inference with optional XAI.

    Parameters
    ----------
    model_id : str
        The model directory name under ``models/registry/``.

    Raises
    ------
    FileNotFoundError
        If ``models/registry/{model_id}/artifacts/`` does not exist.
    """

    # Known fixed output dims for text/image encoders used during training
    TEXT_DIM: int  = 768   # BERT-base CLS pooled dim
    IMAGE_DIM: int = 512   # ImageEncoder projects ResNet-50 GAP (2048) → 512
    TABULAR_ENCODER_DIM: int = 16  # GRN/MLP tabular encoder output dim

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __init__(self, model_id: str) -> None:
        self.model_id: str = model_id
        registry_root: Path = MODEL_REGISTRY_DIR / model_id
        self.artifacts_dir: Path = registry_root / "artifacts"

        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {self.artifacts_dir}. "
                "Run the full 7-phase training pipeline first."
            )

        # Load JSON bookkeeping files
        self.metadata: Dict[str, Any] = self._load_json(registry_root / "metadata.json")
        self.schema: Dict[str, Any]   = self._load_json(self.artifacts_dir / "schema.json")

        # Derived config
        cfg: Dict[str, Any]        = self.metadata.get("config", {})
        self.problem_type: str     = cfg.get("problem_type", "classification_binary")
        self.modalities: List[str] = self.schema.get("global_modalities", ["tabular"])

        # Load preprocessors
        self.tabular_prep: Optional[Any] = self._load_tabular_prep()
        self.tokenizer: Optional[Any]    = self._load_tokenizer()
        self.target_encoder: Optional[Any] = self._load_target_encoder()
        self.probability_calibrator: Optional[Any] = self._load_probability_calibrator()

        # Load trained tabular encoder (GRN/MLP) if saved
        self._tabular_encoder: Optional[nn.Module] = self._load_tabular_encoder()

        # Reconstruct + load the fusion head
        self._head: nn.Module
        self.input_dims: Dict[str, int]
        self._head, self.input_dims = self._load_head()

        # Device placement
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._head.to(self.device)
        self._head.eval()

        # Place tabular encoder on device if loaded
        if self._tabular_encoder is not None:
            self._tabular_encoder.to(self.device)

        # Load frozen encoders for real multimodal inference
        self._text_encoder: Optional[nn.Module] = self._load_text_encoder()
        self._image_encoder: Optional[nn.Module] = self._load_image_encoder()
        self._image_preprocessor: Optional[Any] = None
        if self._image_encoder is not None:
            try:
                from preprocessing.image_preprocessor import ImagePreprocessor
                self._image_preprocessor = ImagePreprocessor()
            except Exception as exc:
                logger.warning("Could not load ImagePreprocessor: %s", exc)

        # Part A.1 — ULA token-mode: detect whether the loaded head uses UnifiedLatentFusion
        # with token_mode=True, so _build_batch can return full token sequences instead
        # of pooled vectors — enabling true cross-modal attention at inference time.
        try:
            from modelss.fusion import UnifiedLatentFusion as _ULAInfer
            _head_fusion = getattr(self._head, "fusion", None)
            self._use_token_sequences: bool = (
                isinstance(_head_fusion, _ULAInfer) and
                getattr(_head_fusion, "token_mode", False)
            )
        except Exception:
            self._use_token_sequences = False

        # Part A.1 — Load LoRA adapters if lora_text.pth / lora_image.pth present.
        # LoRA weights are the low-rank deltas trained during Phase 5 on top of
        # the frozen encoder backbone — they must be re-applied at inference time.
        _lora_cfg = self._load_json(self.artifacts_dir / "encoder_config.json") or {}
        _text_lora_cfg = _lora_cfg.get("text_encoder") or {}
        _image_lora_cfg = _lora_cfg.get("image_encoder") or {}
        try:
            from modelss.adapters.lora import apply_lora, load_lora_state_dict
            _lora_text_path = self.artifacts_dir / "lora_text.pth"
            if _lora_text_path.exists() and self._text_encoder is not None:
                _text_lora_r = int(_text_lora_cfg.get("lora_r", 8))
                _text_lora_alpha = float(_text_lora_cfg.get("lora_alpha", 16.0))
                apply_lora(self._text_encoder, r=_text_lora_r, alpha=_text_lora_alpha)
                load_lora_state_dict(
                    self._text_encoder,
                    torch.load(_lora_text_path, map_location="cpu", weights_only=True),
                )
                logger.info("InferenceEngine: LoRA text adapter loaded (r=%d)", _text_lora_r)
            _lora_image_path = self.artifacts_dir / "lora_image.pth"
            if _lora_image_path.exists() and self._image_encoder is not None:
                _image_lora_r = int(_image_lora_cfg.get("lora_r", _text_lora_cfg.get("lora_r", 8)))
                _image_lora_alpha = float(_image_lora_cfg.get("lora_alpha", _text_lora_cfg.get("lora_alpha", 16.0)))
                apply_lora(self._image_encoder, r=_image_lora_r, alpha=_image_lora_alpha)
                load_lora_state_dict(
                    self._image_encoder,
                    torch.load(_lora_image_path, map_location="cpu", weights_only=True),
                )
                logger.info("InferenceEngine: LoRA image adapter loaded (r=%d)", _image_lora_r)
        except Exception as _lora_inf_exc:
            logger.debug("LoRA inference load skipped: %s", _lora_inf_exc)

        logger.info(
            "InferenceEngine ready: model_id=%s  problem=%s  "
            "modalities=%s  input_dims=%s  device=%s  ula_token_mode=%s",
            model_id, self.problem_type, self.modalities,
            self.input_dims, self.device, self._use_token_sequences,
        )

    # ------------------------------------------------------------------ #
    # Public API – prediction
    # ------------------------------------------------------------------ #

    def predict_batch(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
        execution_context: Optional[Any] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run batch inference under ``torch.no_grad()``.

        Parameters
        ----------
        inputs : list[dict] or pd.DataFrame
            Raw feature values.  Each dict / row should contain the column
            names that were present in the original training data.
            Missing columns are zero-filled; extra columns are ignored.
        execution_context : ExecutionContext, optional
            If supplied, a decision log entry is written after inference.
        modality_mask : Dict[str, bool], optional
            G26: Zero-out specific modality tensors before the forward pass.
            Example: ``{"image": False}`` runs inference without the image
            modality (ablation / missing-modality robustness test).  Keys not
            present default to ``True`` (modality active).

        Returns
        -------
        dict with keys:
            ``predictions``  – list of int (classification) or float (regression)
            ``confidences``  – list of float (max class probability or 1.0 for regression)
            ``problem_type`` – str
            ``n_samples``    – int
            ``modality_mask`` – echoed back when a mask was applied
        """
        batch: Dict[str, torch.Tensor] = self._build_batch(inputs)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # G26: Apply modality mask — zero-out masked modality tensors
        if modality_mask:
            for mod_name, active in modality_mask.items():
                if not active and mod_name in batch:
                    logger.debug(
                        "G26: modality_mask zeroing '%s' tensor for ablation", mod_name
                    )
                    batch[mod_name] = torch.zeros_like(batch[mod_name])

        # Detect which modalities are genuinely present vs. dummy-filled
        trained_mods = set(getattr(self._head, "_keys", []) if self._head else [])
        present_mods = {k for k in batch if k in trained_mods and batch[k].abs().max() > 1e-6}
        if trained_mods and present_mods < trained_mods:
            absent = trained_mods - present_mods
            logger.warning("predict_batch: modalities absent or dummy-filled: %s", absent)

        with torch.no_grad():
            logits: torch.Tensor = self._head(batch)

        predictions, confidences = self._decode_logits(logits)

        # Apply inverse_transform to recover original label space
        pred_list = predictions.tolist()
        if self.target_encoder is not None:
            try:
                if (self.problem_type == "multilabel_classification"
                        and isinstance(self.target_encoder, dict)
                        and self.target_encoder.get("type") == "multilabel"):
                    # Custom dict encoder: decode multi-hot to label lists
                    all_labels = self.target_encoder["all_labels"]
                    preds_np = predictions.numpy()
                    pred_list = [
                        [all_labels[i] for i in range(preds_np.shape[1])
                         if preds_np[row, i] >= 1]
                        for row in range(preds_np.shape[0])
                    ]
                elif hasattr(self.target_encoder, "inverse_transform"):
                    if self.problem_type == "multilabel_classification":
                        pred_list = self.target_encoder.inverse_transform(
                            predictions.numpy()
                        )
                        pred_list = [list(row) for row in pred_list]
                    elif (self.problem_type.startswith("classification")
                          or self.problem_type == "classification_binary"):
                        pred_list = self.target_encoder.inverse_transform(
                            predictions.numpy()
                        ).tolist()
                    else:
                        # StandardScaler: reshape for inverse_transform
                        raw = predictions.numpy().reshape(-1, 1)
                        pred_list = self.target_encoder.inverse_transform(
                            raw
                        ).ravel().tolist()
            except Exception as exc:
                logger.warning("target_encoder inverse_transform failed: %s", exc)

        result = {
            "predictions":  pred_list,
            "confidences":  confidences.tolist(),
            "problem_type": self.problem_type,
            "n_samples":    len(pred_list),
        }
        if modality_mask:
            result["modality_mask"] = {k: bool(v) for k, v in modality_mask.items()}

        if execution_context is not None and hasattr(execution_context, "log_decision"):
            try:
                execution_context.log_decision(
                    "inference",
                    (
                        "Predictions generated: "
                        f"n_samples={len(pred_list)}, "
                        f"problem_type={self.problem_type}"
                        + (f", modality_mask={modality_mask}" if modality_mask else "")
                    ),
                    evidence=(
                        "active_prediction_model_id="
                        f"{getattr(execution_context, 'active_prediction_model_id', None)}"
                    ),
                )
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------ #
    # Public API – explainability
    # ------------------------------------------------------------------ #

    def generate_explanations(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
        target_class: int = 0,
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Compute Captum IntegratedGradients attributions.

        Gradients are enabled only during this call; ``predict_batch`` is
        not affected.

        Parameters
        ----------
        inputs       : raw inputs (same format as ``predict_batch``).
        target_class : class index for attribution (ignored for regression).
        n_steps      : number of integration steps (higher = more accurate).

        Returns
        -------
        dict with keys:
            ``method``       – "IntegratedGradients"
            ``target_class`` – int
            ``tabular``      – dict | None
            ``text``         – dict | None
        """
        try:
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError(
                "captum is required for XAI.  Install: pip install captum"
            )

        batch: Dict[str, torch.Tensor] = self._build_batch(inputs)
        tabular_tensor: Optional[torch.Tensor] = batch.get("tabular")

        explanations: Dict[str, Any] = {
            "method":       "IntegratedGradients",
            "target_class": target_class,
            "tabular":      None,
            "text":         None,
        }

        if tabular_tensor is None:
            logger.warning("generate_explanations: no tabular data – skipping IG")
            return explanations

        # Float tensor that accepts gradients
        tabular_ig: torch.Tensor = (
            tabular_tensor.to(self.device).float().requires_grad_(True)
        )

        # Pre-build frozen tensors for non-tabular modalities so the head
        # always receives its full expected input dict.  Use real encoder
        # outputs when available for more accurate attributions.
        frozen_extras: Dict[str, torch.Tensor] = {}
        if "text_pooled" in self.input_dims:
            if self._text_encoder is not None:
                text_vals = self._extract_text_values(inputs)
                if text_vals:
                    while len(text_vals) < len(tabular_ig):
                        text_vals.append("")
                    with torch.no_grad():
                        frozen_extras["text_pooled"] = (
                            self._text_encoder(text_vals).to(self.device).detach()
                        )
                else:
                    frozen_extras["text_pooled"] = torch.full(
                        (len(tabular_ig), self.TEXT_DIM), 1e-7, device=self.device
                    )
            else:
                frozen_extras["text_pooled"] = torch.full(
                    (len(tabular_ig), self.TEXT_DIM), 1e-7, device=self.device
                )
        if "image_pooled" in self.input_dims:
            image_tensor = self._extract_image_tensors(inputs)
            if image_tensor is not None and self._image_encoder is not None:
                with torch.no_grad():
                    frozen_extras["image_pooled"] = (
                        self._image_encoder(image_tensor.to(self.device)).detach()
                    )
            else:
                frozen_extras["image_pooled"] = torch.full(
                    (len(tabular_ig), self.IMAGE_DIM), 1e-7, device=self.device
                )

        # Captum forward: tabular tensor is the only differentiable input
        def _forward_tabular(tab: torch.Tensor) -> torch.Tensor:
            b: Dict[str, torch.Tensor] = {"tabular": tab}
            b.update(frozen_extras)
            out: torch.Tensor = self._head(b)
            if self.problem_type == "classification_binary":
                return torch.sigmoid(out.squeeze(-1)).unsqueeze(-1)
            if self.problem_type == "multilabel_classification":
                return torch.sigmoid(out)
            if self.problem_type.startswith("classification"):
                return torch.softmax(out, dim=-1)
            return out.squeeze(-1).unsqueeze(-1)

        ig = IntegratedGradients(_forward_tabular)
        baseline = torch.zeros_like(tabular_ig)
        tgt: Optional[int] = (
            target_class
            if self.problem_type.startswith("classification")
               or self.problem_type == "multilabel_classification"
            else None
        )

        try:
            attrs: torch.Tensor = ig.attribute(
                tabular_ig,
                baselines=baseline,
                target=tgt,
                n_steps=n_steps,
                return_convergence_delta=False,
            )
            attrs_np: np.ndarray = attrs.detach().cpu().numpy()
            # Mean absolute attribution per feature across the batch
            mean_attrs: List[float] = np.mean(np.abs(attrs_np), axis=0).tolist()
            feature_names: List[str] = self._get_tabular_feature_names(
                tabular_ig.shape[1]
            )
            explanations["tabular"] = {
                "feature_names":    feature_names,
                "attributions":     mean_attrs,
                "raw_attributions": attrs_np.tolist(),
            }
        except Exception as exc:
            logger.warning("Tabular IG attribution failed: %s", exc)

        # Approximate text token attributions when tokenizer is loaded
        if self.tokenizer is not None and "text_pooled" in self.input_dims:
            text_vals: List[str] = self._extract_text_values(inputs)
            if text_vals:
                explanations["text"] = self._token_attributions(
                    text=text_vals[0],
                    target_class=target_class,
                    n_steps=n_steps,
                    frozen_tabular=batch.get("tabular"),
                )

        # Image GradCAM — uses the last Conv2d in the image encoder
        if self._image_encoder is not None and "image_pooled" in self.input_dims:
            image_tensor: Optional[torch.Tensor] = self._extract_image_tensors(inputs)
            if image_tensor is not None:
                explanations["image"] = self._image_gradcam(
                    image_tensor[:1],     # first sample only
                    target_class=target_class,
                )

        return explanations

    def _image_gradcam(
        self,
        img_tensor: torch.Tensor,
        target_class: int = 0,
    ) -> Dict[str, Any]:
        """
        GradCAM spatial saliency map for image inputs.

        Hooks the last Conv2d in the image encoder, runs a forward-backward
        pass, and computes ReLU(Σ_c α_c · A_c) where α_c is the
        global-average-pooled gradient of the c-th activation map.

        Returns
        -------
        dict with keys:
            ``heatmap``       – list[list[float]] — 2D normalised saliency [0,1]
            ``heatmap_shape`` – [H, W]
            ``method``        – "GradCAM"
            ``input_shape``   – [C, H, W]
        """
        try:
            img = img_tensor.to(self.device).float()
            img.requires_grad_(True)

            # Locate the last Conv2d in the encoder
            last_conv: Optional[torch.nn.Module] = None
            for m in self._image_encoder.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = m

            if last_conv is None:
                # Part E — Attention Rollout for ViT encoders (Abnar & Zuidema, 2020)
                # When no Conv2d exists the encoder is a ViT; roll out attention maps
                # across all transformer layers to produce a patch-level saliency grid.
                return self._attention_rollout(img)

            activations: list = []
            gradients: list = []

            def _fwd(m, i, o):
                activations.clear(); activations.append(o.detach())

            def _bwd(m, gi, go):
                gradients.clear(); gradients.append(go[0].detach())

            fwd_h = last_conv.register_forward_hook(_fwd)
            bwd_h = last_conv.register_backward_hook(_bwd)

            try:
                with torch.enable_grad():
                    feat = self._image_encoder(img)          # (1, D)
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    # Use the most-activated dimension as the scalar to backprop
                    score = feat[0, target_class % feat.shape[-1]]
                    score.backward()
            finally:
                fwd_h.remove()
                bwd_h.remove()

            if not activations or not gradients:
                return {"gradcam_available": False, "note": "Hook capture failed."}

            act = activations[0][0]                                    # (C, H, W)
            grd = gradients[0][0]                                      # (C, H, W)
            weights = grd.mean(dim=(1, 2), keepdim=True)               # (C, 1, 1)
            cam = torch.nn.functional.relu((weights * act).sum(dim=0)) # (H, W)
            cam_np = cam.cpu().numpy()

            lo, hi = float(cam_np.min()), float(cam_np.max())
            cam_norm = ((cam_np - lo) / (hi - lo + 1e-8)).tolist()    # nested list

            return {
                "gradcam_available": True,
                "heatmap": cam_norm,
                "heatmap_shape": list(cam_np.shape),
                "method": "GradCAM",
                "input_shape": list(img_tensor.shape[1:]),
                "note": "GradCAM: ReLU(Σ_c α_c·A_c) over last Conv2d, normalised [0,1].",
            }

        except Exception as exc:
            logger.warning("Image GradCAM failed: %s", exc)
            return {"gradcam_available": False, "note": str(exc)}

    def _attention_rollout(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Attention Rollout XAI for ViT image encoders (Abnar & Zuidema, 2020).

        Registers forward hooks on every TransformerEncoderLayer to capture
        per-head attention weights, then recursively multiplies augmented
        matrices (A_l = 0.5·I + 0.5·mean_heads(attn)) across layers.  The
        CLS-to-patch row of the final product gives a spatial importance map.

        Returns the same dict shape as GradCAM so the frontend can render it
        identically, with ``method="AttentionRollout"``.
        """
        try:
            import torch.nn as _nn
            attns: list = []
            hooks: list = []

            # Collect attention weight tensors from each encoder layer
            _enc = self._image_encoder
            candidates = []
            for m in _enc.modules():
                if isinstance(m, _nn.TransformerEncoderLayer):
                    candidates.append(m)

            if not candidates:
                return {"gradcam_available": False, "note": "No TransformerEncoderLayer found for attention rollout."}

            def _make_hook(store):
                def _hook(module, inp, out):
                    # output_attentions path: out is (tensor, attn_weights)
                    if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                        store.append(out[1].detach().cpu())
                return _hook

            for layer in candidates:
                hooks.append(layer.register_forward_hook(_make_hook(attns)))

            try:
                with torch.no_grad():
                    img = img_tensor.to(self.device).float()
                    if hasattr(_enc, "forward"):
                        import inspect as _inspect
                        _params = _inspect.signature(_enc.forward).parameters
                        if "output_attentions" in _params:
                            _enc(img, output_attentions=True)
                        elif "return_all_tokens" in _params:
                            _enc(img, return_all_tokens=True)
                        else:
                            _enc(img)
                    else:
                        _enc(img)
            finally:
                for h in hooks:
                    h.remove()

            if not attns:
                return {"gradcam_available": False, "note": "Attention hook captured 0 tensors."}

            # Rollout: A_l = 0.5·I + 0.5·mean_heads(attn); product across layers
            T = attns[0].shape[-1]
            rollout = torch.eye(T)
            for a in attns:
                # a: (N, heads, T, T) — average over heads
                a_mean = a.mean(dim=1).squeeze(0)          # (T, T)
                a_aug = 0.5 * torch.eye(T) + 0.5 * a_mean
                rollout = a_aug @ rollout

            # CLS token (index 0) → all patch tokens (1:)
            cls_attn = rollout[0, 1:]                      # (P,)
            patch_grid = int(len(cls_attn) ** 0.5)
            if patch_grid * patch_grid != len(cls_attn):
                patch_grid = int(len(cls_attn) ** 0.5) + 1

            grid = cls_attn[:patch_grid * patch_grid].reshape(patch_grid, patch_grid).numpy()
            lo, hi = float(grid.min()), float(grid.max())
            grid_norm = ((grid - lo) / (hi - lo + 1e-8)).tolist()

            return {
                "gradcam_available": True,
                "heatmap": grid_norm,
                "heatmap_shape": [patch_grid, patch_grid],
                "method": "AttentionRollout",
                "input_shape": list(img_tensor.shape[1:]),
                "note": "Abnar & Zuidema (2020) Attention Rollout across ViT encoder layers.",
            }
        except Exception as exc:
            logger.warning("Attention rollout failed: %s", exc)
            return {"gradcam_available": False, "note": str(exc)}

    # ------------------------------------------------------------------ #
    # Artifact loaders
    # ------------------------------------------------------------------ #

    def _load_tabular_prep(self) -> Optional[Any]:
        path = self.artifacts_dir / "tabular_scaler.joblib"
        if not path.exists():
            return None
        try:
            import joblib
            prep = joblib.load(path)
            logger.info("Loaded tabular_scaler from %s", path)
            return prep
        except Exception as exc:
            logger.warning("Could not load tabular_scaler: %s", exc)
            return None

    def _load_tokenizer(self) -> Optional[Any]:
        tok_dir = self.artifacts_dir / "text_tokenizer"
        if not tok_dir.exists():
            return None
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(tok_dir))
            logger.info("Loaded HF tokenizer from %s", tok_dir)
            return tok
        except Exception as exc:
            logger.warning("Could not load tokenizer: %s", exc)
            return None

    def _load_target_encoder(self) -> Optional[Any]:
        path = self.artifacts_dir / "target_encoder.joblib"
        if not path.exists():
            return None
        try:
            import joblib
            enc = joblib.load(path)
            logger.info("Loaded target_encoder from %s", path)
            return enc
        except Exception as exc:
            logger.warning("Could not load target_encoder: %s", exc)
            return None

    def _load_probability_calibrator(self) -> Optional[Any]:
        path = self.artifacts_dir / "probability_calibrator.joblib"
        if not path.exists():
            return None
        try:
            import joblib
            calibrator = joblib.load(path)
            logger.info("Loaded probability_calibrator from %s", path)
            return calibrator
        except Exception as exc:
            logger.warning("Could not load probability_calibrator: %s", exc)
            return None

    def _load_tabular_encoder(self) -> Optional[nn.Module]:
        """Load trained tabular encoder (GRN/MLP) from saved state dict + config."""
        state_path = self.artifacts_dir / "tabular_encoder_state.pth"
        if not state_path.exists():
            return None

        # Read encoder config to determine class and input_dim
        enc_config = self._load_json(self.artifacts_dir / "encoder_config.json")
        tab_cfg = enc_config.get("tabular_encoder", {}) if enc_config else {}
        encoder_type = tab_cfg.get("type", "TabularEncoder")
        input_dim = tab_cfg.get("input_dim")

        if input_dim is None:
            # Infer from tabular preprocessor output
            if self.tabular_prep is not None:
                input_dim = self.tabular_prep.get_output_dim()
            else:
                logger.warning(
                    "Cannot determine tabular encoder input_dim "
                    "– skipping tabular encoder load"
                )
                return None

        try:
            if encoder_type == "GRNTabularEncoder":
                from models.encoders.tabular import GRNTabularEncoder
                encoder = GRNTabularEncoder(input_dim=input_dim)
            else:
                from models.encoders.tabular import TabularEncoder
                encoder = TabularEncoder(input_dim=input_dim)

            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict, strict=True)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False

            logger.info(
                "TabularEncoder loaded: type=%s  input_dim=%d  output_dim=%d",
                encoder_type, input_dim, encoder.get_output_dim(),
            )
            return encoder

        except Exception as exc:
            logger.warning("Could not load TabularEncoder: %s", exc)
            return None

    def _load_text_encoder(self) -> Optional[nn.Module]:
        """Load frozen TextEncoder from saved state dict or recreate from pretrained."""
        if "text" not in self.modalities:
            return None
        try:
            from models.encoders.text import TextEncoder

            # Read encoder config for model name and max_length
            enc_config = self._load_json(self.artifacts_dir / "encoder_config.json")
            text_cfg = enc_config.get("text_encoder", {})
            model_name = text_cfg.get("model_name", "bert-base-uncased")
            max_length = text_cfg.get("max_length", 128)

            # Fallback: infer model name from saved tokenizer
            if not text_cfg and self.tokenizer is not None:
                model_name = getattr(self.tokenizer, "name_or_path", "bert-base-uncased")

            encoder = TextEncoder(
                model_name=model_name,
                max_length=max_length,
                freeze_backbone=True,
            )
            if text_cfg.get("lora_enabled"):
                try:
                    from modelss.adapters.lora import apply_lora
                    apply_lora(
                        encoder,
                        r=int(text_cfg.get("lora_r", 8)),
                        alpha=float(text_cfg.get("lora_alpha", 16.0)),
                    )
                except Exception as lora_shape_exc:
                    logger.debug("TextEncoder LoRA shape preparation skipped: %s", lora_shape_exc)

            # Load saved state dict if available
            state_path = self.artifacts_dir / "text_encoder_state.pth"
            if state_path.exists():
                state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
                encoder.load_state_dict(state_dict, strict=True)
                logger.info("TextEncoder loaded from saved state dict")
            else:
                logger.info(
                    "TextEncoder recreated from pretrained '%s' (no saved state dict)",
                    model_name,
                )

            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.to(self.device)
            return encoder

        except Exception as exc:
            logger.warning("Could not load TextEncoder: %s", exc)
            return None

    def _load_image_encoder(self) -> Optional[nn.Module]:
        """Load frozen ImageEncoder from saved state dict, with pretrained fallback."""
        if "image" not in self.modalities:
            return None

        state_path = self.artifacts_dir / "image_encoder_state.pth"
        enc_config = self._load_json(self.artifacts_dir / "encoder_config.json")
        image_cfg = enc_config.get("image_encoder", {}) if enc_config else {}
        image_name = str(image_cfg.get("model_name") or "ResNet-50")

        def _build_from_registry() -> nn.Module:
            try:
                import config.encoder_plugins  # noqa: F401
                from automl.jit_encoder_selector import VISION_REGISTRY

                for spec in VISION_REGISTRY:
                    if str(spec.name).lower() == image_name.lower():
                        return spec.factory()
            except Exception as registry_exc:
                logger.debug("Vision registry lookup skipped for '%s': %s", image_name, registry_exc)

            from models.encoders.image import ImageEncoder
            return ImageEncoder(pretrained=True, freeze_backbone=True)

        try:
            encoder = _build_from_registry()
            if image_cfg.get("lora_enabled"):
                try:
                    from modelss.adapters.lora import apply_lora
                    apply_lora(
                        encoder,
                        r=int(image_cfg.get("lora_r", 8)),
                        alpha=float(image_cfg.get("lora_alpha", 16.0)),
                    )
                except Exception as lora_shape_exc:
                    logger.debug("ImageEncoder LoRA shape preparation skipped: %s", lora_shape_exc)
            if state_path.exists():
                state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
                encoder.load_state_dict(state_dict, strict=True)
                logger.info("ImageEncoder loaded from saved state dict (%s)", image_name)
            else:
                logger.warning(
                    "ImageEncoder state dict not found at %s. Recreated '%s' from registry/pretrained weights.",
                    state_path,
                    image_name,
                )

            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.to(self.device)
            return encoder

        except Exception as exc:
            logger.warning("Could not load ImageEncoder: %s", exc)
            return None

    def _load_head(self) -> Tuple[nn.Module, Dict[str, int]]:
        """
        Reconstruct ``_MultimodalHead`` from the saved ``ApexLightningModule``
        state dict.

        The saved state dict uses ``"model.layers.*"`` keys because
        ``ApexLightningModule`` stores the head under ``self.model``.  The
        ``"model."`` prefix is stripped to obtain the bare head state dict,
        then layer shapes are inspected to recover ``hidden_dim``,
        ``total_dim``, and ``num_outputs`` without requiring them stored
        separately.
        """
        weights_path = self.artifacts_dir / "model_weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"model_weights.pth not found at {weights_path}. "
                "Phase 7 must complete successfully before inference."
            )

        full_state: Dict[str, torch.Tensor] = torch.load(
            weights_path, map_location="cpu", weights_only=True,
        )

        # Strip the "model." prefix emitted by ApexLightningModule
        head_state: Dict[str, torch.Tensor] = {
            k[len("model."):]: v
            for k, v in full_state.items()
            if k.startswith("model.")
        }
        if not head_state:
            # Saved as raw head state dict (no LightningModule wrapper)
            head_state = dict(full_state)

        # Infer architecture from tensor shapes.
        # Standard _MultimodalHead layout:
        #   layers.0 → Linear(total_dim → hidden_dim)
        #   layers.3 → Linear(hidden_dim → num_outputs)
        # Fallback: scan for first and last Linear weight keys.
        w0: Optional[torch.Tensor] = None
        b_last: Optional[torch.Tensor] = None

        # Try canonical key patterns first
        for first_key in ("layers.0.weight",):
            if first_key in head_state:
                w0 = head_state[first_key]
                break
        for last_key in ("layers.3.bias", "layers.4.bias", "layers.5.bias"):
            if last_key in head_state:
                b_last = head_state[last_key]
                break

        # Fallback: find first and last *.weight / *.bias pairs
        if w0 is None or b_last is None:
            weight_keys = sorted(
                [k for k in head_state if k.endswith(".weight") and head_state[k].ndim == 2]
            )
            bias_keys = sorted(
                [k for k in head_state if k.endswith(".bias")]
            )
            if weight_keys and w0 is None:
                w0 = head_state[weight_keys[0]]
            if bias_keys and b_last is None:
                b_last = head_state[bias_keys[-1]]

        if w0 is None or b_last is None:
            raise RuntimeError(
                "Cannot infer _MultimodalHead architecture from state dict keys: "
                f"{list(head_state.keys())}. "
                "Ensure the model was saved with the standard _MultimodalHead."
            )

        head_arch = (self.metadata or {}).get("head_architecture") or {}
        try:
            hidden_dim = int(head_arch.get("hidden_dim", w0.shape[0]))
        except (TypeError, ValueError):
            hidden_dim = int(w0.shape[0])
        try:
            total_dim = int(head_arch.get("total_dim", w0.shape[1]))
        except (TypeError, ValueError):
            total_dim = int(w0.shape[1])
        try:
            num_outputs = int(head_arch.get("num_outputs", b_last.shape[0]))
        except (TypeError, ValueError):
            num_outputs = int(b_last.shape[0])

        # Bug 1 fix: try loading persisted input_dims.json first (saved by Phase 7).
        # This is required for non-ResNet encoders (DINOv2=768, SigLIP=768) where
        # the hard-coded heuristic (TEXT=768, IMAGE=512) produces wrong dimensions.
        input_dims: Dict[str, int] = {}
        _idims_path = self.artifacts_dir / "input_dims.json"
        if _idims_path.exists():
            try:
                _persisted = self._load_json(_idims_path)
                if isinstance(_persisted, dict) and _persisted:
                    input_dims = {str(k): int(v) for k, v in _persisted.items()}
                    logger.info("input_dims loaded from persisted file: %s", input_dims)
            except Exception as _load_exc:
                logger.warning("input_dims.json load failed: %s — falling back to reconstruction", _load_exc)

        if not input_dims:
            # Fallback: heuristic reconstruction from preprocessor output dims
            input_dims = self._build_input_dims(total_dim)
            computed_total: int = sum(input_dims.values())
            if computed_total != total_dim:
                raise RuntimeError(
                    f"input_dims reconstruction mismatch: computed {computed_total} != "
                    f"saved total_dim {total_dim}. Re-run Phase 7 training to persist "
                    "input_dims.json with the correct encoder output dimensions."
                )

        from automl.trainer import _MultimodalHead

        fusion_meta = dict((self.metadata or {}).get("fusion", {}) or {})
        fusion_summary = dict(fusion_meta.get("summary", {}) or {})
        fusion_strategy = _canonical_fusion_strategy(
            fusion_meta.get("strategy")
            or (self.metadata or {}).get("fusion_strategy")
            or fusion_summary.get("strategy")
            or fusion_summary.get("fusion_type")
            or "concatenation"
        )
        fusion_config: Dict[str, Any] = {}
        if fusion_strategy == "ula":
            fusion_config.update(self._load_json(self.artifacts_dir / "ula_config.json") or {})

        head = _MultimodalHead(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_outputs=num_outputs,
            fusion_strategy=fusion_strategy,
            fusion_config=fusion_config,
        )
        head.load_state_dict(head_state, strict=True)
        head.eval()

        logger.info(
            "Head reconstructed: input_dims=%s  hidden=%d  outputs=%d",
            input_dims, hidden_dim, num_outputs,
        )
        return head, input_dims

    # ------------------------------------------------------------------ #
    # Batch construction
    # ------------------------------------------------------------------ #

    def _build_batch(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert heterogeneous raw inputs into a model-ready tensor dict.

        Rules
        -----
        * Tabular columns: aligned to the scaler's ``_feature_names_in`` list;
          missing columns are zero-filled, extra columns are dropped.
        * Text: encoded through the frozen TextEncoder (BERT) when available;
          falls back to 1e-7 fill if no encoder is loaded.
        * Image: loaded from paths, preprocessed, encoded through the frozen
          ImageEncoder (ResNet50) when available; falls back to 1e-7 fill.
        * All output tensors are ``torch.float32``.
        """
        df: pd.DataFrame = (
            inputs.copy() if isinstance(inputs, pd.DataFrame)
            else pd.DataFrame(inputs)
        )
        N = len(df)

        batch: Dict[str, torch.Tensor] = {}

        # ── Tabular ─────────────────────────────────────────────────────
        if "tabular" in self.input_dims:
            expected_cols: Optional[List[str]] = getattr(
                self.tabular_prep, "_feature_names_in", None
            )
            if expected_cols is not None:
                # Zero-fill missing training columns; drop unrecognised ones
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                tab_df: pd.DataFrame = df[expected_cols].fillna(0.0)
            else:
                tab_df = df.select_dtypes(include=[np.number]).fillna(0.0)

            if self.tabular_prep is not None and not tab_df.empty:
                try:
                    arr: np.ndarray = self.tabular_prep.transform(tab_df)
                except Exception as exc:
                    logger.warning(
                        "_build_batch: tabular transform failed (%s) – using raw values",
                        exc,
                    )
                    arr = tab_df.values.astype(np.float32)
            else:
                arr = (
                    tab_df.values.astype(np.float32)
                    if not tab_df.empty
                    else np.zeros((N, self.input_dims["tabular"]), dtype=np.float32)
                )

            batch["tabular"] = torch.tensor(arr, dtype=torch.float32)

            # Run through trained tabular encoder (GRN/MLP) if available
            if self._tabular_encoder is not None:
                with torch.no_grad():
                    batch["tabular"] = self._tabular_encoder(
                        batch["tabular"].to(self.device)
                    ).cpu()

        # ── Text: encode through BERT (pooled CLS or full token sequence for ULA) ──
        if "text_pooled" in self.input_dims:
            text_values: List[str] = self._extract_text_values(inputs)
            if text_values and self._text_encoder is not None:
                while len(text_values) < N:
                    text_values.append("")
                with torch.no_grad():
                    if self._use_token_sequences and hasattr(self._text_encoder, "transformer"):
                        # ULA token-mode: full last_hidden_state (N, T, hidden_size)
                        _tok = getattr(self._text_encoder, "tokenizer", None)
                        if _tok is not None:
                            _enc = _tok(
                                text_values, return_tensors="pt", padding=True, truncation=True
                            ).to(self.device)
                            _out = self._text_encoder.transformer(
                                input_ids=_enc["input_ids"],
                                attention_mask=_enc.get("attention_mask"),
                            )
                            batch["text_pooled"] = _out.last_hidden_state  # (N, T, 768)
                        else:
                            batch["text_pooled"] = self._text_encoder(text_values).to(self.device)
                    else:
                        batch["text_pooled"] = self._text_encoder(text_values).to(self.device)
            else:
                if not text_values:
                    logger.debug("_build_batch: no text values found in input")
                if self._text_encoder is None:
                    # Bug 10: upgrade to WARNING — silent dummy fill corrupts predictions
                    logger.warning(
                        "Head expects 'text_pooled' but no TextEncoder loaded — "
                        "using dummy embeddings (accuracy severely degraded). "
                        "Ensure text_encoder_state.pth is in the model artifacts directory."
                    )
                batch["text_pooled"] = torch.full(
                    (N, self.TEXT_DIM), 1e-7, dtype=torch.float32
                )

        # ── Image: encode through encoder (pooled or patch tokens for ULA) ──
        if "image_pooled" in self.input_dims:
            image_tensor: Optional[torch.Tensor] = self._extract_image_tensors(inputs)
            if image_tensor is not None and self._image_encoder is not None:
                with torch.no_grad():
                    if self._use_token_sequences:
                        import inspect as _inspect
                        _sig = _inspect.signature(self._image_encoder.forward)
                        if "return_all_tokens" in _sig.parameters:
                            # ULA token-mode: patch sequence (N, P, D)
                            batch["image_pooled"] = self._image_encoder(
                                image_tensor.to(self.device), return_all_tokens=True
                            )
                        else:
                            batch["image_pooled"] = self._image_encoder(
                                image_tensor.to(self.device)
                            )
                    else:
                        batch["image_pooled"] = self._image_encoder(
                            image_tensor.to(self.device)
                        )  # [N, 512]
            else:
                if self._image_encoder is None and image_tensor is None:
                    # Bug 10: upgrade to WARNING — silent dummy fill corrupts predictions
                    logger.warning(
                        "Head expects 'image_pooled' but no ImageEncoder loaded or no image "
                        "inputs provided — using dummy embeddings (accuracy severely degraded). "
                        "Ensure image_encoder_state.pth is in the model artifacts directory."
                    )
                batch["image_pooled"] = torch.full(
                    (N, self.IMAGE_DIM), 1e-7, dtype=torch.float32
                )

        return batch

    # ------------------------------------------------------------------ #
    # Logit decoding
    # ------------------------------------------------------------------ #

    def _decode_logits(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw model output tensors to (predictions, confidences).

        Returns
        -------
        predictions  : long tensor for classification, float for regression
        confidences  : max class probability for classification, 1.0 for regression
        """
        if self.problem_type == "classification_binary":
            probs_np = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()  # (N,)
            if self.probability_calibrator is not None:
                try:
                    probs_np = self.probability_calibrator.calibrate(
                        logits.detach().cpu().numpy(),
                        probs_np,
                    )
                except Exception as exc:
                    logger.warning("Binary calibration failed: %s", exc)

            probs = torch.tensor(probs_np, dtype=torch.float32)
            preds = (probs >= 0.5).long()
            confidences = torch.where(preds.bool(), probs, 1.0 - probs)
        elif self.problem_type == "multilabel_classification":
            probs_np = torch.sigmoid(logits).detach().cpu().numpy()  # (N, C)
            if self.probability_calibrator is not None:
                try:
                    probs_np = self.probability_calibrator.calibrate(
                        logits.detach().cpu().numpy(),
                        probs_np,
                    )
                except Exception as exc:
                    logger.warning("Multilabel calibration failed: %s", exc)

            probs = torch.tensor(probs_np, dtype=torch.float32)
            preds = (probs >= 0.5).long()
            confidences = probs
        elif self.problem_type.startswith("classification"):
            probs_np = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # (N, C)
            if self.probability_calibrator is not None:
                try:
                    probs_np = self.probability_calibrator.calibrate(
                        logits.detach().cpu().numpy(),
                        probs_np,
                    )
                except Exception as exc:
                    logger.warning("Multiclass calibration failed: %s", exc)

            probs = torch.tensor(probs_np, dtype=torch.float32)
            confidences, preds = probs.max(dim=-1)
        else:
            preds       = logits.squeeze(-1)
            confidences = torch.ones_like(preds)

        return preds.cpu(), confidences.cpu()

    # ------------------------------------------------------------------ #
    # Helper: input_dims reconstruction
    # ------------------------------------------------------------------ #

    def _build_input_dims(self, state_dict_total_dim: int) -> Dict[str, int]:
        """
        Derive ``input_dims`` from the loaded tabular scaler + schema modalities.
        Falls back to ``{"tabular": state_dict_total_dim}`` when no scaler exists.
        """
        dims: Dict[str, int] = {}
        if self.tabular_prep is not None:
            if self._tabular_encoder is not None:
                # Tabular encoder projects preprocessor output to a fixed dim
                dims["tabular"] = self._tabular_encoder.get_output_dim()
            else:
                dims["tabular"] = self.tabular_prep.get_output_dim()
        if "text" in self.modalities:
            dims["text_pooled"] = self.TEXT_DIM
        if "image" in self.modalities:
            dims["image_pooled"] = self.IMAGE_DIM
        if not dims:
            dims = {"tabular": state_dict_total_dim}
        return dims

    # ------------------------------------------------------------------ #
    # Helper: tabular feature names
    # ------------------------------------------------------------------ #

    def _get_tabular_feature_names(self, n_features: int) -> List[str]:
        """
        Return feature names from the fitted ColumnTransformer if available,
        or generic ``feature_0 … feature_N-1`` labels otherwise.
        """
        if self.tabular_prep is not None:
            transformer = getattr(self.tabular_prep, "_transformer", None)
            if transformer is not None:
                try:
                    return list(transformer.get_feature_names_out())
                except Exception:
                    pass
        return [f"feature_{i}" for i in range(n_features)]

    # ------------------------------------------------------------------ #
    # Helper: token-level attribution (dispatcher)
    # ------------------------------------------------------------------ #

    def _token_attributions(
        self,
        text: str,
        target_class: int,
        n_steps: int,
        frozen_tabular: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute per-token IG attributions.

        Tries the real BERT word-embedding pathway first (accurate); falls
        back to the approximate random-embedding method when no text encoder
        is loaded.
        """
        if self._text_encoder is not None:
            result = self._token_attributions_bert(
                text, target_class, n_steps, frozen_tabular,
            )
            if result is not None:
                return result

        return self._token_attributions_approximate(
            text, target_class, n_steps, frozen_tabular,
        )

    # ------------------------------------------------------------------ #
    # Real BERT token-level attribution
    # ------------------------------------------------------------------ #

    def _token_attributions_bert(
        self,
        text: str,
        target_class: int,
        n_steps: int,
        frozen_tabular: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Token-level IG attributions through real BERT word embeddings.

        Replaces the token-ID lookup with ``inputs_embeds`` so that Captum
        can differentiate through the full encoder → fusion head path.
        Gradients flow through the frozen BERT computation graph (parameter
        grads are not accumulated, only input-embedding grads are needed).
        """
        if self._text_encoder is None:
            return None

        try:
            from captum.attr import IntegratedGradients

            tokenizer = self._text_encoder.tokenizer
            max_length = self._text_encoder.max_length

            enc = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            input_ids: torch.Tensor = enc["input_ids"].to(self.device)
            attention_mask: torch.Tensor = enc["attention_mask"].to(self.device)
            tokens: List[str] = tokenizer.convert_ids_to_tokens(
                input_ids[0].tolist()
            )

            # Access real BERT word embeddings
            word_embeddings: nn.Embedding = (
                self._text_encoder.transformer.embeddings.word_embeddings
            )
            input_embeds: torch.Tensor = (
                word_embeddings(input_ids).detach().requires_grad_(True)
            )  # (1, seq_len, hidden_size)
            baseline_embeds = torch.zeros_like(input_embeds)

            _frozen_tab: Optional[torch.Tensor] = (
                frozen_tabular[:1].to(self.device).float()
                if frozen_tabular is not None else None
            )

            def _forward_text_bert(embeds: torch.Tensor) -> torch.Tensor:
                # Forward through BERT using inputs_embeds (bypasses embedding lookup)
                outputs = self._text_encoder.transformer(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                )
                last_hidden: torch.Tensor = outputs.last_hidden_state

                # Pool: CLS for encoder models, last-token for causal
                if self._text_encoder._is_causal:
                    seq_lens = attention_mask.sum(dim=1) - 1
                    batch_idx = torch.arange(
                        last_hidden.size(0), device=self.device
                    )
                    pooled = last_hidden[batch_idx, seq_lens]
                else:
                    pooled = last_hidden[:, 0, :]

                # Optional projection (e.g. bert-large → 768)
                if self._text_encoder._projection is not None:
                    pooled = self._text_encoder._projection(pooled)

                # Build fusion head input
                b: Dict[str, torch.Tensor] = {"text_pooled": pooled}
                if "tabular" in self.input_dims:
                    b["tabular"] = (
                        _frozen_tab
                        if _frozen_tab is not None
                        else torch.full(
                            (1, self.input_dims["tabular"]), 1e-7,
                            device=self.device,
                        )
                    )
                if "image_pooled" in self.input_dims:
                    b["image_pooled"] = torch.full(
                        (1, self.IMAGE_DIM), 1e-7, device=self.device,
                    )

                out: torch.Tensor = self._head(b)
                if self.problem_type == "classification_binary":
                    return torch.sigmoid(out.squeeze(-1)).unsqueeze(-1)
                if self.problem_type == "multilabel_classification":
                    return torch.sigmoid(out)
                if self.problem_type.startswith("classification"):
                    return torch.softmax(out, dim=-1)
                return out.squeeze(-1).unsqueeze(-1)

            tgt: Optional[int] = (
                target_class
                if self.problem_type.startswith("classification")
                   or self.problem_type == "multilabel_classification"
                else None
            )

            ig = IntegratedGradients(_forward_text_bert)
            attrs: torch.Tensor = ig.attribute(
                input_embeds,
                baselines=baseline_embeds,
                target=tgt,
                n_steps=n_steps,
                return_convergence_delta=False,
            )

            # Sum along embedding dim → per-token scalar salience
            token_attrs: np.ndarray = (
                attrs.detach().cpu().squeeze(0).sum(dim=-1).numpy()
            )

            # Exclude padding tokens
            pad_id: int = tokenizer.pad_token_id or 0
            non_pad_idx: List[int] = [
                i for i, tid in enumerate(input_ids[0].tolist())
                if tid != pad_id
            ]

            return {
                "tokens":       [tokens[i]             for i in non_pad_idx],
                "attributions": [float(token_attrs[i]) for i in non_pad_idx],
                "note": (
                    "Token attributions computed via IntegratedGradients "
                    "through the real BERT encoder word embeddings and "
                    "fusion head."
                ),
            }

        except Exception as exc:
            logger.warning("Real BERT token attribution failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Approximate token-level attribution (fallback)
    # ------------------------------------------------------------------ #

    def _token_attributions_approximate(
        self,
        text: str,
        target_class: int,
        n_steps: int,
        frozen_tabular: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Approximate token-level Captum IG attribution via a simulated
        embedding layer.

        Used as a fallback when no real BERT text encoder is loaded.
        A random Gaussian ``nn.Embedding(vocab_size, TEXT_DIM)`` is
        constructed, token IDs are embedded and mean-pooled to TEXT_DIM,
        then routed through the fusion head.
        """
        if self.tokenizer is None:
            return None

        try:
            from captum.attr import IntegratedGradients

            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.schema.get("text_max_length", 128),
            )
            input_ids: torch.Tensor = enc["input_ids"].to(self.device)  # (1, 128)
            tokens: List[str] = self.tokenizer.convert_ids_to_tokens(
                input_ids[0].tolist()
            )

            # Simulated embedding layer: random init, not task-trained
            emb_layer = nn.Embedding(
                self.tokenizer.vocab_size, self.TEXT_DIM
            ).to(self.device)
            nn.init.normal_(emb_layer.weight, mean=0.0, std=0.02)

            _frozen_tab: Optional[torch.Tensor] = (
                frozen_tabular[:1].to(self.device).float()
                if frozen_tabular is not None else None
            )

            def _forward_text_embed(emb: torch.Tensor) -> torch.Tensor:
                pooled: torch.Tensor = emb.mean(dim=1)         # (1, TEXT_DIM)
                b: Dict[str, torch.Tensor] = {"text_pooled": pooled}
                if "tabular" in self.input_dims:
                    b["tabular"] = (
                        _frozen_tab
                        if _frozen_tab is not None
                        else torch.full((1, self.input_dims["tabular"]), 1e-7, device=self.device)
                    )
                if "image_pooled" in self.input_dims:
                    b["image_pooled"] = torch.full((1, self.IMAGE_DIM), 1e-7, device=self.device)
                out: torch.Tensor = self._head(b)
                if self.problem_type == "classification_binary":
                    return torch.sigmoid(out.squeeze(-1)).unsqueeze(-1)
                if self.problem_type == "multilabel_classification":
                    return torch.sigmoid(out)
                if self.problem_type.startswith("classification"):
                    return torch.softmax(out, dim=-1)
                return out.squeeze(-1).unsqueeze(-1)

            input_emb: torch.Tensor = (
                emb_layer(input_ids).detach().requires_grad_(True)
            )                                                   # (1, 128, TEXT_DIM)
            baseline_emb = torch.zeros_like(input_emb)
            tgt: Optional[int] = (
                target_class
                if self.problem_type.startswith("classification")
                   or self.problem_type == "multilabel_classification"
                else None
            )

            ig = IntegratedGradients(_forward_text_embed)
            attrs: torch.Tensor = ig.attribute(
                input_emb,
                baselines=baseline_emb,
                target=tgt,
                n_steps=n_steps,
                return_convergence_delta=False,
            )
            # Sum along embedding dim to get per-token scalar salience
            token_attrs: np.ndarray = (
                attrs.detach().cpu().squeeze(0).sum(dim=-1).numpy()   # (128,)
            )

            # Exclude padding tokens
            pad_id: int = self.tokenizer.pad_token_id or 0
            non_pad_idx: List[int] = [
                i for i, tid in enumerate(input_ids[0].tolist())
                if tid != pad_id
            ]

            return {
                "tokens":       [tokens[i]             for i in non_pad_idx],
                "attributions": [float(token_attrs[i]) for i in non_pad_idx],
                "note": (
                    "Approximate token attributions (fallback): computed "
                    "via a random embedding layer, not the real BERT encoder. "
                    "Retrain and save encoder weights for accurate attributions."
                ),
            }

        except Exception as exc:
            logger.warning("Token attribution failed: %s", exc)
            return None
        finally:
            # Free the per-call simulated embedding layer from GPU
            try:
                del emb_layer
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Helper: extract raw text values from inputs
    # ------------------------------------------------------------------ #

    def _extract_text_values(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> List[str]:
        """Extract text values using schema-detected text columns with fallbacks."""
        per_ds = self.schema.get("per_dataset", [{}])
        text_cols: List[str] = []
        for ds_entry in per_ds:
            detected = ds_entry.get("detected_columns", {}) if isinstance(ds_entry, dict) else {}
            for col in detected.get("text", []):
                if col not in text_cols:
                    text_cols.append(col)

        if isinstance(inputs, pd.DataFrame):
            df = inputs
        else:
            df = pd.DataFrame(inputs)

        candidate_cols: List[str] = []
        for col in text_cols + ["text", "report", "description", "content", "body"]:
            if col in df.columns and col not in candidate_cols:
                candidate_cols.append(col)

        if not candidate_cols:
            return []

        subset = df[candidate_cols]
        output: List[str] = []
        for _, row in subset.iterrows():
            parts: List[str] = []
            for val in row.tolist():
                if pd.isna(val):
                    continue
                text = str(val).strip()
                if text.lower() in {"", "nan", "none", "null", "<na>"}:
                    continue
                parts.append(text)
            output.append(" ".join(parts))

        return output

    def _extract_image_tensors(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Optional[torch.Tensor]:
        """Load images from paths, preprocess, and stack into a batch tensor."""
        if self._image_preprocessor is None:
            return None

        per_ds = self.schema.get("per_dataset", [{}])
        image_cols: List[str] = []
        for ds_entry in per_ds:
            detected = ds_entry.get("detected_columns", {}) if isinstance(ds_entry, dict) else {}
            for col in detected.get("image", []):
                if col not in image_cols:
                    image_cols.append(col)

        if isinstance(inputs, pd.DataFrame):
            df = inputs
        else:
            df = pd.DataFrame(inputs)

        candidate_cols: List[str] = []
        for candidate in image_cols + ["image_path", "image", "img_path", "file_path"]:
            if candidate in df.columns and candidate not in candidate_cols:
                candidate_cols.append(candidate)

        if not candidate_cols:
            return None

        from PIL import Image as PILImage

        tensors: List[torch.Tensor] = []
        for _, row in df[candidate_cols].iterrows():
            path_val: Optional[str] = None
            for raw in row.tolist():
                if pd.isna(raw):
                    continue
                candidate_path = str(raw).strip()
                if candidate_path.lower() in {"", "nan", "none", "null", "<na>"}:
                    continue
                path_val = candidate_path
                break

            try:
                if not path_val:
                    raise ValueError("no usable image path in row")
                pil_img = PILImage.open(str(path_val)).convert("RGB")
                tensor = self._image_preprocessor.preprocess(pil_img)
                tensors.append(tensor)
            except Exception as exc:
                logger.warning("Image load failed for '%s': %s", path_val, exc)
                tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))

        if not tensors:
            return None

        result_stack = torch.stack(tensors)

        # Bug 11: track image path failure rate and warn when degraded
        _zero = torch.zeros(3, 224, 224)
        _fail_count = sum(1 for t in tensors if torch.equal(t, _zero))
        if _fail_count > 0:
            _rate = _fail_count / len(tensors)
            _log = logger.warning if _rate > 0.3 else logger.info
            _log(
                "Image loading: %d/%d paths failed (%.0f%%)%s",
                _fail_count, len(tensors), _rate * 100,
                " — predictions may be unreliable" if _rate > 0.3 else "",
            )

        return result_stack

    # ------------------------------------------------------------------ #
    # Static: safe JSON loader
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("JSON load failed for %s: %s", path, exc)
            return {}
