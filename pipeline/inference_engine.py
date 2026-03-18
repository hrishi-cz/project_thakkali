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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight vision encoder wrapper for inference (mirrors JIT selector)
# ---------------------------------------------------------------------------

class _InferenceVisionWrapper(nn.Module):
    """Thin wrapper so non-ImageEncoder vision backbones share the same interface."""

    def __init__(self, backbone: nn.Module, projection: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)

    def get_output_dim(self) -> int:
        return self._output_dim


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
        registry_root: Path = Path("models") / "registry" / model_id
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
        self._image_encoder_name: str = "ResNet-50"  # updated by _load_image_encoder
        self._image_encoder: Optional[nn.Module] = self._load_image_encoder()
        self._image_preprocessor: Optional[Any] = None
        if self._image_encoder is not None:
            try:
                from preprocessing.image_preprocessor import ImagePreprocessor
                self._image_preprocessor = ImagePreprocessor()
            except Exception as exc:
                logger.warning("Could not load ImagePreprocessor: %s", exc)

        logger.info(
            "InferenceEngine ready: model_id=%s  problem=%s  "
            "modalities=%s  input_dims=%s  device=%s",
            model_id, self.problem_type, self.modalities,
            self.input_dims, self.device,
        )

    # ------------------------------------------------------------------ #
    # Public API – prediction
    # ------------------------------------------------------------------ #

    def predict_batch(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run batch inference under ``torch.no_grad()``.

        Parameters
        ----------
        inputs : list[dict] or pd.DataFrame
            Raw feature values.  Each dict / row should contain the column
            names that were present in the original training data.
            Missing columns are zero-filled; extra columns are ignored.

        Returns
        -------
        dict with keys:
            ``predictions``  – list of int (classification) or float (regression)
            ``confidences``  – list of float (max class probability or 1.0 for regression)
            ``problem_type`` – str
            ``n_samples``    – int
        """
        batch: Dict[str, torch.Tensor] = self._build_batch(inputs)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        try:
            with torch.no_grad():
                logits: torch.Tensor = self._head(batch)

            predictions, confidences = self._decode_logits(logits, threshold=threshold)

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

            return {
                "predictions":  pred_list,
                "confidences":  confidences.tolist(),
                "problem_type": self.problem_type,
                "n_samples":    len(pred_list),
            }
        finally:
            del batch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Public API – explainability
    # ------------------------------------------------------------------ #

    def generate_explanations(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
        target_class: int = 0,
        n_steps: int = 50,
        method: str = "ig",
    ) -> Dict[str, Any]:
        """
        Compute Captum attributions using the selected method.

        Gradients are enabled only during this call; ``predict_batch`` is
        not affected.

        Parameters
        ----------
        inputs       : raw inputs (same format as ``predict_batch``).
        target_class : class index for attribution (ignored for regression).
        n_steps      : number of integration steps (higher = more accurate).
        method       : attribution method — "ig", "gradient_shap", "saliency",
                       "occlusion", "feature_ablation", or "smoothgrad".

        Returns
        -------
        dict with keys:
            ``method``             – display name of the attribution method
            ``target_class``       – int
            ``tabular``            – dict | None
            ``text``               – dict | None
            ``image``              – dict | None
            ``attention``          – dict | None
            ``convergence_delta``  – float | None (for IG/GradientShap)
        """
        _METHOD_DISPLAY = {
            "ig": "IntegratedGradients",
            "gradient_shap": "GradientShap",
            "saliency": "Saliency",
            "occlusion": "Occlusion",
            "feature_ablation": "FeatureAblation",
            "smoothgrad": "SmoothGrad",
        }

        try:
            import captum.attr  # noqa: F401 – verify captum available
        except ImportError:
            raise ImportError(
                "captum is required for XAI.  Install: pip install captum"
            )

        batch: Dict[str, torch.Tensor] = self._build_batch(inputs)
        tabular_tensor: Optional[torch.Tensor] = batch.get("tabular")

        explanations: Dict[str, Any] = {
            "method":       _METHOD_DISPLAY.get(method, method),
            "target_class": target_class,
            "tabular":      None,
            "text":         None,
            "image":        None,
            "attention":    None,
        }

        if tabular_tensor is None:
            logger.warning("generate_explanations: no tabular data – skipping tabular attribution")
            return explanations

        try:
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

            baseline = torch.zeros_like(tabular_ig)
            tgt: Optional[int] = (
                target_class
                if self.problem_type.startswith("classification")
                   or self.problem_type == "multilabel_classification"
                else None
            )

            # Clamp target_class to valid range for the head's output layer
            if tgt is not None:
                try:
                    head_out = self._head.output.out_features  # type: ignore[union-attr]
                    if tgt < 0 or tgt >= head_out:
                        logger.warning(
                            "target_class %d out of range [0, %d); clamping to 0",
                            tgt, head_out,
                        )
                        tgt = 0
                except Exception:
                    if tgt < 0:
                        tgt = 0

            try:
                attr_method = self._create_tabular_attribution(method, _forward_tabular)

                if method == "occlusion":
                    attrs = attr_method.attribute(
                        tabular_ig,
                        sliding_window_shapes=(1,),
                        target=tgt,
                    )
                elif method == "smoothgrad":
                    attrs = attr_method.attribute(
                        tabular_ig,
                        baselines=baseline,
                        target=tgt,
                        n_steps=n_steps,
                        nt_samples=5,
                        nt_type="smoothgrad",
                    )
                elif method in ("ig", "gradient_shap"):
                    result = attr_method.attribute(
                        tabular_ig,
                        baselines=baseline,
                        target=tgt,
                        n_steps=n_steps,
                        return_convergence_delta=True,
                    )
                    if isinstance(result, tuple):
                        attrs, convergence_delta = result
                        explanations["convergence_delta"] = float(
                            convergence_delta.mean().item()
                        )
                    else:
                        attrs = result
                elif method == "saliency":
                    attrs = attr_method.attribute(tabular_ig, target=tgt)
                elif method == "feature_ablation":
                    attrs = attr_method.attribute(tabular_ig, target=tgt)
                else:
                    # Fallback to IG for unknown methods
                    from captum.attr import IntegratedGradients
                    attrs = IntegratedGradients(_forward_tabular).attribute(
                        tabular_ig, baselines=baseline, target=tgt,
                        n_steps=n_steps,
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
                logger.warning("Tabular %s attribution failed: %s", method, exc, exc_info=True)

            # Text token attributions when tokenizer is loaded
            if self.tokenizer is not None and "text_pooled" in self.input_dims:
                text_vals: List[str] = self._extract_text_values(inputs)
                if text_vals:
                    explanations["text"] = self._token_attributions(
                        text=text_vals[0],
                        target_class=target_class,
                        n_steps=n_steps,
                        frozen_tabular=batch.get("tabular"),
                    )

            # Image saliency attribution
            if self._image_encoder is not None and "image_pooled" in self.input_dims:
                explanations["image"] = self._image_attributions(
                    inputs, target_class=target_class, n_steps=n_steps,
                )

            # Attention weight visualization (AttentionFusion only)
            explanations["attention"] = self._attention_weights(inputs)

            # Build diagnostics for any NULL XAI outputs
            diag: Dict[str, str] = {}
            if explanations["tabular"] is None:
                diag["tabular"] = "Attribution computation failed (see server logs)"
            if explanations["text"] is None:
                if self.tokenizer is None:
                    diag["text"] = "No text tokenizer loaded for this model"
                elif "text_pooled" not in self.input_dims:
                    diag["text"] = "Text modality was not detected during training"
                else:
                    diag["text"] = "Text attribution computation returned no results"
            if explanations["image"] is None:
                if self._image_encoder is None:
                    diag["image"] = "No image encoder loaded for this model"
                elif "image_pooled" not in self.input_dims:
                    diag["image"] = "Image modality was not detected during training"
                else:
                    diag["image"] = "Image attribution computation returned no results"
            if explanations["attention"] is None:
                diag["attention"] = "Model uses ConcatenationFusion (no attention weights)"
            if diag:
                explanations["diagnostics"] = diag

            return explanations
        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                from modelss.encoders.tabular import GRNTabularEncoder
                encoder = GRNTabularEncoder(input_dim=input_dim)
            else:
                from modelss.encoders.tabular import TabularEncoder
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
            from modelss.encoders.text import TextEncoder

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
        """Load frozen image encoder from saved state dict.

        Reads the actual encoder name from training metadata so the correct
        architecture (ResNet-50, EfficientNet-B0, ViT-B-16, etc.) is loaded
        instead of always defaulting to ResNet-50.
        """
        if "image" not in self.modalities:
            return None

        state_path = self.artifacts_dir / "image_encoder_state.pth"
        if not state_path.exists():
            logger.warning(
                "ImageEncoder state dict not found at %s. "
                "Image features will be zero-filled. "
                "Retrain to save encoder weights for proper image inference.",
                state_path,
            )
            return None

        # Determine actual encoder name from training metadata
        enc_name = "ResNet-50"  # default fallback
        phases = self.metadata.get("phases_summary", {})
        training = phases.get("TRAINING", {})
        enc_sel = training.get("encoder_selection", {})
        if enc_sel.get("image_encoder"):
            enc_name = enc_sel["image_encoder"]

        try:
            encoder = self._create_image_encoder_by_name(enc_name)
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict, strict=True)

            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.to(self.device)

            self._image_encoder_name: str = enc_name
            logger.info("ImageEncoder loaded: %s", enc_name)
            return encoder

        except Exception as exc:
            logger.warning("Could not load ImageEncoder '%s': %s", enc_name, exc)
            return None

    def _create_image_encoder_by_name(self, name: str) -> nn.Module:
        """Dispatch table for image encoder architectures."""
        import torchvision.models as tv

        if name == "ResNet-50":
            from modelss.encoders.image import ImageEncoder
            return ImageEncoder(pretrained=True, freeze_backbone=True)

        if name == "MobileNetV3-Small":
            try:
                backbone = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            except (TypeError, AttributeError):
                backbone = tv.mobilenet_v3_small(pretrained=True)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            projection = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
            return _InferenceVisionWrapper(backbone, projection, 512)

        if name == "EfficientNet-B0":
            try:
                backbone = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)
            except (TypeError, AttributeError):
                backbone = tv.efficientnet_b0(pretrained=True)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            projection = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
            return _InferenceVisionWrapper(backbone, projection, 512)

        if name == "ConvNeXt-Tiny":
            try:
                backbone = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            except (TypeError, AttributeError):
                backbone = tv.convnext_tiny(pretrained=True)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            projection = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
            return _InferenceVisionWrapper(backbone, projection, 512)

        if name == "ViT-B-16":
            try:
                backbone = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1)
            except (TypeError, AttributeError):
                backbone = tv.vit_b_16(pretrained=True)
            in_features = backbone.heads[0].in_features
            backbone.heads = nn.Identity()
            projection = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
            return _InferenceVisionWrapper(backbone, projection, 512)

        # Unknown encoder — fall back to ResNet-50
        logger.warning("Unknown image encoder '%s', falling back to ResNet-50", name)
        from modelss.encoders.image import ImageEncoder
        return ImageEncoder(pretrained=True, freeze_backbone=True)

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

        hidden_dim: int  = int(w0.shape[0])
        total_dim: int   = int(w0.shape[1])
        num_outputs: int = int(b_last.shape[0])

        # Derive input_dims from loaded preprocessor + schema
        input_dims: Dict[str, int] = self._build_input_dims(total_dim)

        # Sanity-check computed total vs. state dict total
        computed_total: int = sum(input_dims.values())
        if computed_total != total_dim:
            logger.warning(
                "input_dims total %d != state-dict total_dim %d "
                "– falling back to single tabular bucket.",
                computed_total, total_dim,
            )
            input_dims = {"tabular": total_dim}

        from automl.trainer import _MultimodalHead

        head = _MultimodalHead(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_outputs=num_outputs,
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
        self._last_warnings: List[str] = []

        # ── Tabular ─────────────────────────────────────────────────────
        if "tabular" in self.input_dims:
            expected_cols: Optional[List[str]] = getattr(
                self.tabular_prep, "_feature_names_in", None
            )
            if expected_cols is not None:
                # Gather text/image columns so they are not flagged as "extra"
                _schema_ds = self.schema.get("per_dataset", [{}])
                _det = _schema_ds[0].get("detected_columns", {}) if _schema_ds else {}
                _non_tabular: set = set(
                    _det.get("text", [])
                    + _det.get("image", [])
                    + ["text", "report", "description", "content", "body",
                       "image", "image_path", "img_path", "photo"]
                )

                # Track schema mismatches for caller visibility
                missing_cols = [c for c in expected_cols if c not in df.columns]
                extra_cols = [
                    c for c in df.columns
                    if c not in expected_cols and c not in _non_tabular
                ]
                if missing_cols:
                    self._last_warnings.append(
                        f"Missing columns (zero-filled): {missing_cols}"
                    )
                if extra_cols:
                    self._last_warnings.append(
                        f"Extra columns (ignored): {extra_cols}"
                    )
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

        # ── Text: encode through BERT when available ────────────────────
        if "text_pooled" in self.input_dims:
            text_values: List[str] = self._extract_text_values(inputs)
            if text_values and self._text_encoder is not None:
                # Pad to batch size if fewer text values than rows
                while len(text_values) < N:
                    text_values.append("")
                with torch.no_grad():
                    text_pooled = self._text_encoder(text_values)  # [N, 768]
                batch["text_pooled"] = text_pooled.to(self.device)
            else:
                if not text_values:
                    logger.debug("_build_batch: no text values found in input")
                if self._text_encoder is None:
                    logger.debug("_build_batch: no text encoder loaded")
                batch["text_pooled"] = torch.full(
                    (N, self.TEXT_DIM), 1e-7, dtype=torch.float32
                )

        # ── Image: encode through ResNet when available ─────────────────
        if "image_pooled" in self.input_dims:
            image_tensor: Optional[torch.Tensor] = self._extract_image_tensors(inputs)
            if image_tensor is not None and self._image_encoder is not None:
                with torch.no_grad():
                    image_pooled = self._image_encoder(
                        image_tensor.to(self.device)
                    )  # [N, 512]
                batch["image_pooled"] = image_pooled
            else:
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
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw model output tensors to (predictions, confidences).

        Parameters
        ----------
        logits    : raw head output
        threshold : decision boundary for binary / multilabel (default 0.5)

        Returns
        -------
        predictions  : long tensor for classification, float for regression
        confidences  : max class probability for classification, 1.0 for regression
        """
        if self.problem_type == "classification_binary":
            probs = torch.sigmoid(logits.squeeze(-1))      # (N,)
            preds = (probs >= threshold).long()
            confidences = torch.where(preds.bool(), probs, 1.0 - probs)
        elif self.problem_type == "multilabel_classification":
            probs = torch.sigmoid(logits)                  # (N, C)
            preds = (probs >= threshold).long()             # (N, C) multi-hot
            confidences = probs                            # (N, C) per-class conf
        elif self.problem_type.startswith("classification"):
            probs = torch.softmax(logits, dim=-1)          # (N, C)
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

        When a tabular encoder (GRN/MLP) is present the encoded dimension
        differs from the raw feature count, so use ``encoded_0 … encoded_N``.
        """
        if self._tabular_encoder is not None:
            return [f"encoded_{i}" for i in range(n_features)]
        if self.tabular_prep is not None:
            transformer = getattr(self.tabular_prep, "_transformer", None)
            if transformer is not None:
                try:
                    names = list(transformer.get_feature_names_out())
                    if len(names) == n_features:
                        return names
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
            logger.warning("Real BERT token attribution failed: %s", exc, exc_info=True)
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
            logger.warning("Token attribution failed: %s", exc, exc_info=True)
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
        detected = per_ds[0].get("detected_columns", {}) if per_ds else {}
        text_cols: List[str] = detected.get("text", [])

        if isinstance(inputs, pd.DataFrame):
            df = inputs
        else:
            df = pd.DataFrame(inputs)

        # Try schema-detected text columns first, then common names
        for col in text_cols + ["text", "report", "description", "content", "body"]:
            if col in df.columns:
                return df[col].fillna("").astype(str).tolist()
        return []

    def _extract_image_tensors(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Optional[torch.Tensor]:
        """Load images from paths, preprocess, and stack into a batch tensor."""
        if self._image_preprocessor is None:
            return None

        per_ds = self.schema.get("per_dataset", [{}])
        detected = per_ds[0].get("detected_columns", {}) if per_ds else {}
        image_cols: List[str] = detected.get("image", [])

        if isinstance(inputs, pd.DataFrame):
            df = inputs
        else:
            df = pd.DataFrame(inputs)

        # Find the image column
        col: Optional[str] = None
        for candidate in image_cols + ["image_path", "image", "img_path", "file_path"]:
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            return None

        tensors: List[torch.Tensor] = []
        for img_val in df[col]:
            try:
                pil_img = self._resolve_image(img_val)
                tensor = self._image_preprocessor.preprocess(pil_img)
                tensors.append(tensor)
            except Exception as exc:
                logger.warning("Image load failed for input: %s", exc)
                tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))

        return torch.stack(tensors) if tensors else None

    # ------------------------------------------------------------------ #
    # Helper: resolve image from path, base64, or bytes
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_image(value: Any) -> "PILImage.Image":
        """
        Resolve an image value to a PIL Image.

        Accepts:
          - Raw ``bytes`` — opened directly via ``BytesIO``.
          - A data-URI string (``data:image/...;base64,...``) — base64-decoded.
          - A long string without path separators — attempted base64 decode.
          - Anything else — treated as a filesystem path.
        """
        from PIL import Image as PILImage
        import base64
        import io

        if isinstance(value, bytes):
            return PILImage.open(io.BytesIO(value)).convert("RGB")

        value_str = str(value)

        # Handle data URI: "data:image/png;base64,iVBOR..."
        if value_str.startswith("data:image"):
            _, encoded = value_str.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

        # Heuristic: long string that doesn't look like a path → try base64
        # Base64 alphabet includes '/' so we can't filter on path separators;
        # instead check that the string doesn't start with common path prefixes.
        _looks_like_path = (
            value_str.startswith("/")
            or value_str.startswith("\\")
            or (len(value_str) > 1 and value_str[1] == ":")  # Windows drive
            or value_str.startswith("./")
            or value_str.startswith("../")
        )
        if len(value_str) > 200 and not _looks_like_path:
            try:
                img_bytes = base64.b64decode(value_str, validate=True)
                return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass  # fall through to URL / file path

        # URL: fetch image over HTTP(S)
        if value_str.startswith("http://") or value_str.startswith("https://"):
            import urllib.request
            with urllib.request.urlopen(value_str, timeout=30) as resp:
                img_bytes = resp.read()
            return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

        # Default: filesystem path
        return PILImage.open(value_str).convert("RGB")

    # ------------------------------------------------------------------ #
    # Multi-method attribution factory
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_tabular_attribution(method: str, forward_fn):
        """Create a Captum attribution object for the given method."""
        from captum.attr import (
            IntegratedGradients,
            GradientShap,
            Saliency,
            Occlusion,
            FeatureAblation,
            NoiseTunnel,
        )

        _dispatch = {
            "ig":               lambda: IntegratedGradients(forward_fn),
            "gradient_shap":    lambda: GradientShap(forward_fn),
            "saliency":         lambda: Saliency(forward_fn),
            "occlusion":        lambda: Occlusion(forward_fn),
            "feature_ablation": lambda: FeatureAblation(forward_fn),
            "smoothgrad":       lambda: NoiseTunnel(IntegratedGradients(forward_fn)),
        }

        factory = _dispatch.get(method)
        if factory is None:
            logger.warning("Unknown attribution method '%s', falling back to IG", method)
            return IntegratedGradients(forward_fn)
        return factory()

    # ------------------------------------------------------------------ #
    # Image saliency – GradCAM / IG fallback
    # ------------------------------------------------------------------ #

    def _get_gradcam_target_layer(self) -> Optional[nn.Module]:
        """Return the last convolutional layer for GradCAM, or None for ViT."""
        if self._image_encoder is None:
            return None

        name = getattr(self, "_image_encoder_name", "ResNet-50")
        encoder = self._image_encoder

        try:
            if name == "ResNet-50":
                # ImageEncoder wraps resnet50 — backbone.layer4[-1]
                return encoder.backbone.layer4[-1]
            if name in ("MobileNetV3-Small", "EfficientNet-B0", "ConvNeXt-Tiny"):
                return encoder.backbone.features[-1]
            if name == "ViT-B-16":
                # ViT has no conv layers — GradCAM not applicable
                return None
        except (AttributeError, IndexError) as exc:
            logger.warning("Could not resolve GradCAM layer for '%s': %s", name, exc)
        return None

    def _image_attributions(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
        target_class: int = 0,
        n_steps: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """Compute image saliency via GradCAM (preferred) or IG fallback."""
        if self._image_encoder is None:
            return None

        image_tensor = self._extract_image_tensors(inputs)
        if image_tensor is None:
            return None

        try:
            img_input = image_tensor[:1].to(self.device).float().requires_grad_(True)

            # Build forward function: image → encoder → fusion head → output
            batch_base: Dict[str, torch.Tensor] = self._build_batch(inputs)

            frozen_extras: Dict[str, torch.Tensor] = {}
            if "tabular" in self.input_dims:
                tab = batch_base.get("tabular")
                frozen_extras["tabular"] = (
                    tab[:1].to(self.device).float()
                    if tab is not None
                    else torch.full((1, self.input_dims["tabular"]), 1e-7, device=self.device)
                )
            if "text_pooled" in self.input_dims:
                if self._text_encoder is not None:
                    text_vals = self._extract_text_values(inputs)
                    if text_vals:
                        with torch.no_grad():
                            frozen_extras["text_pooled"] = (
                                self._text_encoder(text_vals[:1]).to(self.device).detach()
                            )
                    else:
                        frozen_extras["text_pooled"] = torch.full(
                            (1, self.TEXT_DIM), 1e-7, device=self.device
                        )
                else:
                    frozen_extras["text_pooled"] = torch.full(
                        (1, self.TEXT_DIM), 1e-7, device=self.device
                    )

            tgt: Optional[int] = (
                target_class
                if self.problem_type.startswith("classification")
                   or self.problem_type == "multilabel_classification"
                else None
            )

            def _forward_image(img: torch.Tensor) -> torch.Tensor:
                pooled = self._image_encoder(img)
                b: Dict[str, torch.Tensor] = {"image_pooled": pooled}
                b.update(frozen_extras)
                out = self._head(b)
                if self.problem_type == "classification_binary":
                    return torch.sigmoid(out.squeeze(-1)).unsqueeze(-1)
                if self.problem_type == "multilabel_classification":
                    return torch.sigmoid(out)
                if self.problem_type.startswith("classification"):
                    return torch.softmax(out, dim=-1)
                return out.squeeze(-1).unsqueeze(-1)

            # Try GradCAM first
            gradcam_layer = self._get_gradcam_target_layer()
            used_method = "GradCAM"

            if gradcam_layer is not None:
                try:
                    from captum.attr import LayerGradCam

                    gc_attr = LayerGradCam(_forward_image, gradcam_layer)
                    attribution = gc_attr.attribute(img_input, target=tgt)

                    # Upsample to input spatial dims
                    saliency = torch.nn.functional.interpolate(
                        attribution.abs(),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)  # (224, 224)

                    saliency_b64 = self._saliency_to_base64(img_input[0], saliency)
                    return {
                        "saliency_base64": saliency_b64,
                        "method": "GradCAM",
                        "note": f"GradCAM saliency via {self._image_encoder_name}",
                    }
                except Exception as exc:
                    logger.warning("GradCAM failed, falling back to IG: %s", exc)
                    used_method = "IntegratedGradients"

            # Fallback: IG on raw pixels
            from captum.attr import IntegratedGradients

            ig = IntegratedGradients(_forward_image)
            attrs = ig.attribute(
                img_input,
                baselines=torch.zeros_like(img_input),
                target=tgt,
                n_steps=min(n_steps, 30),  # fewer steps for large image tensors
            )
            # Sum along channel dim → (224, 224)
            saliency = attrs.detach().abs().squeeze(0).sum(dim=0)
            saliency_b64 = self._saliency_to_base64(img_input[0], saliency)
            return {
                "saliency_base64": saliency_b64,
                "method": used_method,
                "note": f"Pixel-level IG saliency via {self._image_encoder_name}",
            }

        except Exception as exc:
            logger.warning("Image attribution failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _saliency_to_base64(
        original_image: torch.Tensor,
        saliency: torch.Tensor,
    ) -> str:
        """Blend saliency heatmap with original image and return base64 PNG."""
        import base64
        import io

        from PIL import Image as PILImage

        # Normalize saliency to [0, 1]
        sal_np = saliency.detach().cpu().float().numpy()
        sal_min, sal_max = sal_np.min(), sal_np.max()
        if sal_max - sal_min > 1e-8:
            sal_np = (sal_np - sal_min) / (sal_max - sal_min)
        else:
            sal_np = np.zeros_like(sal_np)

        # Apply jet colormap
        try:
            import matplotlib.cm as cm
            heatmap_rgba = cm.jet(sal_np)  # (H, W, 4)
            heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            # Fallback: simple red channel intensity
            heatmap_rgb = np.zeros((*sal_np.shape, 3), dtype=np.uint8)
            heatmap_rgb[:, :, 0] = (sal_np * 255).astype(np.uint8)

        # Denormalize original image for blending
        img_np = original_image.detach().cpu().float().numpy()  # (3, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
        # Reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # Resize heatmap to match image dims if needed
        if heatmap_rgb.shape[:2] != img_np.shape[:2]:
            heatmap_pil = PILImage.fromarray(heatmap_rgb).resize(
                (img_np.shape[1], img_np.shape[0]),
                PILImage.BILINEAR,
            )
            heatmap_rgb = np.array(heatmap_pil)

        # Blend: 60% original + 40% heatmap
        blended = (0.6 * img_np.astype(float) + 0.4 * heatmap_rgb.astype(float))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Encode to base64 PNG
        pil_out = PILImage.fromarray(blended)
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    # Attention weight extraction (no Captum dependency)
    # ------------------------------------------------------------------ #

    def _attention_weights(
        self,
        inputs: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """Extract per-modality attention weights from AttentionFusion if used."""
        try:
            # Find AttentionFusion module inside the head
            from modelss.fusion import AttentionFusion

            fusion_module: Optional[AttentionFusion] = None
            for module in self._head.modules():
                if isinstance(module, AttentionFusion):
                    fusion_module = module
                    break

            if fusion_module is None:
                return None

            # Build batch and get encoder outputs
            batch = self._build_batch(inputs)

            # Collect modality features in the same order as fusion expects
            features: List[torch.Tensor] = []
            modality_names: List[str] = []

            # The head's input_dims ordering determines the feature order
            for key in sorted(self.input_dims.keys()):
                if key == "tabular":
                    tab = batch.get("tabular")
                    if tab is not None:
                        features.append(tab[:1].to(self.device).float())
                        modality_names.append("Tabular")
                    else:
                        features.append(
                            torch.full((1, self.input_dims[key]), 1e-7, device=self.device)
                        )
                        modality_names.append("Tabular")
                elif key == "image_pooled":
                    if self._image_encoder is not None:
                        img_tensor = self._extract_image_tensors(inputs)
                        if img_tensor is not None:
                            with torch.no_grad():
                                pooled = self._image_encoder(img_tensor[:1].to(self.device))
                            features.append(pooled)
                        else:
                            features.append(
                                torch.full((1, self.IMAGE_DIM), 1e-7, device=self.device)
                            )
                    else:
                        features.append(
                            torch.full((1, self.IMAGE_DIM), 1e-7, device=self.device)
                        )
                    modality_names.append("Image")
                elif key == "text_pooled":
                    if self._text_encoder is not None:
                        text_vals = self._extract_text_values(inputs)
                        if text_vals:
                            with torch.no_grad():
                                pooled = self._text_encoder(text_vals[:1]).to(self.device)
                            features.append(pooled)
                        else:
                            features.append(
                                torch.full((1, self.TEXT_DIM), 1e-7, device=self.device)
                            )
                    else:
                        features.append(
                            torch.full((1, self.TEXT_DIM), 1e-7, device=self.device)
                        )
                    modality_names.append("Text")

            if not features:
                return None

            # Run through AttentionFusion internals to extract weights
            with torch.no_grad():
                projected = [
                    proj(feat.detach())
                    for proj, feat in zip(fusion_module.projections, features)
                ]
                stacked = torch.stack(projected, dim=1)  # (1, n_mod, latent_dim)
                scores = fusion_module.attention_scoring(stacked)  # (1, n_mod, 1)
                weights = torch.softmax(scores, dim=1)  # (1, n_mod, 1)

            weight_list = weights.squeeze(0).squeeze(-1).cpu().tolist()

            return {
                "modality_names": modality_names,
                "weights": [round(w, 4) for w in weight_list],
                "note": "Per-modality attention weights from the AttentionFusion layer.",
            }

        except Exception as exc:
            logger.debug("Attention weight extraction failed: %s", exc)
            return None

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


# ---------------------------------------------------------------------------
# Segmentation inference engine
# ---------------------------------------------------------------------------


class SegmentationInferenceEngine:
    """
    Load Phase-7 segmentation model artifacts and run per-pixel inference.

    Artifacts consumed
    ------------------
    models/registry/{model_id}/artifacts/
    ├── model_weights.pth   ← SegmentationModel state dict
    ├── seg_config.json     ← {decoder, num_classes, input_size}
    └── schema.json         ← GlobalSchema (problem_type == "segmentation")

    ``predict_batch`` returns per-pixel class indices and (optionally) the
    softmax probability map.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        registry_root = Path("models") / "registry" / model_id
        self.artifacts_dir = registry_root / "artifacts"

        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Segmentation model artifacts not found at {self.artifacts_dir}."
            )

        self.metadata: Dict[str, Any] = self._load_json(registry_root / "metadata.json")
        self.schema: Dict[str, Any] = self._load_json(self.artifacts_dir / "schema.json")
        self.seg_config: Dict[str, Any] = self._load_json(self.artifacts_dir / "seg_config.json")

        self.problem_type = "segmentation"
        self.num_classes: int = self.seg_config.get("num_classes", 2)
        self.input_size: int = self.seg_config.get("input_size", 256)

        # Build model and load weights
        from modelss.decoders.unet import SegmentationModel

        self._model = SegmentationModel(
            num_classes=self.num_classes,
            pretrained=False,
            freeze_backbone=False,
        )
        weights_path = self.artifacts_dir / "model_weights.pth"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("Segmentation model loaded: %s (%d classes)", model_id, self.num_classes)
        else:
            logger.warning("No model_weights.pth found for segmentation model %s", model_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        self._model.eval()

        # Preprocessor
        from preprocessing.image_preprocessor import SegmentationPreprocessor
        self._preprocessor = SegmentationPreprocessor(
            target_size=(self.input_size, self.input_size)
        )

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_batch(
        self,
        inputs: Any,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run segmentation inference on one or more images.

        Parameters
        ----------
        inputs : list[dict] | pd.DataFrame
            Each element must contain an ``"image_path"`` key pointing to
            an image file, or an ``"image"`` key with a PIL Image.

        Returns
        -------
        dict with keys:
            ``predictions``  – list of 2D numpy arrays (H×W class indices)
            ``confidences``  – list of floats (mean max softmax probability)
            ``prob_maps``    – list of 3D numpy arrays (C×H×W softmax probs)
            ``problem_type`` – ``"segmentation"``
            ``n_samples``    – int
        """
        from PIL import Image as PILImage

        if isinstance(inputs, pd.DataFrame):
            rows = inputs.to_dict(orient="records")
        elif isinstance(inputs, list):
            rows = inputs
        else:
            rows = [inputs]

        img_tensors: List[torch.Tensor] = []
        for row in rows:
            if "image_path" in row:
                img = PILImage.open(row["image_path"]).convert("RGB")
            elif "image" in row:
                img = row["image"]
                if isinstance(img, str):
                    img = PILImage.open(img).convert("RGB")
            else:
                raise ValueError("Segmentation prediction requires 'image_path' or 'image' key.")

            # Create a dummy mask for preprocessing (we only need the image tensor)
            dummy_mask = PILImage.new("L", img.size, 0)
            img_t, _ = self._preprocessor(img, dummy_mask)
            img_tensors.append(img_t)

        batch = torch.stack(img_tensors).to(self.device)

        try:
            with torch.no_grad():
                logits = self._model(batch)  # (N, C, H, W)
                probs = torch.softmax(logits, dim=1)  # (N, C, H, W)
                preds = logits.argmax(dim=1)  # (N, H, W)

            pred_list = [p.cpu().numpy() for p in preds]
            prob_list = [p.cpu().numpy() for p in probs]
            conf_list = [float(p.max(dim=0).values.mean()) for p in probs]

            return {
                "predictions": pred_list,
                "confidences": conf_list,
                "prob_maps": prob_list,
                "problem_type": "segmentation",
                "n_samples": len(pred_list),
            }
        finally:
            del batch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Static helpers
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
