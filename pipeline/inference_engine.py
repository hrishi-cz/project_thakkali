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

        return {
            "predictions":  pred_list,
            "confidences":  confidences.tolist(),
            "problem_type": self.problem_type,
            "n_samples":    len(pred_list),
        }

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

        return explanations

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
        """Load frozen ImageEncoder from saved state dict (no safe fallback)."""
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

        try:
            from modelss.encoders.image import ImageEncoder

            encoder = ImageEncoder(pretrained=True, freeze_backbone=True)
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state_dict, strict=True)

            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.to(self.device)

            logger.info("ImageEncoder loaded from saved state dict")
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw model output tensors to (predictions, confidences).

        Returns
        -------
        predictions  : long tensor for classification, float for regression
        confidences  : max class probability for classification, 1.0 for regression
        """
        if self.problem_type == "classification_binary":
            probs = torch.sigmoid(logits.squeeze(-1))      # (N,)
            preds = (probs >= 0.5).long()
            confidences = torch.where(preds.bool(), probs, 1.0 - probs)
        elif self.problem_type == "multilabel_classification":
            probs = torch.sigmoid(logits)                  # (N, C)
            preds = (probs >= 0.5).long()                  # (N, C) multi-hot
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

        from PIL import Image as PILImage

        tensors: List[torch.Tensor] = []
        for path_val in df[col]:
            try:
                pil_img = PILImage.open(str(path_val)).convert("RGB")
                tensor = self._image_preprocessor.preprocess(pil_img)
                tensors.append(tensor)
            except Exception as exc:
                logger.warning("Image load failed for '%s': %s", path_val, exc)
                tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))

        return torch.stack(tensors) if tensors else None

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
