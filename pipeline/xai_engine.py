import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone

if TYPE_CHECKING:
    from core.execution_context import ExecutionContext

logger = logging.getLogger(__name__)

class XAIEngine:
    """
    Post-training XAI integration using SHAP.
    Computes global feature importances for tabular data and 
    generates structured explanations.
    """
    def __init__(self, model_id: str, device: str = "cpu"):
        from pipeline.inference_engine import MultimodalInferenceEngine
        self.model_id = model_id
        self.device = torch.device(device)
        self.engine = MultimodalInferenceEngine(model_id)
        
    def explain_tabular(self, df_background: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SHAP values for tabular features using DeepExplainer.
        """
        try:
            import shap
        except ImportError:
            return {"error": "shap is required for XAIEngine. Install: pip install shap"}

        if not self.engine.tabular_prep:
            return {"error": "No tabular preprocessor found for this model."}
            
        try:
            # 1. Prepare background data
            tabular_bg = self.engine.tabular_prep.transform(df_background)
            tabular_bg_tensor = torch.tensor(tabular_bg, dtype=torch.float32).to(self.device).requires_grad_(True)
            
            # 2. Prepare test data
            tabular_test = self.engine.tabular_prep.transform(df_test)
            tabular_test_tensor = torch.tensor(tabular_test, dtype=torch.float32).to(self.device).requires_grad_(True)
            
            # 3. Model wrapper that locks non-tabular modalities
            def model_wrapper(x):
                b = {"tabular": x}
                # Supply zeroed tensors for text/image if the model expects them
                if "text_pooled" in self.engine.input_dims:
                    b["text_pooled"] = torch.zeros(x.shape[0], self.engine.input_dims["text_pooled"]).to(x.device)
                if "image_pooled" in self.engine.input_dims:
                    b["image_pooled"] = torch.zeros(x.shape[0], self.engine.input_dims["image_pooled"]).to(x.device)
                
                out = self.engine._head(b)
                # Ensure correct shape for SHAP explainer
                if self.engine.problem_type == "classification_binary":
                    return torch.sigmoid(out.squeeze(-1)).unsqueeze(-1)
                elif self.engine.problem_type == "multilabel_classification":
                    return torch.sigmoid(out)
                elif self.engine.problem_type.startswith("classification"):
                    return torch.softmax(out, dim=-1)
                return out.squeeze(-1).unsqueeze(-1)
                
            # 4. Compute SHAP values
            explainer = shap.DeepExplainer(model_wrapper, tabular_bg_tensor)
            shap_values = explainer.shap_values(tabular_test_tensor)
            
            # 5. Format Output
            feature_names = self.engine.tabular_prep.get_feature_names_out()
            if isinstance(shap_values, list): # Multi-class
                # Average importance across all classes and test samples
                importances = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
                raw_values = [sv.tolist() for sv in shap_values]
            else:
                importances = np.abs(shap_values).mean(0)
                raw_values = shap_values.tolist()
            
            ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            return {
                "method": "SHAP (DeepExplainer)",
                "feature_ranking": [{"feature": k, "importance": float(v)} for k, v in ranking],
                "shap_values_raw": raw_values,
                "feature_names": feature_names.tolist() if isinstance(feature_names, np.ndarray) else list(feature_names)
            }
            
        except Exception as e:
            logger.error(f"XAIEngine tabular explain failed: {e}", exc_info=True)
            return {"error": f"SHAP explanation failed: {str(e)}"}

    def explain_text(self, df_background: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SHAP attribution for text using DeepExplainer.
        """
        try:
            import shap
            if "text_pooled" not in self.engine.input_dims or not self.engine._text_encoder:
                return {"error": "No text feature found in model."}
                
            text_bg = self.engine._extract_text_values(df_background.to_dict('records'))
            text_test = self.engine._extract_text_values(df_test.to_dict('records'))
            if not text_bg or not text_test:
                return {"error": "Missing text data."}
                
            # For pure text SHAP we would need a HuggingFace explainer or similar
            # that operates on tokens. Since we just have text_pooled, we use DeepExplainer
            # on the pooled embeddings for global feature importance.
            with torch.no_grad():
                bg_emb = self.engine._text_encoder(text_bg).to(self.device).detach()
                bg_emb.requires_grad_(True)
                test_emb = self.engine._text_encoder(text_test).to(self.device).detach()
                test_emb.requires_grad_(True)
                
            def text_wrapper(x):
                b = {"text_pooled": x}
                if "tabular" in self.engine.input_dims:
                    b["tabular"] = torch.zeros(x.shape[0], self.engine.input_dims["tabular"]).to(x.device)
                if "image_pooled" in self.engine.input_dims:
                    b["image_pooled"] = torch.zeros(x.shape[0], self.engine.input_dims["image_pooled"]).to(x.device)
                return self.engine._head(b)
                
            explainer = shap.DeepExplainer(text_wrapper, bg_emb)
            shap_values = explainer.shap_values(test_emb)
            return {
                "method": "SHAP (DeepExplainer - Text Embeddings)",
                "importances": np.abs(shap_values).mean(0).tolist() if not isinstance(shap_values, list) else np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0).tolist()
            }
        except Exception as e:
            return {"error": str(e)}

    def explain_image(self, df_background: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Gradient-based saliency map for image inputs.

        Uses the same gradient-w.r.t-input approach as XAIExplainer._explain_image_batch:
        computes |∂output/∂pixel| for the first test image and returns a
        normalised spatial heatmap aggregated across color channels.
        """
        try:
            if "image_pooled" not in self.engine.input_dims or not self.engine._image_encoder:
                return {"error": "No image encoder found in model."}

            image_col = next(
                (c for c in df_test.columns if c in ("image", "image_path", "img")),
                None,
            )
            if image_col is None:
                return {"error": "No image column found in df_test."}

            # Load first test image
            try:
                from PIL import Image as _PILImage
                from preprocessing.image_preprocessor import ImagePreprocessor as _ImgPrep

                img_path = str(df_test[image_col].iloc[0])
                img = _PILImage.open(img_path).convert("RGB")
                prep = _ImgPrep()
                img_tensor = prep.preprocess(img)  # (3, H, W)
                img_tensor = img_tensor.unsqueeze(0).to(self.device).float()
                img_tensor.requires_grad_(True)
            except Exception as exc:
                return {"error": f"Image load/preprocess failed: {exc}"}

            # Gradient saliency
            try:
                with torch.enable_grad():
                    enc_out = self.engine._image_encoder(img_tensor)
                    if isinstance(enc_out, tuple):
                        enc_out = enc_out[0]
                    score = enc_out.max()
                    score.backward()

                grads = img_tensor.grad.detach()  # (1, 3, H, W)
                saliency = grads[0].abs().mean(dim=0).cpu().numpy()  # (H, W)
                saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

                return {
                    "method": "Gradient Saliency",
                    "gradcam_available": True,
                    "heatmap_shape": list(saliency_norm.shape),
                    "heatmap_min": float(saliency_norm.min()),
                    "heatmap_max": float(saliency_norm.max()),
                    "heatmap_mean": float(saliency_norm.mean()),
                    "info": "Pixel saliency = mean |∂output/∂pixel| across channels, normalised to [0,1].",
                }
            except Exception as exc:
                return {"error": f"Gradient saliency failed: {exc}"}

        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# XAIExplainer: Training-Phase Integration (SHAP + GradCAM + Attention + Fusion)
# ============================================================================

class XAIExplainer:
    """
    Lightweight explainability layer for multimodal models during training.
    Generates artifacts that go directly into model registry metadata.
    
    Usage in training_orchestrator.py Phase 5:
        from pipeline.xai_engine import XAIExplainer, generate_xai_artifacts
        
        explainer = XAIExplainer(model, modalities=["tabular", "image", "text"])
        xai_artifacts = generate_xai_artifacts(model, batch, modalities)
        training_summary["xai"] = xai_artifacts
    """

    def __init__(self, model: torch.nn.Module, modalities: list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            ApexLightningModule or multimodal model with encoders.
        modalities : list
            Active modalities (["tabular", "image", "text"]).
        """
        self.model = model
        self.modalities = modalities
        try:
            self.device = next(model.parameters()).device
        except:
            self.device = torch.device("cpu")

    # -----------------------------------------------------------------------
    # Batch-based artifact generation
    # -----------------------------------------------------------------------

    def generate_artifacts(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate XAI artifacts from a sample batch.
        
        Returns:
            {
                "tabular": {"type": "shap", "values": [...], "feature_importance": [...]},
                "image": {"type": "gradcam", "heatmap": [...]},
                "text": {"type": "attention", "weights": [...]},
                "fusion": {"type": "fusion", "weights": {...}}
            }
        """
        artifacts = {}

        # Tabular SHAP: use first 50 samples
        if "tabular" in batch and "tabular" in self.modalities:
            try:
                tab_data = batch["tabular"]
                if isinstance(tab_data, torch.Tensor):
                    tab_data = tab_data[:50].cpu().numpy()
                artifacts["tabular"] = self._explain_tabular_batch(tab_data)
                logger.info("  ✓ Tabular SHAP generated")
            except Exception as e:
                logger.warning(f"  Tabular XAI failed: {e}")
                artifacts["tabular"] = {"error": str(e)}

        # Image GradCAM: first image only
        if "image" in batch and "image" in self.modalities:
            try:
                img_data = batch["image"][:1]
                artifacts["image"] = self._explain_image_batch(img_data)
                logger.info("  ✓ Image GradCAM generated")
            except Exception as e:
                logger.warning(f"  Image XAI failed: {e}")
                artifacts["image"] = {"error": str(e)}

        # Text Attention: first text only
        if "text" in batch and "text" in self.modalities:
            try:
                text_data = batch["text"][:1]
                artifacts["text"] = self._explain_text_batch(text_data)
                logger.info("  ✓ Text Attention generated")
            except Exception as e:
                logger.warning(f"  Text XAI failed: {e}")
                artifacts["text"] = {"error": str(e)}

        # Fusion Importance
        try:
            artifacts["fusion"] = self._explain_fusion_batch()
            logger.info("  ✓ Fusion importance extracted")
        except Exception as e:
            logger.warning(f"  Fusion XAI failed: {e}")
            artifacts["fusion"] = {"error": str(e)}

        # Modality ablation attribution (systematic "remove each modality" experiment)
        if len(self.modalities) > 1:
            try:
                artifacts["modality_attribution"] = self.modality_ablation_attribution(batch)
                logger.info("  ✓ Modality ablation attribution computed")
            except Exception as e:
                logger.warning(f"  Modality ablation failed: {e}")
                artifacts["modality_attribution"] = {"error": str(e)}

        return artifacts

    def _explain_tabular_batch(self, X: np.ndarray) -> Dict[str, Any]:
        """Return lightweight tabular feature-importance proxy for training artifacts."""
        try:
            # Get tabular encoder
            tabular_enc = getattr(self.model, "tabular_encoder", None)
            if tabular_enc is None:
                return {
                    "type": "feature_importance",
                    "method": "dummy",
                    "feature_importances": [],
                    "error": "No tabular encoder",
                }

            if X.ndim != 2 or X.shape[1] == 0:
                return {
                    "type": "feature_importance",
                    "method": "dummy",
                    "feature_importances": [],
                    "error": "Tabular batch must be 2-D with at least one feature",
                }

            # Use a deterministic proxy (feature std-dev) to avoid returning
            # fake equal SHAP weights during training-time snapshots.
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            proxy = np.std(X_clean, axis=0)
            proxy = np.maximum(proxy, 0.0)
            total = float(np.sum(proxy))
            if total <= 0.0:
                proxy = np.ones(X_clean.shape[1], dtype=np.float64)
                total = float(np.sum(proxy))
            normalized = (proxy / total).tolist()

            return {
                "type": "feature_importance",
                "method": "variance_proxy",
                "proxy_method": "variance_proxy",
                "feature_importance": normalized,
                "feature_importances": normalized,
                "n_features": int(X_clean.shape[1]),
                "info": (
                    "Training-time proxy: feature importance estimated by column variance "
                    "on the final training batch. Run XAIEngine.explain_tabular() for "
                    "real DeepExplainer SHAP attributions."
                ),
            }
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {
                "type": "feature_importance",
                "method": "dummy",
                "feature_importances": [],
                "error": str(e),
            }

    def _explain_image_batch(
        self,
        img_tensor: torch.Tensor,
        registry_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        GradCAM-style pixel-level saliency for images.

        When `registry_dir` and `model_id` are provided the raw heatmap array
        is persisted as ``{registry_dir}/{model_id}/xai_heatmap_latest.npy``
        so downstream endpoints can serve it as a spatial overlay.
        """
        try:
            image_enc = getattr(self.model, "image_encoder", None)
            if image_enc is None:
                return {"type": "gradcam", "method": "dummy", "error": "No image encoder"}

            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32)

            img_tensor = img_tensor.to(self.device)
            img_tensor.requires_grad = True

            # Find last Conv2d for CAM hook
            last_conv: Optional[torch.nn.Module] = None
            activations: list = []
            gradients: list = []

            for module in image_enc.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module

            if last_conv is None:
                return {"type": "gradcam", "method": "dummy", "error": "No Conv2d found"}

            def _save_act(m, i, o):  # noqa: N802
                activations.clear()
                activations.append(o.detach())

            def _save_grad(m, gi, go):  # noqa: N802
                gradients.clear()
                gradients.append(go[0].detach())

            fwd_hook = last_conv.register_forward_hook(_save_act)
            bwd_hook = last_conv.register_backward_hook(_save_grad)

            try:
                with torch.enable_grad():
                    output = image_enc(img_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    score = output[0].max()
                    score.backward()
            finally:
                fwd_hook.remove()
                bwd_hook.remove()

            # GradCAM: weight activations by global-avg-pooled gradients
            if activations and gradients:
                act = activations[0][0]       # (C, H, W)
                grd = gradients[0][0]         # (C, H, W)
                weights = grd.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
                cam = torch.nn.functional.relu((weights * act).sum(dim=0))
                cam_np = cam.cpu().numpy()
            else:
                # Fallback: raw input gradient saliency
                cam_np = img_tensor.grad[0].abs().mean(dim=0).cpu().numpy()

            cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)

            # --- Persist heatmap to model registry ---
            saved_path: Optional[str] = None
            if registry_dir and model_id:
                try:
                    import numpy as _np
                    from pathlib import Path as _Path
                    _hmap_dir = _Path(registry_dir) / model_id
                    _hmap_dir.mkdir(parents=True, exist_ok=True)
                    _hmap_path = _hmap_dir / "xai_heatmap_latest.npy"
                    _np.save(str(_hmap_path), cam_norm.astype("float32"))
                    saved_path = str(_hmap_path)
                except Exception as _e:
                    logger.debug("GradCAM save failed: %s", _e)

            result: Dict[str, Any] = {
                "type": "gradcam",
                "method": "gradcam",
                "heatmap_shape": list(cam_norm.shape),
                "heatmap_min": cam_min,
                "heatmap_max": cam_max,
                "heatmap_mean": float(cam_norm.mean()),
                "info": "GradCAM: ReLU(Σ_c α_c · A_c), normalised to [0,1]",
            }
            if saved_path:
                result["heatmap_path"] = saved_path
            return result

        except Exception as e:
            logger.warning("GradCAM computation failed: %s", e)
            return {"type": "gradcam", "method": "dummy", "error": str(e)}

    def _explain_text_batch(
        self,
        text_tensor: torch.Tensor,
        raw_texts: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Token-level text attribution using Integrated Gradients (captum).

        Falls back to last-layer attention weights when captum is unavailable,
        and to a dummy payload when no text encoder is present.

        Parameters
        ----------
        text_tensor :
            ``input_ids`` tensor of shape ``(N, seq_len)`` or a non-tensor
            placeholder (numpy object array from training batches).
        raw_texts :
            Optional list of raw string texts for token decoding in the output.
        """
        try:
            text_enc = getattr(self.model, "text_encoder", None)
            if text_enc is None:
                return {"type": "attention", "method": "dummy", "error": "No text encoder"}

            if not isinstance(text_tensor, torch.Tensor):
                arr = np.asarray(text_tensor)
                seq_len = int(arr.shape[1]) if arr.ndim > 1 else 1
                return {
                    "type": "attention",
                    "method": "dummy",
                    "seq_len": seq_len,
                    "info": "Text tensor unavailable at this training stage.",
                }

            text_tensor = text_tensor.to(self.device)

            # ── Path A: Integrated Gradients via captum ─────────────────────
            try:
                from captum.attr import LayerIntegratedGradients

                # Identify the embedding layer (works for BERT-family models)
                emb_layer: Optional[torch.nn.Module] = None
                for name in ("embeddings", "word_embeddings", "embed_tokens"):
                    emb_layer = getattr(text_enc, name, None)
                    if emb_layer is not None:
                        break
                if emb_layer is None:
                    for _m in text_enc.modules():
                        if isinstance(_m, torch.nn.Embedding):
                            emb_layer = _m
                            break

                if emb_layer is not None:
                    def _forward_fn(input_ids: torch.Tensor) -> torch.Tensor:
                        with torch.no_grad():
                            out = text_enc(input_ids)
                        if isinstance(out, tuple):
                            out = out[0]
                        # Return scalar score per sample (max activation)
                        return out.max(dim=-1).values

                    lig = LayerIntegratedGradients(_forward_fn, emb_layer)
                    baseline = torch.zeros_like(text_tensor[:1])  # PAD token baseline
                    sample = text_tensor[:1]  # explain first sample

                    attrs, delta = lig.attribute(
                        sample,
                        baselines=baseline,
                        return_convergence_delta=True,
                        n_steps=30,
                    )
                    # attrs: (1, seq_len, embed_dim) → aggregate across embed dim
                    token_scores = attrs[0].norm(dim=-1).detach().cpu().numpy()
                    # Normalise
                    score_sum = float(token_scores.sum()) or 1.0
                    token_scores_norm = (token_scores / score_sum).tolist()

                    # Decode tokens if possible
                    token_ids = sample[0].cpu().tolist()
                    tokenizer = getattr(text_enc, "tokenizer", None) or getattr(
                        getattr(text_enc, "_tokenizer", None), "tokenizer", None
                    )
                    tokens: list = []
                    if tokenizer is not None:
                        try:
                            tokens = tokenizer.convert_ids_to_tokens(token_ids)
                        except Exception:
                            pass
                    tokens = tokens or [str(t) for t in token_ids]

                    return {
                        "type": "integrated_gradients",
                        "method": "LayerIntegratedGradients",
                        "token_scores": token_scores_norm,
                        "tokens": tokens[:len(token_scores_norm)],
                        "convergence_delta": float(delta.mean().abs()) if delta is not None else None,
                        "seq_len": len(token_scores_norm),
                        "top_tokens": sorted(
                            zip(tokens, token_scores_norm),
                            key=lambda x: x[1], reverse=True,
                        )[:10],
                    }
            except ImportError:
                pass  # captum not installed — fall through to attention
            except Exception as _ig_err:
                logger.debug("IG text attribution failed: %s", _ig_err)

            # ── Path B: Last-layer attention weights ────────────────────────
            with torch.no_grad():
                try:
                    outputs = text_enc(text_tensor, output_attentions=True)
                    if hasattr(outputs, "attentions") and outputs.attentions:
                        attn = outputs.attentions[-1]   # (N, heads, seq, seq)
                        # CLS-row attention: how much each token attends to CLS
                        cls_attn = attn[0].mean(dim=0)[0].detach().cpu().numpy()
                        norm_sum = float(cls_attn.sum()) or 1.0
                        cls_attn_norm = (cls_attn / norm_sum).tolist()
                        return {
                            "type": "attention",
                            "method": "last_layer_attention",
                            "token_scores": cls_attn_norm,
                            "seq_len": int(text_tensor.shape[1]),
                            "attention_shape": list(attn[0].shape),
                        }
                except Exception:
                    pass

            return {
                "type": "attention",
                "method": "dummy",
                "info": "Install captum for token-level IG; attention extraction requires HuggingFace model",
                "seq_len": int(text_tensor.shape[1]) if text_tensor.dim() > 1 else 1,
            }
        except Exception as e:
            logger.warning("Text attribution extraction failed: %s", e)
            return {"type": "attention", "method": "dummy", "error": str(e)}

    @torch.no_grad()
    def modality_ablation_attribution(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Systematic modality ablation: zero-out each modality in turn and
        measure the drop in prediction confidence.

        Returns normalised contributions summing to 1.0, e.g.
        ``{"tabular": 0.23, "text": 0.65, "image": 0.12}``.
        """
        # Build a tensor batch (convert ndarrays to tensors)
        tensor_batch: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if k == "target":
                continue
            if isinstance(v, torch.Tensor):
                tensor_batch[k] = v.float()
            else:
                try:
                    tensor_batch[k] = torch.tensor(np.asarray(v), dtype=torch.float32)
                except Exception:
                    pass

        if not tensor_batch:
            return {}

        # Baseline prediction confidence
        try:
            logits = self.model(tensor_batch)
            if logits.dim() > 1 and logits.shape[-1] > 1:
                baseline_conf = float(
                    torch.softmax(logits, dim=-1).max(dim=-1).values.mean()
                )
            else:
                baseline_conf = float(torch.sigmoid(logits).mean())
        except Exception as exc:
            logger.debug("modality_ablation_attribution: baseline forward failed: %s", exc)
            return {}

        contributions: Dict[str, float] = {}
        for mod in self.modalities:
            # Find the tensor key for this modality
            mod_keys = [k for k in tensor_batch if mod in k]
            if not mod_keys:
                contributions[mod] = 0.0
                continue

            # Zero-ablate
            ablated = {
                k: (torch.zeros_like(v) if k in mod_keys else v)
                for k, v in tensor_batch.items()
            }
            try:
                ablated_logits = self.model(ablated)
                if ablated_logits.dim() > 1 and ablated_logits.shape[-1] > 1:
                    ablated_conf = float(
                        torch.softmax(ablated_logits, dim=-1).max(dim=-1).values.mean()
                    )
                else:
                    ablated_conf = float(torch.sigmoid(ablated_logits).mean())
                contributions[mod] = max(0.0, baseline_conf - ablated_conf)
            except Exception:
                contributions[mod] = 0.0

        total = sum(contributions.values()) or 1.0
        return {k: round(v / total, 4) for k, v in contributions.items()}

    def _explain_fusion_batch(self) -> Dict[str, Any]:
        """Extract fusion importance (weights)."""
        try:
            # Try to access fusion module
            fusion = getattr(self.model.model if hasattr(self.model, 'model') else self.model, 
                           "fusion", None)

            def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
                positive = {k: float(max(0.0, v)) for k, v in weights.items()}
                total = float(sum(positive.values()))
                if total <= 0.0:
                    return {m: 1.0 / len(self.modalities) for m in self.modalities}
                return {k: v / total for k, v in positive.items()}
            
            if fusion is None:
                # Default: equal weights for all modalities
                weights = {m: 1.0 / len(self.modalities) for m in self.modalities}
                return {
                    "type": "fusion",
                    "method": "dummy",
                    "weights": weights,
                    "strategy": "equal",
                }

            # Preferred path: consume model-provided attention summary.
            if hasattr(fusion, "get_attention_summary"):
                summary = fusion.get_attention_summary() or {}
                raw = summary.get("modality_importance") if isinstance(summary, dict) else {}
                if isinstance(raw, dict) and raw:
                    mapped: Dict[str, float] = {}
                    for idx, modality in enumerate(self.modalities):
                        mapped[modality] = float(raw.get(modality, raw.get(f"modality_{idx}", 0.0)))
                    return {
                        "type": "fusion",
                        "method": "learned_weights",
                        "weights": _normalize(mapped),
                        "strategy": "attention_summary",
                        "summary": summary,
                    }

            # GraphFusion fallback: use adjacency diagonal as self-importance.
            if hasattr(fusion, "graph"):
                diag_weights: Dict[str, float] = {}
                try:
                    adj = fusion.graph()
                    if torch.is_tensor(adj):
                        diag = torch.diagonal(adj).detach().cpu().tolist()
                    else:
                        diag = list(np.diag(np.asarray(adj)))

                    for idx, modality in enumerate(self.modalities):
                        diag_weights[modality] = float(diag[idx]) if idx < len(diag) else 0.0
                except Exception:
                    diag_weights = {m: 1.0 / len(self.modalities) for m in self.modalities}

                return {
                    "type": "fusion",
                    "method": "learned_weights",
                    "weights": _normalize(diag_weights),
                    "strategy": "graph_attention",
                }

            # Default
            weights = {m: 1.0 / len(self.modalities) for m in self.modalities}
            return {
                "type": "fusion",
                "method": "dummy",
                "weights": weights,
                "strategy": "unknown",
            }
        except Exception as e:
            logger.warning(f"Fusion importance extraction failed: {e}")
            return {
                "type": "fusion",
                "method": "dummy",
                "error": str(e),
            }


# ============================================================================
# Convenience function for training integration
# ============================================================================

def generate_xai_artifacts(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    modalities: list,
    execution_context: Optional["ExecutionContext"] = None,
) -> Dict[str, Any]:
    """
    Generate XAI artifacts from a batch during training.
    
    Call this in training_orchestrator.py Phase 5 after model training:
    
        xai_artifacts = generate_xai_artifacts(model, sample_batch, modalities)
        training_summary["xai"] = xai_artifacts
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained ApexLightningModule.
    batch : Dict[str, Tensor]
        Sample batch with "tabular", "image", "text", "target".
    modalities : list
        Active modalities.
    
    Returns
    -------
    Dict with "tabular", "image", "text", "fusion" subkeys.
    """
    explainer = XAIExplainer(model, modalities)
    artifacts = explainer.generate_artifacts(batch)

    num_samples = 0
    for value in batch.values():
        try:
            num_samples = int(len(value))
            break
        except Exception:
            continue

    artifacts.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    artifacts.setdefault("modalities", list(modalities))
    artifacts.setdefault("num_samples", int(max(0, num_samples)))

    if execution_context is not None:
        try:
            # Populate xai_config so training orchestrator can persist it to metadata
            xai_cfg: Dict[str, Any] = {
                "enabled": True,
                "modalities": list(modalities),
                "methods": {},
                "generated_at": artifacts.get("timestamp"),
            }
            for mod in ("tabular", "text", "image", "fusion"):
                mod_artifact = artifacts.get(mod)
                if isinstance(mod_artifact, dict):
                    xai_cfg["methods"][mod] = mod_artifact.get("method", "unknown")
            if "modality_attribution" in artifacts:
                xai_cfg["modality_attribution"] = artifacts["modality_attribution"]
            execution_context.xai_config = xai_cfg

            if hasattr(execution_context, "log_decision"):
                execution_context.log_decision(
                    "xai",
                    f"XAI artifacts generated for modalities: {modalities}",
                    evidence=f"keys={list(artifacts.keys())}",
                )
        except Exception:
            pass

    return artifacts
