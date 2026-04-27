"""Build deterministic preprocessing plans from detected schema metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PreprocessingPlanner:
    """Generate a preprocessing and encoder runtime plan for downstream phases."""

    @staticmethod
    def _normalize_scores(scores: Optional[Dict[str, float]]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for key, value in dict(scores or {}).items():
            try:
                normalized[str(key)] = float(value)
            except Exception:
                continue
        return normalized

    @staticmethod
    def _collect_modality_score(
        scores: Dict[str, float],
        modality: str,
        default: float,
    ) -> float:
        values = [
            float(v)
            for k, v in scores.items()
            if str(modality).lower() in str(k).lower()
        ]
        if values:
            return float(sum(values) / len(values))
        return float(default)

    @staticmethod
    def _as_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(value or {}) if isinstance(value, dict) else {}

    @staticmethod
    def _mean_numeric(values: Any) -> float:
        if not isinstance(values, dict):
            return 0.0
        nums = [float(v) for v in values.values() if isinstance(v, (int, float))]
        if not nums:
            return 0.0
        return float(sum(nums) / len(nums))

    @staticmethod
    def _resolve_text_tokenizer(avg_text_len: float, text_task_type: str) -> str:
        if text_task_type == "ner_sequence":
            return "bert-base-uncased"
        if avg_text_len > 200 or text_task_type == "seq2seq":
            return "microsoft/deberta-v3-base"
        if avg_text_len < 50:
            return "distilbert-base-uncased"
        return "bert-base-uncased"

    @staticmethod
    def _tokenizer_to_encoder_key(tokenizer_name: str) -> str:
        normalized = str(tokenizer_name).lower()
        if "distilbert" in normalized:
            return "distilbert"
        if "deberta" in normalized:
            return "deberta"
        if "minilm" in normalized:
            return "minilm"
        return "bert"

    @staticmethod
    def _resolve_image_size(
        total_samples: int,
        image_dataset_size: int,
        image_separability: float,
        image_hints: Dict[str, Any],
    ) -> List[int]:
        hinted_resize = image_hints.get("resize")
        if isinstance(hinted_resize, (list, tuple)) and len(hinted_resize) == 2:
            try:
                return [max(16, int(hinted_resize[0])), max(16, int(hinted_resize[1]))]
            except Exception:
                pass

        effective_size = int(image_dataset_size or total_samples or 0)
        if 0 < effective_size < 5_000:
            return [128, 128]
        if effective_size >= 10_000 and 0.0 < image_separability < 0.35:
            return [384, 384]
        return [224, 224]

    def create_plan(
        self,
        schema_info: Dict[str, Any],
        total_samples: int,
        predictability_scores: Optional[Dict[str, float]] = None,
        modality_presence: Optional[Dict[str, bool]] = None,
        drift_adjusted_predictability: Optional[Dict[str, float]] = None,
        drifted_features: Optional[List[str]] = None,
        global_schema: Optional[Dict[str, Any]] = None,
        preprocessing_hints: Optional[Dict[str, Any]] = None,
        feature_intelligence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        schema_info = dict(schema_info or {})
        global_schema = dict(global_schema or {})
        preprocessing_hints = self._as_dict(preprocessing_hints)
        feature_intelligence = self._as_dict(feature_intelligence)

        modalities: List[str] = list(schema_info.get("global_modalities", []))
        if not modalities:
            modalities = list(global_schema.get("global_modalities", []))

        for modality, present in dict(modality_presence or {}).items():
            mod_name = str(modality)
            if present and mod_name not in modalities:
                modalities.append(mod_name)

        n_modalities = len(modalities)
        total_samples = int(max(0, total_samples))
        drifted_features = [str(col) for col in (drifted_features or [])]

        base_scores = self._normalize_scores(predictability_scores)
        drift_scores = self._normalize_scores(drift_adjusted_predictability)
        effective_scores = dict(base_scores)
        if drift_scores:
            effective_scores.update(drift_scores)

        tab_pred = self._collect_modality_score(effective_scores, "tabular", 0.5)
        text_pred = self._collect_modality_score(effective_scores, "text", 0.5)
        image_pred = self._collect_modality_score(effective_scores, "image", 0.5)

        text_hints = self._as_dict(preprocessing_hints.get("text"))
        image_hints = self._as_dict(preprocessing_hints.get("image"))
        tabular_hints = self._as_dict(preprocessing_hints.get("tabular"))
        multimodal_hints = self._as_dict(preprocessing_hints.get("multimodal"))
        uncertainty_summary = self._as_dict(feature_intelligence.get("uncertainty_summary"))
        mean_uncertainty = self._mean_numeric(uncertainty_summary)
        avg_text_len = 0.0
        try:
            avg_text_len = float(
                feature_intelligence.get("avg_text_len")
                or text_hints.get("avg_text_len")
                or 0.0
            )
        except Exception:
            avg_text_len = 0.0

        text_mode = str(text_hints.get("mode", "free_text") or "free_text").strip().lower()
        text_task_type = str(
            text_hints.get("task_type")
            or feature_intelligence.get("text_task_type")
            or ("text_classification" if "text" in modalities else "")
        ).strip().lower()

        image_dataset_size = int(feature_intelligence.get("image_dataset_size", 0) or 0)
        image_separability = float(feature_intelligence.get("image_label_separability", 0.0) or 0.0)
        image_class_balance = float(feature_intelligence.get("image_class_balance", 0.0) or 0.0)

        schema_confidence = 0.0
        for candidate in (
            schema_info.get("detection_confidence"),
            global_schema.get("detection_confidence"),
            global_schema.get("global_schema_confidence"),
        ):
            try:
                schema_confidence = float(candidate)
                break
            except Exception:
                continue

        # Heuristic policy tuned for predictable latency and stable memory use.
        if avg_text_len > 350:
            text_max_length = 512
        elif avg_text_len > 150:
            text_max_length = 256
        elif avg_text_len > 50:
            text_max_length = 128
        elif "text" in modalities:
            text_max_length = 64
        else:
            text_max_length = 128
        if text_hints.get("max_length") is not None:
            try:
                text_max_length = max(16, int(text_hints["max_length"]))
            except Exception:
                pass
        if text_pred < 0.30:
            text_max_length = min(text_max_length, 96)

        text_tokenizer = self._resolve_text_tokenizer(avg_text_len, text_task_type)
        if text_mode == "structured_label":
            text_pooling = "cls"
        elif text_mode == "free_text" and avg_text_len > 100:
            text_pooling = "mean"
        else:
            text_pooling = "cls"
        if str(text_hints.get("mode", "")).strip().lower() == "structured_label":
            text_pooling = "cls"

        use_embedding_cache = bool(total_samples >= 2_000 and n_modalities >= 2)

        tab_near_unique = 0.75 if tab_pred < 0.30 else 0.5
        tab_max_cardinality = 32 if tab_pred < 0.30 else 50
        tab_scaler = "robust" if (drifted_features or tab_pred < 0.35) else "standard"
        if tabular_hints.get("scaler"):
            tab_scaler = str(tabular_hints.get("scaler"))

        image_target_size = self._resolve_image_size(
            total_samples=total_samples,
            image_dataset_size=image_dataset_size,
            image_separability=image_separability,
            image_hints=image_hints,
        )
        if "augment" in image_hints:
            image_augment = bool(image_hints.get("augment"))
        else:
            effective_image_size = int(image_dataset_size or total_samples or 0)
            image_augment = bool(effective_image_size < 10_000)

        fusion_weights = self._as_dict(multimodal_hints.get("weights"))
        if not fusion_weights:
            fusion_weights = {
                "tabular": 0.5 if "tabular" in modalities else 0.0,
                "text": 0.4 if "text" in modalities else 0.0,
                "image": 0.1 if "image" in modalities else 0.0,
            }

        preferred_text_model = self._tokenizer_to_encoder_key(text_tokenizer)
        if "image" in modalities:
            if image_target_size[0] <= 128:
                preferred_image_model = "mobilenet"
            elif image_target_size[0] >= 384:
                preferred_image_model = "efficientnet"
            else:
                preferred_image_model = "resnet50" if image_dataset_size >= 5_000 else "mobilenet"
        else:
            preferred_image_model = None

        modality_predictability: Dict[str, float] = {}
        for modality in modalities:
            mod = str(modality).lower()
            default_score = 0.5
            if mod == "tabular":
                default_score = tab_pred
            elif mod == "text":
                default_score = text_pred
            elif mod == "image":
                default_score = image_pred
            modality_predictability[mod] = self._collect_modality_score(
                effective_scores,
                mod,
                default_score,
            )

        plan: Dict[str, Any] = {
            "version": "2.1",
            "modalities": modalities,
            "runtime": {
                "use_embedding_cache": use_embedding_cache,
                "high_volume_mode": total_samples >= 100_000,
                "schema_confidence": schema_confidence,
                "drift_feedback_applied": bool(drift_scores or drifted_features),
                "text_task_type": text_task_type or None,
                "mean_uncertainty": round(mean_uncertainty, 4),
                "fusion_weights": fusion_weights,
            },
            "tabular": {
                "imputer": "median",
                "scaler": tab_scaler,
                "imputer_strategy": "most_frequent" if tab_pred < 0.30 else "median",
                "near_unique_ratio": tab_near_unique,
                "max_cardinality": tab_max_cardinality,
                "drifted_features": list(drifted_features),
                "weak_feature_threshold": float(max(0.1, 1.0 - tab_pred)),
                "mean_uncertainty": round(mean_uncertainty, 4),
            },
            "text": {
                "tokenizer": text_tokenizer,
                "max_length": text_max_length,
                "pooling": text_pooling,
                "mode": text_mode,
                "task_type": text_task_type or None,
                "avg_text_len": round(avg_text_len, 2),
            },
            "image": {
                "target_size": image_target_size,
                "normalize": "imagenet",
                "augment_train": image_augment,
                "mode": str(image_hints.get("mode", "supervised" if image_pred >= 0.25 else "self_supervised")),
                "label_separability": round(image_separability, 4),
                "class_balance": round(image_class_balance, 4),
                "dataset_size": image_dataset_size,
            },
            "weak_modalities": [
                str(modality)
                for modality, score in modality_predictability.items()
                if score < 0.25
            ],
            "strong_modalities": [
                str(modality)
                for modality, score in modality_predictability.items()
                if score > 0.75
            ],
            "modality_predictability": modality_predictability,
            "encoder_config": {
                "text": {
                    "pooling": text_pooling,
                    "max_length": text_max_length,
                    "model_name": text_tokenizer,
                    "preferred_model": preferred_text_model,
                    "task_type": text_task_type or None,
                },
                "image": {
                    "use_projection_relu": True,
                    "preferred_model": preferred_image_model,
                    "target_size": image_target_size,
                },
                "tabular": {
                    "input_dropout": round(min(0.30, max(0.0, mean_uncertainty * 0.10)), 4),
                    "preferred_model": "grn" if (feature_intelligence.get("n_features", 0) or 0) > 32 or mean_uncertainty > 0.25 else "mlp",
                },
            },
            "context_signals": {
                "modality_presence": {
                    str(k): bool(v) for k, v in dict(modality_presence or {}).items()
                },
                "predictability_scores": effective_scores,
                "drift_adjusted_predictability": drift_scores,
                "drifted_features": list(drifted_features),
                "preprocessing_hints": preprocessing_hints,
                "feature_intelligence": feature_intelligence,
            },
        }
        return plan
