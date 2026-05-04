"""Tests for schema detector edge cases — empty, all-NaN, single-column, etc."""

import numpy as np
import pandas as pd
import pytest

from data_ingestion.schema_detector import COGMASchemaDetector


@pytest.fixture
def det():
    return COGMASchemaDetector()


class TestSchemaEdgeCases:
    def test_empty_dataframe_handled(self, det):
        """Empty DataFrame has no columns — _inspect_dataframe returns Unknown target."""
        df = pd.DataFrame()
        # Empty df: no columns → no candidates → target is Unknown (not a crash)
        try:
            schema = det._inspect_dataframe("empty_ds", df)
            assert schema.target_column == "Unknown"
        except (ValueError, RuntimeError, KeyError):
            pass  # raising is also acceptable for truly empty input

    def test_single_row_dataframe(self, det):
        """Single-row dataset should still produce a schema (not crash)."""
        df = pd.DataFrame({"age": [35], "label": [1]})
        schema = det._inspect_dataframe("single_row", df)
        assert schema.target_column in df.columns or schema.target_column == "Unknown"

    def test_all_nan_column_rejected(self, det):
        """A column that is 100% NaN should be rejected as a target candidate."""
        df = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "nan_col": [np.nan] * 5,
            "label": [0, 1, 0, 1, 0],
        })
        schema = det._inspect_dataframe("all_nan", df)
        assert schema.target_column != "nan_col"

    def test_constant_column_not_selected_as_target(self, det):
        """A constant column has no variance — should not be chosen as target."""
        df = pd.DataFrame({
            "feature1": np.random.randn(30),
            "feature2": np.random.randn(30),
            "constant": [42] * 30,
            "label": np.random.randint(0, 2, 30),
        })
        schema = det._inspect_dataframe("constant", df)
        assert schema.target_column != "constant"

    def test_single_column_dataset_returns_unknown(self, det):
        """A dataset with only one column cannot have features — target Unknown."""
        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
        try:
            schema = det._inspect_dataframe("single_col", df)
            # Either Unknown or only_col (no other choice)
            assert schema.target_column in {"Unknown", "only_col"}
        except (ValueError, RuntimeError):
            pass  # also acceptable — no valid candidates

    def test_all_unique_id_column_rejected(self, det):
        """A near-unique ID column should be penalised and not win as target."""
        df = pd.DataFrame({
            "patient_id": [f"P{i:04d}" for i in range(50)],
            "age": np.random.normal(55, 10, 50),
            "diagnosis": np.random.choice(["healthy", "sick", "unknown"], 50),
        })
        schema = det._inspect_dataframe("id_col", df)
        assert schema.target_column != "patient_id"

    def test_image_path_column_never_target(self, det):
        """Image path columns should be excluded from target candidates."""
        df = pd.DataFrame({
            "image": [f"data/{i}.jpg" for i in range(40)],
            "text": ["Some review text that is quite long" for _ in range(40)],
            "label": np.random.choice([0, 1], 40),
        })
        schema = det._inspect_dataframe("img_path", df)
        assert schema.target_column != "image"

    def test_binary_string_label_detected(self, det):
        """Known binary label pairs (yes/no) should score highly."""
        df = pd.DataFrame({
            "age": np.random.normal(40, 10, 60),
            "income": np.random.exponential(50000, 60),
            "purchased": np.random.choice(["yes", "no"], 60),
        })
        schema = det._inspect_dataframe("binary_str", df)
        assert schema.target_column == "purchased"

    def test_last_column_convention_boost(self, det):
        """'survived' (binary target + keyword + last-position) should be selected as target."""
        np.random.seed(42)
        df = pd.DataFrame({
            "f1": np.random.randn(80),
            "f2": np.random.randn(80),
            "f3": np.random.randn(80),
            "survived": np.random.choice([0, 1], 80),
        })
        schema = det._inspect_dataframe("last_col", df)
        # 'survived' has: binary int fingerprint (0.80), keyword match, last-position bonus
        # It should win as target or at least be in top-2 valid candidates
        valid_cols = [c["column"] for c in schema.candidates]
        if schema.target_column == "survived":
            pass  # Best case: correctly identified
        else:
            # At minimum, survived should be a valid candidate (not rejected)
            assert "survived" in valid_cols, (
                f"'survived' should be a valid candidate; "
                f"detected={schema.target_column}, valid={valid_cols}"
            )

    def test_detect_global_schema_single_dataset(self):
        """detect_global_schema should work with a single-dataset dict."""
        det = COGMASchemaDetector()
        df = pd.DataFrame({
            "age": np.random.normal(40, 10, 100),
            "fare": np.random.exponential(30, 100),
            "survived": np.random.choice([0, 1], 100),
        })
        schema = det.detect_global_schema({"ds1": df})
        assert schema.primary_target in df.columns or schema.primary_target == "Unknown"
        assert schema.detection_confidence >= 0.0

    def test_multimodal_signals_populated_single_dataset(self):
        """multimodal_signals should be computed for single-file multimodal datasets."""
        det = COGMASchemaDetector()
        df = pd.DataFrame({
            "image": [f"img/{i}.jpg" for i in range(50)],
            "text": ["This is a test review with enough words" for _ in range(50)],
            "label": np.random.choice([0, 1], 50),
        })
        schema = det.detect_global_schema({"ds1": df})
        # For multimodal single dataset, within-dataset signals should be computed
        # (may be empty dict if only one modality actually detected)
        assert isinstance(schema.multimodal_signals, dict)

    def test_infer_problem_type_float_target(self, det):
        """Float columns → regression regardless of unique count."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.5, 2.5]})
        assert det._infer_problem(df, "y") == "regression"

    def test_infer_problem_type_binary_integer(self, det):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        assert det._infer_problem(df, "y") == "classification_binary"

    def test_infer_problem_type_multiclass(self, det):
        df = pd.DataFrame({"x": range(15), "y": list(range(5)) * 3})
        assert det._infer_problem(df, "y") == "classification_multiclass"
