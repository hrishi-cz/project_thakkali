"""Regression tests for Integrator field-name detection tagging."""

import numpy as np

from data_ingestion.integrator import Integrator


def test_process_multimodal_tags_field_name_metadata() -> None:
    integrator = Integrator(min_predictability=0.0)

    raw_data = {
        "customer_features": np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ]
        )
    }
    labels = np.array([0, 1, 0])

    results = integrator.process_multimodal(
        raw_data_dict=raw_data,
        y=labels,
        task_type="classification",
    )

    assert "customer_features" in results
    metadata = results["customer_features"]
    assert metadata.metadata.get("field_name") == "customer_features"
    assert isinstance(metadata.detection_method, str) and metadata.detection_method


def test_process_single_modality_forced_detection_tag() -> None:
    integrator = Integrator(min_predictability=0.0)

    metadata = integrator.process_single_modality(
        raw_data=np.array([[1.0, 2.0], [2.0, 3.0]]),
        modality="tabular",
        field_name="features",
    )

    assert metadata.detection_method == "forced"
