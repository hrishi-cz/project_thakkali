"""Unit tests for post-hoc probability calibration."""

from __future__ import annotations

import numpy as np

from pipeline.calibration import ProbabilityCalibrator


def test_binary_isotonic_calibration_fit_and_apply() -> None:
    rng = np.random.default_rng(42)
    logits = rng.normal(loc=0.0, scale=1.2, size=(300, 1))
    probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    targets = (probs > 0.5).astype(int)

    calibrator = ProbabilityCalibrator(min_samples=30)
    report = calibrator.fit(logits, targets, problem_type="classification_binary")

    assert report["enabled"] is True
    calibrated = calibrator.calibrate(logits, probs)
    assert calibrated.shape == probs.shape
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_multiclass_temperature_scaling_fit_and_apply() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(250, 4))
    targets = np.argmax(logits + rng.normal(scale=0.3, size=(250, 4)), axis=1)

    calibrator = ProbabilityCalibrator(min_samples=30)
    report = calibrator.fit(logits, targets, problem_type="classification_multiclass")

    assert report["enabled"] is True
    raw_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    calibrated = calibrator.calibrate(logits, raw_probs)

    assert calibrated.shape == raw_probs.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_multilabel_isotonic_calibration_handles_per_class_fit() -> None:
    rng = np.random.default_rng(99)
    logits = rng.normal(size=(220, 3))
    probs = 1.0 / (1.0 + np.exp(-logits))
    targets = (probs > np.array([0.45, 0.55, 0.50])).astype(int)

    calibrator = ProbabilityCalibrator(min_samples=40)
    report = calibrator.fit(logits, targets, problem_type="multilabel_classification")

    assert report["enabled"] is True
    calibrated = calibrator.calibrate(logits, probs)
    assert calibrated.shape == probs.shape
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
