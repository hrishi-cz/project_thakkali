"""Tests for ConformalCalibrator in pipeline/calibration.py."""

import numpy as np
import pytest

from pipeline.calibration import ConformalCalibrator


class TestConformalCalibratorMulticlass:
    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(0)
        self.n_cal = 500
        self.n_test = 200
        self.n_classes = 4
        self.logits = np.random.randn(self.n_cal + self.n_test, self.n_classes)
        self.labels = np.random.randint(0, self.n_classes, self.n_cal + self.n_test)

    def _fit(self, alpha=0.10):
        cal = ConformalCalibrator(alpha=alpha, problem_type="classification_multiclass")
        report = cal.calibrate(self.logits[:self.n_cal], self.labels[:self.n_cal])
        return cal, report

    def test_fit_sets_q_hat(self):
        cal, _ = self._fit()
        assert cal.fitted
        assert cal.q_hat is not None
        assert 0.0 <= cal.q_hat <= 1.0

    def test_fit_report_fields(self):
        _, report = self._fit()
        assert "q_hat" in report
        assert "alpha" in report
        assert "empirical_coverage" in report
        assert "n_calibration" in report

    def test_coverage_approximately_correct(self):
        """Marginal coverage should be ≥ 1-alpha with high probability."""
        cal, _ = self._fit(alpha=0.1)
        sets = cal.predict_set(self.logits[self.n_cal:])
        coverage = np.mean([self.labels[self.n_cal + i] in s for i, s in enumerate(sets)])
        # Allow 10% slack for small N
        assert coverage >= 0.80, f"Coverage {coverage:.2f} too low"

    def test_stricter_alpha_gives_larger_sets(self):
        cal_strict = ConformalCalibrator(alpha=0.01, problem_type="classification_multiclass")
        cal_loose  = ConformalCalibrator(alpha=0.30, problem_type="classification_multiclass")
        cal_strict.calibrate(self.logits[:self.n_cal], self.labels[:self.n_cal])
        cal_loose.calibrate(self.logits[:self.n_cal], self.labels[:self.n_cal])

        sets_strict = cal_strict.predict_set(self.logits[self.n_cal:])
        sets_loose  = cal_loose.predict_set(self.logits[self.n_cal:])

        avg_strict = np.mean([len(s) for s in sets_strict])
        avg_loose  = np.mean([len(s) for s in sets_loose])
        assert avg_strict >= avg_loose, "Stricter alpha should yield larger sets"

    def test_predict_set_returns_list_of_lists(self):
        cal, _ = self._fit()
        test_logits = self.logits[self.n_cal:self.n_cal + 10]
        sets = cal.predict_set(test_logits)
        assert len(sets) == 10
        for s in sets:
            assert isinstance(s, list)
            for label in s:
                assert 0 <= label < self.n_classes

    def test_predict_before_fit_raises(self):
        cal = ConformalCalibrator(alpha=0.1, problem_type="classification_multiclass")
        with pytest.raises(RuntimeError, match="calibrate()"):
            cal.predict_set(np.random.randn(4, 4))


class TestConformalCalibratorBinary:
    def test_binary_coverage(self):
        np.random.seed(1)
        n = 300
        logits = np.random.randn(n)
        labels = (logits > 0).astype(int)  # perfectly separable
        cal = ConformalCalibrator(alpha=0.10, problem_type="classification_binary")
        cal.calibrate(logits[:200], labels[:200])
        sets = cal.predict_set(logits[200:])
        coverage = np.mean([labels[200 + i] in s for i, s in enumerate(sets)])
        assert coverage >= 0.85

    def test_binary_set_contains_valid_labels(self):
        np.random.seed(2)
        logits = np.random.randn(100)
        labels = np.random.randint(0, 2, 100)
        cal = ConformalCalibrator(alpha=0.1, problem_type="classification_binary")
        cal.calibrate(logits[:60], labels[:60])
        sets = cal.predict_set(logits[60:])
        for s in sets:
            assert all(lbl in (0, 1) for lbl in s)


class TestConformalCalibratorRegression:
    def test_regression_interval_coverage(self):
        np.random.seed(3)
        n = 600
        x = np.random.randn(n)
        logits = x + np.random.randn(n) * 0.5  # noisy predictions
        true_y = x
        cal = ConformalCalibrator(alpha=0.10, problem_type="regression")
        cal.calibrate(logits[:400].reshape(-1, 1), true_y[:400])
        intervals = cal.predict_set(logits[400:].reshape(-1, 1))
        coverage = np.mean([
            lo <= true_y[400 + i] <= hi for i, (lo, hi) in enumerate(intervals)
        ])
        assert coverage >= 0.85

    def test_regression_intervals_are_tuples(self):
        np.random.seed(4)
        logits = np.random.randn(50).reshape(-1, 1)
        labels = np.random.randn(50)
        cal = ConformalCalibrator(alpha=0.1, problem_type="regression")
        cal.calibrate(logits[:30], labels[:30])
        intervals = cal.predict_set(logits[30:])
        for lo, hi in intervals:
            assert lo <= hi, "Interval lower bound must be ≤ upper bound"
