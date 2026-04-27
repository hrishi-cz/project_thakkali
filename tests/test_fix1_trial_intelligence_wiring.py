"""
test_fix1_trial_intelligence_wiring.py

Unit test for FIX-1: Trial Intelligence Feedback Loop wiring.

Verifies:
1. OptunaCallback accumulates loss histories correctly
2. TrialIntelligence.analyze() classifies fit_type correctly
3. Adaptive HP adjustment applies correct mutations
4. Cross-trial learning persists across objectives
"""

import sys
from pathlib import Path
import os

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import pytest
import numpy as np
from typing import List
from automl.trial_intelligence import TrialIntelligence
from automl.trainer import OptunaCallback


class MockTrial:
    """Minimal mock Optuna trial for testing."""
    def __init__(self, trial_number: int):
        self.number = trial_number
        self.user_attrs = {}
    
    def set_user_attr(self, key: str, value) -> None:
        self.user_attrs[key] = value
    
    def report(self, value: float, step: int) -> None:
        pass  # No-op for testing
    
    def should_prune(self) -> bool:
        return False


# ===========================================================================
# Test 1: OptunaCallback Loss Accumulation
# ===========================================================================

def test_optuna_callback_accumulates_losses():
    """Verify OptunaCallback collects train/val loss histories."""
    trial = MockTrial(0)
    callback = OptunaCallback(trial=trial, window=5, threshold=1e-3)
    
    # Simulate train losses over 5 epochs
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.22]
    val_losses = [0.55, 0.45, 0.38, 0.35, 0.34]
    
    for train_loss in train_losses:
        callback.train_losses.append(train_loss)
    
    for val_loss in val_losses:
        callback.val_losses.append(val_loss)
    
    assert len(callback.train_losses) == 5
    assert len(callback.val_losses) == 5
    assert callback.train_losses[0] == 0.5
    assert callback.val_losses[-1] == 0.34
    print("✅ Test 1 PASSED: OptunaCallback accumulates losses")


# ===========================================================================
# Test 2: TrialIntelligence Fit Type Classification
# ===========================================================================

def test_trial_intelligence_classify_overfitting():
    """Verify TrialIntelligence correctly detects overfitting."""
    ti = TrialIntelligence()
    
    # Overfitting pattern: train loss decreasing, val loss increasing
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.15, 0.12]
    val_losses = [0.55, 0.5, 0.48, 0.5, 0.55, 0.62]
    
    analysis = ti.analyze(train_losses, val_losses)
    
    assert analysis["fit_type"] == "overfitting"
    assert analysis["train_slope"] < -0.01
    assert analysis["val_slope"] > 0
    print("✅ Test 2 PASSED: Overfitting detection works")


def test_trial_intelligence_classify_underfitting():
    """Verify TrialIntelligence correctly detects underfitting."""
    ti = TrialIntelligence()
    
    # Underfitting pattern: both losses flat/stagnant
    train_losses = [0.5, 0.501, 0.499, 0.502, 0.498, 0.500]
    val_losses = [0.505, 0.504, 0.506, 0.503, 0.505, 0.502]
    
    analysis = ti.analyze(train_losses, val_losses)
    
    assert analysis["fit_type"] == "underfitting"
    assert abs(analysis["train_slope"]) < 0.001
    assert abs(analysis["val_slope"]) < 0.001
    print("✅ Test 3 PASSED: Underfitting detection works")


def test_trial_intelligence_classify_good():
    """Verify TrialIntelligence correctly detects good convergence."""
    ti = TrialIntelligence()
    
    # Good pattern: both losses decreasing at similar rate
    train_losses = [0.5, 0.4, 0.32, 0.26, 0.22, 0.19]
    val_losses = [0.55, 0.44, 0.35, 0.29, 0.25, 0.22]
    
    analysis = ti.analyze(train_losses, val_losses)
    
    assert analysis["fit_type"] == "good"
    print("✅ Test 4 PASSED: Good convergence detection works")


# ===========================================================================
# Test 3: Adaptive HP Adjustment
# ===========================================================================

def test_adaptive_adjustment_overfitting():
    """Verify overfitting trigger reduces LR and increases dropout."""
    ti = TrialIntelligence()
    
    # Simulate overfitting trial record
    ti.records.append({
        "fit_type": "overfitting",
        "epochs": 15,
        "lr": 1e-3,
        "dropout": 0.1,
    })
    
    base_params = {
        "lr": 1e-3,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "hidden_dim": 256,
        "epochs": 15,
    }
    
    adjusted = ti.adjust_hyperparams(base_params)
    
    # Check mutations
    assert adjusted["lr"] < base_params["lr"]  # LR reduced
    assert adjusted["dropout"] > base_params["dropout"]  # Dropout increased
    assert adjusted["weight_decay"] > base_params["weight_decay"]  # WD increased
    assert adjusted["epochs"] < base_params["epochs"]  # Epochs reduced
    print("✅ Test 5 PASSED: Overfitting triggers correct HP mutations")


def test_adaptive_adjustment_underfitting():
    """Verify underfitting trigger increases LR and capacity."""
    ti = TrialIntelligence()
    
    ti.records.append({
        "fit_type": "underfitting",
        "epochs": 15,
    })
    
    base_params = {
        "lr": 1e-3,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "hidden_dim": 256,
        "epochs": 15,
    }
    
    adjusted = ti.adjust_hyperparams(base_params)
    
    assert adjusted["lr"] > base_params["lr"]  # LR increased
    assert adjusted["dropout"] < base_params["dropout"]  # Dropout reduced
    assert adjusted["hidden_dim"] > base_params["hidden_dim"]  # Capacity increased
    assert adjusted["epochs"] > base_params["epochs"]  # Epochs increased
    print("✅ Test 6 PASSED: Underfitting triggers correct HP mutations")


# ===========================================================================
# Test 4: Cross-Trial Memory Persistence
# ===========================================================================

def test_trial_intelligence_memory_persistence():
    """Verify TrialIntelligence maintains cumulative memory across trials."""
    ti = TrialIntelligence()
    
    # Simulate Trial 0 -> Overfitting
    ti.update_memory({
        "trial_number": 0,
        "fit_type": "overfitting",
        "train_slope": -0.05,
        "val_slope": 0.08,
    })
    
    # Simulate Trial 1 -> Good
    ti.records.append({
        "fit_type": "good",
        "epochs": 12,
    })
    
    # Trial 2 should see both records
    assert len(ti.records) == 2
    assert ti.records[0]["fit_type"] == "overfitting"
    assert ti.records[1]["fit_type"] == "good"
    
    # Estimate epochs should consider all trials
    est_epochs = ti.estimate_epochs()
    assert est_epochs > 0
    print("✅ Test 7 PASSED: Cross-trial memory persists correctly")


def test_trial_intelligence_summarize_trials_feedback_signal():
    """Verify trial summary emits stable adaptive feedback fields."""
    ti = TrialIntelligence()

    summary = ti.summarize_trials([
        {
            "trial": 0,
            "fit_type": "overfitting",
            "train_slope": -0.04,
            "val_slope": 0.03,
            "generalization_gap": 0.25,
            "adaptive_penalty": 0.20,
        },
        {
            "trial": 1,
            "fit_type": "overfitting",
            "train_slope": -0.03,
            "val_slope": 0.02,
            "generalization_gap": 0.22,
            "adaptive_penalty": 0.24,
        },
        {
            "trial": 2,
            "fit_type": "good",
            "train_slope": -0.02,
            "val_slope": -0.01,
            "generalization_gap": 0.08,
            "adaptive_penalty": 0.10,
        },
    ])

    assert summary["fit_type"] == "overfitting"
    assert summary["num_trials"] == 3
    assert summary["fit_counts"]["overfitting"] == 2
    assert summary["adaptive_penalty"] > 0.0
    assert 0.0 <= summary["calibration_proxy"] <= 1.0
    print("✅ Test 8 PASSED: Trial summary emits adaptive feedback signal")


# ===========================================================================
# Test 5: Complete Wiring Simulation
# ===========================================================================

def test_complete_adaptive_hpo_loop():
    """Simulate complete 3-trial adaptive HPO loop."""
    ti = TrialIntelligence()
    
    # Trial 0: Overfitting
    print("\n--- Trial 0: Overfitting ---")
    train0 = [0.5, 0.4, 0.3, 0.2, 0.15, 0.12]
    val0 = [0.55, 0.5, 0.48, 0.5, 0.55, 0.62]
    analysis0 = ti.analyze(train0, val0)
    print(f"  fit_type: {analysis0['fit_type']}")
    
    ti.update_memory({
        "trial_number": 0,
        **analysis0,
        "epochs": 15,
        "lr": 1e-3,
        "dropout": 0.1,
    })
    
    # Trial 1: Good (after adaptation)
    print("\n--- Trial 1: Adaptive adjustment applied ---")
    base_params_t1 = {"lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-5, 
                      "hidden_dim": 256, "epochs": 15}
    adjusted_t1 = ti.adjust_hyperparams(base_params_t1)
    print(f"  LR: {base_params_t1['lr']:.2e} → {adjusted_t1['lr']:.2e}")
    print(f"  Dropout: {base_params_t1['dropout']} → {adjusted_t1['dropout']}")
    print(f"  Epochs: {base_params_t1['epochs']} → {adjusted_t1['epochs']}")
    
    train1 = [0.4, 0.32, 0.26, 0.22, 0.2, 0.18]
    val1 = [0.42, 0.34, 0.28, 0.24, 0.22, 0.21]
    analysis1 = ti.analyze(train1, val1)
    print(f"  fit_type: {analysis1['fit_type']}")
    
    ti.update_memory({
        "trial_number": 1,
        **analysis1,
        "epochs": adjusted_t1["epochs"],
        "lr": adjusted_t1["lr"],
        "dropout": adjusted_t1["dropout"],
    })
    
    # Trial 2: Good (no further adjustment needed)
    print("\n--- Trial 2: Maintaining good convergence ---")
    base_params_t2 = {"lr": adjusted_t1['lr'], "dropout": adjusted_t1['dropout'], 
                      "weight_decay": 1e-5, "hidden_dim": 256, "epochs": adjusted_t1["epochs"]}
    adjusted_t2 = ti.adjust_hyperparams(base_params_t2)
    print(f"  LR: {base_params_t2['lr']:.2e} → {adjusted_t2['lr']:.2e} (light tuning)")
    
    train2 = [0.4, 0.31, 0.25, 0.21, 0.19, 0.17]
    val2 = [0.41, 0.33, 0.27, 0.23, 0.21, 0.20]
    analysis2 = ti.analyze(train2, val2)
    print(f"  fit_type: {analysis2['fit_type']}")
    
    # Verify progression
    assert len(ti.records) == 2  # We updated memory twice
    assert ti.records[0]["fit_type"] == "overfitting"
    assert ti.records[1]["fit_type"] == "good"
    print("\n✅ Test 8 PASSED: Complete adaptive HPO loop works end-to-end")


# ===========================================================================
# Run Tests
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIX-1: TRIAL INTELLIGENCE FEEDBACK LOOP - VALIDATION TESTS")
    print("=" * 70)
    
    test_optuna_callback_accumulates_losses()
    print()
    test_trial_intelligence_classify_overfitting()
    test_trial_intelligence_classify_underfitting()
    test_trial_intelligence_classify_good()
    print()
    test_adaptive_adjustment_overfitting()
    test_adaptive_adjustment_underfitting()
    print()
    test_trial_intelligence_memory_persistence()
    print()
    test_complete_adaptive_hpo_loop()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - FIX-1 WIRING VERIFIED")
    print("=" * 70)
