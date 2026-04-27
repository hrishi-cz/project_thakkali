"""Assert every primary ExecutionContext mutator emits a decision log entry."""

from core.execution_context import ExecutionContext, create_execution_context


def _count_decisions(ctx: ExecutionContext) -> int:
    return len(ctx.execution_log)


def test_update_preprocessing_logs_decision() -> None:
    ctx = create_execution_context(session_id="t1")
    before = _count_decisions(ctx)
    ctx.update_preprocessing({"tabular": {"strategy": "median"}})
    assert _count_decisions(ctx) == before + 1
    assert ctx.execution_log[-1]["stage"] == "preprocessing"


def test_update_model_selection_logs_decision() -> None:
    ctx = create_execution_context(session_id="t2")
    before = _count_decisions(ctx)
    ctx.update_model_selection(
        candidates=[{"name": "mlp"}],
        reason="best candidate",
    )
    assert _count_decisions(ctx) == before + 1
    assert ctx.execution_log[-1]["stage"] == "model_selection"


def test_update_fusion_logs_decision() -> None:
    ctx = create_execution_context(session_id="t3")
    before = _count_decisions(ctx)
    ctx.update_fusion("attention", {"tabular": 0.6, "text": 0.4})
    assert _count_decisions(ctx) == before + 1
    assert ctx.execution_log[-1]["stage"] == "fusion"


def test_update_training_logs_decision() -> None:
    ctx = create_execution_context(session_id="t4")
    before = _count_decisions(ctx)
    ctx.update_training({"val_loss": 0.3})
    assert _count_decisions(ctx) == before + 1
    assert ctx.execution_log[-1]["stage"] == "training"


def test_update_fit_analysis_logs_decision() -> None:
    ctx = create_execution_context(session_id="t5")
    before = _count_decisions(ctx)
    ctx.update_fit_analysis({"fit_type": "ok", "train_slope": -0.1, "val_slope": -0.08})
    assert _count_decisions(ctx) == before + 1
    assert ctx.execution_log[-1]["stage"] == "training_fit_analysis"
