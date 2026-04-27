"""Pipeline package for orchestrating ML workflows."""

__all__ = [
	"DatasetManager",
	"RetrainingPipeline",
	"AdaptiveRetrainingPipeline",
	"RetrainingOrchestrator",
	"PipelineOrchestrator",
]


def __getattr__(name: str):
	if name == "DatasetManager":
		from .dataset_manager import DatasetManager

		return DatasetManager
	if name == "RetrainingPipeline":
		from .retrain_executor import RetrainingPipeline

		return RetrainingPipeline
	if name == "AdaptiveRetrainingPipeline":
		from .retraining_pipeline import AdaptiveRetrainingPipeline

		return AdaptiveRetrainingPipeline
	if name == "RetrainingOrchestrator":
		from .retraining_orchestrator import RetrainingOrchestrator

		return RetrainingOrchestrator
	if name == "PipelineOrchestrator":
		from .training_orchestrator import PipelineOrchestrator

		return PipelineOrchestrator
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
