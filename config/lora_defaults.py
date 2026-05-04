"""Default hyperparameters for LoRA (Low-Rank Adaptation) fine-tuning."""

DEFAULT_R: int = 8
DEFAULT_ALPHA: float = 16.0
DEFAULT_LR_MULT: float = 0.1

R_CHOICES = [4, 8, 16, 32]
TARGET_MODULES = ("query", "value", "out_proj", "linear1", "q_proj", "v_proj")
