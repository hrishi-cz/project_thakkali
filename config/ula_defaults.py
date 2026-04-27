"""Default hyperparameters for Unified Latent Alignment (ULA) fusion."""

DEFAULT_LATENT_DIM: int = 256
DEFAULT_N_LAYERS: int = 2
DEFAULT_N_HEADS: int = 4
DEFAULT_DROPOUT: float = 0.1
TOKEN_MODE_DEFAULT: bool = False  # opt-in via fusion_config

LATENT_DIM_CHOICES = [128, 256, 512]
N_LAYERS_RANGE = (1, 4)
N_HEADS_CHOICES = [2, 4, 8]
