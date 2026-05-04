"""
Shared type definitions for APEX/AutoVision+ pipeline.

This module contains dataclasses and enums used across multiple modules
to maintain consistency and avoid duplication. These types complement
the ExecutionContext which serves as the primary state container.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class Phase(Enum):
    """
    Workflow phases in the APEX/AutoVision+ pipeline.
    
    The 7-phase architecture:
    1. DATA_INGESTION: Multi-source async data loading
    2. SCHEMA_DETECTION: COGMA 6-stage column type inference
    3. PREPROCESSING: Modality-specific transformation
    4. MODEL_SELECTION: GPU-aware architecture search
    5. TRAINING: Optuna HPO + Lightning training
    6. DRIFT_DETECTION: Statistical distribution monitoring
    7. MODEL_REGISTRY: Artifact serialization + versioning
    """
    DATA_INGESTION = 1
    SCHEMA_DETECTION = 2
    PREPROCESSING = 3
    MODEL_SELECTION = 4
    TRAINING = 5
    DRIFT_DETECTION = 6
    MODEL_REGISTRY = 7


@dataclass
class TrainingConfig:
    """
    Configuration for complete training workflow.
    
    This dataclass encapsulates all parameters needed to execute
    the 7-phase training pipeline, from data ingestion through
    model registry.
    
    Attributes
    ----------
    dataset_sources : List[str]
        URLs or local paths to datasets (supports CSV, JSON, Parquet, images)
    problem_type : str
        One of: "regression", "classification_binary", "classification_multiclass"
    modalities : List[str]
        Subset of ["image", "text", "tabular"] present in the data
    target_column : str | None
        Name of the target column (auto-detected if None)
    test_split : float
        Proportion of data for test set (default: 0.2)
    val_split : float
        Proportion of training data for validation (default: 0.2)
    seed : int
        Random seed for reproducibility (default: 42)
    device : str
        Torch device ("cuda" or "cpu", auto-detected by default)
    """
    dataset_sources: List[str]
    problem_type: str
    modalities: List[str]
    target_column: Optional[str] = None
    test_split: float = 0.2
    val_split: float = 0.2
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelSelectionResult:
    """
    Result from Phase 4 model selection.
    
    Captures the architecture decisions made by the AutoML engine,
    including encoder choices, fusion strategy, and hyperparameters.
    
    Attributes
    ----------
    image_encoder : str | None
        Selected vision encoder (e.g., "resnet50", "vit_base_patch16_224")
    text_encoder : str | None
        Selected language model (e.g., "bert-base-uncased", "distilbert")
    tabular_encoder : str | None
        Selected tabular architecture (e.g., "mlp", "tab_transformer")
    fusion_strategy : str
        Multimodal fusion approach (e.g., "concatenation", "cross_attention")
    batch_size : int
        Training batch size
    epochs : int
        Number of training epochs
    learning_rate : float
        Initial learning rate
    dropout : float
        Dropout probability (0.0 - 1.0)
    weight_decay : float
        L2 regularization coefficient
    selection_rationale : str
        Human-readable explanation of why these choices were made
    """
    image_encoder: Optional[str]
    text_encoder: Optional[str]
    tabular_encoder: Optional[str]
    fusion_strategy: str
    batch_size: int
    epochs: int
    learning_rate: float
    dropout: float
    weight_decay: float
    selection_rationale: str


@dataclass
class TrainingMetrics:
    """
    Training metrics from Phase 5.
    
    Captures per-epoch metrics during model training. Supports both
    regression (loss only) and classification (loss + accuracy + F1).
    
    Attributes
    ----------
    epoch : int
        Epoch number (1-indexed)
    train_loss : float
        Training loss for this epoch
    val_loss : float
        Validation loss for this epoch
    train_accuracy : float | None
        Training accuracy (classification only)
    val_accuracy : float | None
        Validation accuracy (classification only)
    train_f1 : float | None
        Training F1 score (classification only)
    val_f1 : float | None
        Validation F1 score (classification only)
    """
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
    val_f1: Optional[float] = None
