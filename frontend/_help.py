"""Glossary-linked help text helper for Streamlit form fields.

Usage::

    from frontend._help import glossary_help, GLOSSARY
    st.selectbox("Fusion Strategy", options=[...], help=glossary_help("Fusion Strategy"))
"""

from __future__ import annotations

from typing import Dict


# Comprehensive glossary — every term used in the APEX UI
GLOSSARY: Dict[str, str] = {
    # Training hyperparameters
    "Learning Rate": (
        "Controls the speed at which the model adjusts its weights. "
        "Too high → training diverges. Too low → training stalls. "
        "Typical range: 1e-5 to 1e-3."
    ),
    "Dropout": (
        "Randomly deactivates neurons during training to prevent overfitting. "
        "Higher values (0.3–0.5) help on small datasets. "
        "Set to 0 for maximum capacity on large datasets."
    ),
    "Weight Decay": (
        "L2 regularization penalty on large weights. Forces simpler, "
        "more generalizable patterns. Typical: 1e-5 to 1e-3."
    ),
    "Epochs": (
        "Number of full passes through the training data. "
        "More epochs → better fit but risk of overfitting. "
        "APEX uses early stopping to auto-detect the optimal epoch."
    ),
    "Batch Size": (
        "Number of samples processed before updating model weights. "
        "Larger batches → faster but need more GPU memory. "
        "32 is a safe default; 16 for GPU-constrained setups."
    ),

    # Fusion strategies
    "Fusion Strategy": (
        "How APEX combines information from different data types. "
        "Concatenation: stacks embeddings (fast, stable). "
        "Attention: learns which modality matters more. "
        "Graph: models interactions between modalities explicitly."
    ),
    "Concatenation Fusion": "Stacks modality embeddings side-by-side. O(1) overhead, most stable.",
    "Attention Fusion": "Learns per-modality importance weights. Best for 2+ modalities.",
    "Uncertainty Graph Fusion": "Combines graph structure with uncertainty weighting. Best for 3+ modalities.",
    "CrossFuse Complementarity": "Measures and exploits complementary information between modalities.",

    # Model selection
    "Probe Score": (
        "A quick Random Forest is trained on preprocessed features to estimate "
        "which model architecture will perform best. Higher score = better expected fit."
    ),
    "Tier": (
        "Hardware-based model sizing. Lightweight (<6GB GPU), "
        "Medium (6-12GB), Large (>12GB). Prevents out-of-memory errors."
    ),
    "VRAM Budget": (
        "Maximum GPU memory the model is allowed to use. "
        "APEX measures this via dry-runs before training."
    ),

    # Calibration & metrics
    "ECE (Expected Calibration Error)": (
        "Measures how well predicted probabilities match actual outcomes. "
        "ECE < 0.05 = well-calibrated. "
        "A well-calibrated model saying '80% confident' is right ~80% of the time."
    ),
    "Brier Score": (
        "Mean squared error between predicted probabilities and true labels. "
        "Combines calibration quality and discrimination ability. Lower is better."
    ),
    "Confidence": (
        "How certain the model is about its prediction. "
        "High confidence (>90%) = very likely correct. "
        "Low confidence (<50%) = model is unsure, treat prediction with caution."
    ),
    "F1 Score": (
        "Harmonic mean of precision and recall. "
        "Good for imbalanced datasets where accuracy is misleading. "
        "Range: 0 (worst) to 1 (perfect)."
    ),

    # Preprocessing
    "Label Smoothing": (
        "Softens hard labels (e.g. [0,1] → [0.05, 0.95]) to prevent "
        "overconfident predictions. Typical: 0.05–0.1."
    ),
    "Feature Sparsity": (
        "Fraction of zero values in the feature matrix. "
        "High sparsity (>0.5) → sparse models (XGBoost) may outperform neural nets."
    ),
    "Label Entropy": (
        "Shannon entropy of target labels. Low (<0.5) = easy classification. "
        "High (>1.5) = many classes or balanced distribution."
    ),

    # Drift & monitoring
    "Concept Drift": (
        "When the relationship between inputs and outputs changes over time. "
        "APEX detects this via DDM [10] and DriftLens [11] cosine-distance methods."
    ),
    "Data Drift": (
        "When the distribution of input features changes but the true "
        "relationship stays the same. Often precedes concept drift."
    ),

    # XAI
    "Integrated Gradients": (
        "Attribution method that explains predictions by computing "
        "the integral of gradients along a path from a baseline to the input. "
        "Provides per-feature importance scores."
    ),
    "GradCAM": (
        "Gradient-weighted Class Activation Mapping — highlights which regions "
        "of an image most influenced the prediction."
    ),
    "Token Attribution": (
        "Per-token importance scores for text inputs. "
        "Green = helps prediction, Red = suppresses prediction."
    ),

    # Training techniques
    "SWA (Stochastic Weight Averaging)": (
        "Averages model weights across training epochs for better generalization. "
        "Based on Izmailov et al. (UAI 2018) [14]."
    ),
    "PCGrad (Gradient Surgery)": (
        "Resolves conflicting gradients between modalities by projecting "
        "gradients onto non-conflicting directions. Yu et al. (NeurIPS 2020) [15]."
    ),
    "Focal Loss": (
        "Down-weights easy samples, focusing training on hard/misclassified "
        "examples. Crucial for class-imbalanced datasets. Lin et al. (ICCV 2017) [13]."
    ),
    "EWC (Elastic Weight Consolidation)": (
        "Prevents catastrophic forgetting during retraining by penalizing "
        "changes to important weights. Kirkpatrick et al. (PNAS 2017) [8]."
    ),
    "Contrastive Loss (NT-Xent)": (
        "CLIP-style loss that aligns embeddings from different modalities. "
        "Same-entity pairs should be close; different-entity pairs far apart. "
        "Auto-activated when ≥2 modalities are present."
    ),
    "Modality Dropout": (
        "Randomly masks out entire modalities during training (p=0.15) "
        "to improve robustness when modalities are missing at inference time."
    ),

    # Architecture
    "Cross-Layer RGAT Head": (
        "Relational Graph Attention Network head for capturing "
        "cross-modal relational structure. Inspired by RGAT (NeurIPS 2025) [2]."
    ),
    "JIT Encoder Selection": (
        "Just-In-Time encoder selection based on live VRAM measurements. "
        "Novel to APEX — selects the largest encoder that fits in GPU memory."
    ),
}


def glossary_help(term: str) -> str:
    """Return a help string for a glossary term.

    Parameters
    ----------
    term : str
        The glossary key (e.g. ``"Fusion Strategy"``).

    Returns
    -------
    str
        The glossary explanation, or empty string if not found.
    """
    entry = GLOSSARY.get(term, "")
    if entry:
        return f"{entry}\n\n📖 See glossary for more details."
    return ""
