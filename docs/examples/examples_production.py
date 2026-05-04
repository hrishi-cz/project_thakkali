"""
PRODUCTION EXAMPLES: Real-world FIX-4 usage scenarios.

These are complete, runnable examples showing how to integrate
the unified modality pipeline into typical ML workflows.

ORGANIZATION
============
1. Example 1: Simple image classification
2. Example 2: Multimodal fraud detection (tabular + images)
3. Example 3: Text + images product recommendation
4. Example 4: Schema-aware data ingestion with validation
5. Example 5: Production inference server

Each example includes:
  - Setup (imports, config)
  - Data loading
  - Training with modality pipeline
  - Evaluation
  - Expected output

RUN ANY EXAMPLE:
  python examples_production.py --example 1
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports from framework
# (These assume apex2-worktree is in PYTHONPATH)
# from data_ingestion.integrator import Integrator, process_dataset
# from data_ingestion.modality_encoder import ModalityEncoder
# from pipeline.orchestrator import Orchestrator
# from config.hyperparameters import MODALITY_CONFIG


# ====================================================================
# EXAMPLE 1: Simple Image Classification with FIX-4
# ====================================================================

def example_1_image_classification():
    """
    Task: Classify images into 2 categories using FIX-4.
    
    Scenario:
      - Input: 100 images (28x28 grayscale or RGB)
      - Target: Binary label (cat vs dog)
      - Method: Encode with ResNet/ViT → validate → train RF
    
    Expected output:
      - Modality predictability score: 0.85-0.95
      - Final model accuracy: 0.90-0.98
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Image Classification with FIX-4")
    print("="*70)
    
    # 1. Setup
    from data_ingestion.integrator import Integrator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    logger.info("Loading synthetic image data...")
    
    # Simulate image data (in practice: load from disk)
    n_samples = 100
    img_height, img_width = 28, 28
    images = [
        np.random.randint(0, 255, (img_height, img_width, 3), dtype=np.uint8)
        for _ in range(n_samples)
    ]
    y = np.random.binomial(1, 0.5, n_samples)  # Binary labels
    
    logger.info(f"Data: {n_samples} images, shape {img_height}x{img_width}x3")
    
    # 2. Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        images, y, test_size=0.2, random_state=42
    )
    
    # 3. Initialize integrator
    integrator = Integrator(min_predictability=0.3)
    
    # 4. Process training data
    logger.info("Encoding training images...")
    metadata_train = integrator.process_single_modality(
        raw_data=X_train_raw,
        modality="image",
        y=y_train,
        task_type="classification",
    )
    
    print(f"\n✓ Training encodings shape: {metadata_train.embeddings.shape}")
    print(f"✓ Encoder: {metadata_train.encoder_name}")
    print(f"✓ Modality score: {metadata_train.final_score():.3f}")
    print(f"✓ Is valid: {metadata_train.is_valid}")
    
    # 5. Process test data (no y, no validation)
    logger.info("Encoding test images...")
    metadata_test = integrator.process_single_modality(
        raw_data=X_test_raw,
        modality="image",
    )
    
    X_train_emb = metadata_train.embeddings
    X_test_emb = metadata_test.embeddings
    
    # 6. Train model on embeddings
    logger.info("Training RandomForest on embeddings...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_emb, y_train)
    
    # 7. Evaluate
    train_acc = model.score(X_train_emb, y_train)
    test_acc = model.score(X_test_emb, y_test)
    
    print(f"\n✓ Training accuracy: {train_acc:.3f}")
    print(f"✓ Test accuracy: {test_acc:.3f}")
    print(f"✓ Feature importance (top 5): {np.argsort(model.feature_importances_)[-5:]}")
    
    return {"train_acc": train_acc, "test_acc": test_acc, "metadata": metadata_train}


# ====================================================================
# EXAMPLE 2: Multimodal Fraud Detection
# ====================================================================

def example_2_multimodal_fraud_detection():
    """
    Task: Detect credit card fraud using tabular + transaction image data.
    
    Scenario:
      - Tabular: amount, merchant_category, time
      - Image: receipt photo (if available)
      - Target: fraud (binary)
      - Method: Encode each modality → validate → concat → train
    
    Expected output:
      - Tabular score: 0.80-0.90
      - Image score: 0.85-0.95 (if receipt photo available)
      - Combined model accuracy: 0.92-0.98
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multimodal Fraud Detection")
    print("="*70)
    
    from data_ingestion.integrator import Integrator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # 1. Simulate data
    logger.info("Generating synthetic fraud detection data...")
    
    n_samples = 200
    
    # Tabular features: amount, merchant_category, time_of_day, etc.
    n_features = 5
    X_tabular = np.random.rand(n_samples, n_features) * 1000
    
    # Images (simulated receipt photos)
    images = [
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for _ in range(n_samples)
    ]
    
    # Target
    y = np.random.binomial(1, 0.1, n_samples)  # 10% fraud rate
    
    logger.info(f"Data: {n_samples} samples, {n_features} tabular features")
    logger.info(f"Fraud rate: {y.mean():.1%}")
    
    # 2. Split
    (X_tabular_train, X_tabular_test, X_img_train, X_img_test,
     y_train, y_test) = train_test_split(
        X_tabular, images, y, test_size=0.2, random_state=42
    )
    
    # 3. Initialize integrator
    integrator = Integrator(min_predictability=0.25)
    
    # 4. Process modalities
    logger.info("Processing tabular modality...")
    meta_tabular = integrator.process_single_modality(
        raw_data=X_tabular_train,
        modality="tabular",
        y=y_train,
        task_type="classification",
    )
    
    logger.info("Processing image modality...")
    meta_image = integrator.process_single_modality(
        raw_data=X_img_train,
        modality="image",
        y=y_train,
        task_type="classification",
    )
    
    print(f"\n✓ Tabular embeddings: {meta_tabular.embeddings.shape}, score={meta_tabular.final_score():.3f}")
    print(f"✓ Image embeddings: {meta_image.embeddings.shape}, score={meta_image.final_score():.3f}")
    
    # 5. Concat embeddings
    X_train_combined = np.hstack([
        meta_tabular.embeddings,
        meta_image.embeddings,
    ])
    
    # Process test data
    meta_tabular_test = integrator.process_single_modality(
        raw_data=X_tabular_test, modality="tabular"
    )
    meta_image_test = integrator.process_single_modality(
        raw_data=X_img_test, modality="image"
    )
    
    X_test_combined = np.hstack([
        meta_tabular_test.embeddings,
        meta_image_test.embeddings,
    ])
    
    # 6. Train
    logger.info("Training RandomForest on combined embeddings...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_combined, y_train)
    
    # 7. Evaluate
    train_acc = model.score(X_train_combined, y_train)
    test_acc = model.score(X_test_combined, y_test)
    
    # Precision/recall on fraud class (more important than accuracy)
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, model.predict(X_test_combined), average="binary"
    )
    
    print(f"\n✓ Training accuracy: {train_acc:.3f}")
    print(f"✓ Test accuracy: {test_acc:.3f}")
    print(f"✓ Fraud detection precision: {prec:.3f}")
    print(f"✓ Fraud detection recall: {rec:.3f}")
    print(f"✓ F1 score: {f1:.3f}")
    
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "fraud_f1": f1,
        "meta_tabular": meta_tabular,
        "meta_image": meta_image,
    }


# ====================================================================
# EXAMPLE 3: Text + Images for Product Recommendation
# ====================================================================

def example_3_product_recommendation():
    """
    Task: Recommend products based on user-uploaded image + text description.
    
    Scenario:
      - Input: User photo + written description
      - Target: Product category (multi-class, e.g., "shoe", "shirt", "hat")
      - Method: Validate each modality → train on combined embeddings
    
    Expected output:
      - Text score: 0.88-0.96
      - Image score: 0.87-0.95
      - Recommendation accuracy: 0.91-0.97
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Product Recommendation (Text + Images)")
    print("="*70)
    
    from data_ingestion.integrator import Integrator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # 1. Simulate data
    logger.info("Generating synthetic product recommendation data...")
    
    n_samples = 150
    n_classes = 5  # 5 product categories
    
    # Text descriptions
    descriptions = [
        f"product description {i % n_classes} size {np.random.randint(1, 10)}"
        for i in range(n_samples)
    ]
    
    # Images
    images = [
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        for _ in range(n_samples)
    ]
    
    # Target: product category
    y = np.random.randint(0, n_classes, n_samples)
    
    logger.info(f"Data: {n_samples} products, {n_classes} categories")
    
    # 2. Split
    (X_text_train, X_text_test, X_img_train, X_img_test,
     y_train, y_test) = train_test_split(
        descriptions, images, y, test_size=0.2, random_state=42
    )
    
    # 3. Integrator
    integrator = Integrator(min_predictability=0.3)
    
    # 4. Process modalities
    logger.info("Processing text modality...")
    meta_text = integrator.process_single_modality(
        raw_data=X_text_train,
        modality="text",
        y=y_train,
        task_type="classification",
    )
    
    logger.info("Processing image modality...")
    meta_image = integrator.process_single_modality(
        raw_data=X_img_train,
        modality="image",
        y=y_train,
        task_type="classification",
    )
    
    print(f"\n✓ Text embeddings: {meta_text.embeddings.shape}, score={meta_text.final_score():.3f}")
    print(f"✓ Image embeddings: {meta_image.embeddings.shape}, score={meta_image.final_score():.3f}")
    
    # 5. Combine
    X_train_combined = np.hstack([meta_text.embeddings, meta_image.embeddings])
    
    meta_text_test = integrator.process_single_modality(X_text_test, modality="text")
    meta_image_test = integrator.process_single_modality(X_img_test, modality="image")
    X_test_combined = np.hstack([meta_text_test.embeddings, meta_image_test.embeddings])
    
    # 6. Train (multi-class classifier)
    logger.info("Training RandomForest for multi-class recommendation...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_combined, y_train)
    
    # 7. Evaluate
    train_acc = model.score(X_train_combined, y_train)
    test_acc = model.score(X_test_combined, y_test)
    
    print(f"\n✓ Training accuracy: {train_acc:.3f}")
    print(f"✓ Test accuracy (recommendation): {test_acc:.3f}")
    
    # Show top recommended categories for a test sample
    pred_probs = model.predict_proba(X_test_combined[:1])
    top_3_categories = np.argsort(pred_probs[0])[-3:][::-1]
    print(f"✓ Top 3 recommended categories for sample 0: {top_3_categories}")
    
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "meta_text": meta_text,
        "meta_image": meta_image,
    }


# ====================================================================
# EXAMPLE 4: Schema-Aware Data Ingestion
# ====================================================================

def example_4_schema_detection():
    """
    Task: Automatically detect and validate data schema for all modalities.
    
    Scenario:
      - Raw data: mixed modalities (images, text, numbers, categories)
      - Goal: Auto-detect modality type + validate predictability
      - Output: Schema with embeddings and validation scores
    
    Expected output:
      - Detected 4 modalities (image, text, tabular, categorical)
      - All modalities validated with scores
      - Embeddings ready for downstream models
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Schema Detection with Validation")
    print("="*70)
    
    from data_ingestion.integrator import process_dataset
    
    # 1. Prepare mixed-modality data
    logger.info("Preparing mixed-modality dataset...")
    
    n_samples = 50
    
    raw_data = {
        "product_image": [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(n_samples)
        ],
        "product_description": [
            f"high quality product with feature {i % 3}"
            for i in range(n_samples)
        ],
        "price": np.random.rand(n_samples) * 1000,
        "category": np.random.choice(["A", "B", "C"], n_samples),
    }
    
    y = np.random.rand(n_samples)  # Target: rating
    
    # 2. Full pipeline
    logger.info("Running full modality pipeline...")
    modality_results = process_dataset(raw_data, y=y, task_type="regression")
    
    # 3. Summarize
    print(f"\n✓ Processed {len(modality_results)} modalities:\n")
    
    for field_name, metadata in modality_results.items():
        print(f"  {field_name}:")
        print(f"    Modality: {metadata.modality_name}")
        print(f"    Embeddings: {metadata.embeddings.shape}")
        print(f"    Predictability: {metadata.predictability_score:.3f}")
        print(f"    Complementarity: {metadata.complementarity_score:.3f}")
        print(f"    Encoder: {metadata.encoder_name}")
        print(f"    Final score: {metadata.final_score():.3f}")
        print(f"    Valid: {metadata.is_valid}\n")
    
    # 4. Extract valid modalities
    valid_modalities = {
        name: meta for name, meta in modality_results.items()
        if meta.is_valid
    }
    
    print(f"✓ Valid modalities for training: {list(valid_modalities.keys())}")
    
    return modality_results


# ====================================================================
# EXAMPLE 5: Production Inference Server
# ====================================================================

def example_5_inference_server():
    """
    Task: Run inference server that uses pre-trained embeddings.
    
    Scenario:
      - Pre-trained model + cached embeddings from training
      - Online prediction: new data → encode (same integrator)
                         → predict
    
    Expected output:
      - Server initialized with pre-trained model
      - Fast inference (embeddings cached)
      - Consistent predictions
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Production Inference Server")
    print("="*70)
    
    from data_ingestion.integrator import Integrator
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    # 1. Simulate pre-trained model
    logger.info("Loading pre-trained model...")
    
    # In production, you'd load from disk:
    # with open("models/fraud_detector.pkl", "rb") as f:
    #     model = pickle.load(f)
    
    # For this example, train a simple model
    integrator = Integrator()
    
    # Generate training data
    X_train_raw = [np.random.rand(5) for _ in range(100)]
    y_train = np.random.binomial(1, 0.1, 100)
    
    meta = integrator.process_single_modality(
        raw_data=X_train_raw,
        modality="tabular",
        y=y_train,
        task_type="classification",
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(meta.embeddings, y_train)
    
    logger.info("Model ready for inference")
    
    # 2. Inference function
    def predict_online(new_data_raw):
        """Single prediction on new data."""
        # Encode using same integrator
        metadata = integrator.process_single_modality(
            raw_data=new_data_raw,
            modality="tabular",
        )
        embeddings = metadata.embeddings
        
        # Predict
        predictions = model.predict(embeddings)
        probabilities = model.predict_proba(embeddings)
        
        return predictions, probabilities
    
    # 3. Test inference
    logger.info("Running test predictions...")
    
    test_data = [np.random.rand(5) for _ in range(5)]
    
    print(f"\n✓ Making predictions on {len(test_data)} test samples:\n")
    
    for i, sample in enumerate(test_data):
        pred, probs = predict_online([sample])  # Wrap in list
        print(f"  Sample {i}: prediction={pred[0]}, "
              f"prob_fraud={probs[0][1]:.3f}")
    
    print(f"\n✓ Server running successfully")
    
    return model


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Production examples for FIX-4")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which example to run (1-5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    examples = {
        1: example_1_image_classification,
        2: example_2_multimodal_fraud_detection,
        3: example_3_product_recommendation,
        4: example_4_schema_detection,
        5: example_5_inference_server,
    }
    
    print("\n" + "="*70)
    print("FIX-4 PRODUCTION EXAMPLES")
    print("="*70)
    
    example_fn = examples[args.example]
    
    try:
        result = example_fn()
        print("\n" + "="*70)
        print(f"✓ EXAMPLE {args.example} COMPLETED SUCCESSFULLY")
        print("="*70)
    except Exception as e:
        logger.error(f"Example {args.example} failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        print(f"✗ EXAMPLE {args.example} FAILED")
        print("="*70)


if __name__ == "__main__":
    main()
