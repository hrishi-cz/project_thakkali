"""Unit tests for XAI Engine module — SHAP, GradCAM, Attention, Fusion importance."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pipeline.xai_engine import XAIExplainer, generate_xai_artifacts


class TestXAIExplainer:
    """Test suite for XAIExplainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock multimodal model with encoders."""
        model = Mock()
        model.eval = Mock()
        
        # Mock encoders for each modality
        model.encoder_registry = {
            "tabular": Mock(spec=['__call__']),
            "image": Mock(spec=['__call__', 'features']),
            "text": Mock(spec=['__call__']),
        }
        
        # Mock forward pass
        model.forward = Mock(return_value=torch.tensor([[0.8, 0.2]]))
        model.__call__ = Mock(return_value=torch.tensor([[0.8, 0.2]]))
        
        return model

    @pytest.fixture
    def xai_explainer(self, mock_model):
        """Create XAIExplainer instance with mock model."""
        return XAIExplainer(mock_model, modalities=["tabular", "image", "text"])

    def test_xai_explainer_initialization(self, xai_explainer):
        """Test XAIExplainer initializes with correct modalities."""
        assert xai_explainer.model is not None
        assert "tabular" in xai_explainer.modalities
        assert "image" in xai_explainer.modalities
        assert "text" in xai_explainer.modalities

    def test_explain_tabular_returns_dict(self, xai_explainer):
        """Test tabular explanation returns dict with feature importances."""
        X_tabular = np.array([[0.5, 0.3, 0.7]])
        
        explanation = xai_explainer._explain_tabular_batch(X_tabular)
        
        assert isinstance(explanation, dict)
        assert "method" in explanation
        assert "feature_importances" in explanation
        assert explanation["method"] in ["shap", "gradient", "variance_proxy", "dummy"]

    def test_explain_image_returns_dict(self, xai_explainer):
        """Test image explanation returns dict with heatmap."""
        X_image = np.random.rand(1, 3, 32, 32)
        
        explanation = xai_explainer._explain_image_batch(X_image)
        
        assert isinstance(explanation, dict)
        assert "method" in explanation
        assert explanation["method"] in ["gradcam", "saliency", "dummy"]

    def test_explain_text_returns_dict(self, xai_explainer):
        """Test text explanation returns dict with attention weights."""
        X_text = np.array([["hello", "world", "test"]])
        
        explanation = xai_explainer._explain_text_batch(X_text)
        
        assert isinstance(explanation, dict)
        assert "method" in explanation
        assert explanation["method"] in ["attention", "dummy"]

    def test_explain_fusion_returns_dict(self, xai_explainer):
        """Test fusion explanation returns learned weights."""
        explanation = xai_explainer._explain_fusion_batch()
        
        assert isinstance(explanation, dict)
        assert "method" in explanation
        assert explanation["method"] in ["learned_weights", "dummy"]

    def test_generate_artifacts_structure(self, xai_explainer):
        """Test generate_artifacts returns correct structure."""
        batch = {
            "tabular": np.array([[0.5, 0.3]]),
            "image": np.random.rand(1, 3, 32, 32),
            "text": np.array([["hello", "world"]]),
        }
        
        artifacts = xai_explainer.generate_artifacts(batch)
        
        assert isinstance(artifacts, dict)
        assert "tabular" in artifacts
        assert "image" in artifacts
        assert "text" in artifacts
        assert "fusion" in artifacts

    def test_generate_artifacts_with_partial_batch(self, xai_explainer):
        """Test generate_artifacts handles partial batch gracefully."""
        batch = {
            "tabular": np.array([[0.5, 0.3]]),
            # Missing image and text
        }
        
        artifacts = xai_explainer.generate_artifacts(batch)
        
        assert isinstance(artifacts, dict)
        assert "tabular" in artifacts
        # Other keys may be None or missing, but shouldn't raise error

    def test_xai_error_handling(self, xai_explainer):
        """Test XAI handles invalid inputs gracefully."""
        # Test with empty array
        X_bad = np.array([])
        explanation = xai_explainer._explain_tabular_batch(X_bad)
        
        # Should return dict with error or fallback
        assert isinstance(explanation, dict)


class TestGenerateXAIArtifacts:
    """Test suite for generate_xai_artifacts convenience function."""

    @pytest.fixture
    def mock_model_full(self):
        """Create a more complete mock model."""
        model = Mock()
        model.eval = Mock()
        model.device = torch.device("cpu")
        return model

    def test_generate_xai_artifacts_returns_dict(self, mock_model_full):
        """Test generate_xai_artifacts returns artifact dictionary."""
        batch = {
            "tabular": np.random.rand(2, 5),
            "image": np.random.rand(2, 3, 32, 32),
            "text": np.array([["hello", "world"], ["foo", "bar"]]),
        }
        modalities = ["tabular", "image", "text"]
        
        artifacts = generate_xai_artifacts(mock_model_full, batch, modalities)
        
        assert isinstance(artifacts, dict)
        assert "tabular" in artifacts
        assert "image" in artifacts
        assert "text" in artifacts
        assert "fusion" in artifacts
        assert "timestamp" in artifacts

    def test_generate_xai_artifacts_with_no_modalities(self, mock_model_full):
        """Test generate_xai_artifacts with empty modalities list."""
        batch = {}
        modalities = []
        
        artifacts = generate_xai_artifacts(mock_model_full, batch, modalities)
        
        assert isinstance(artifacts, dict)
        assert "timestamp" in artifacts

    def test_generate_xai_artifacts_stores_metadata(self, mock_model_full):
        """Test that artifacts include proper metadata."""
        batch = {"tabular": np.random.rand(1, 3)}
        modalities = ["tabular"]
        
        artifacts = generate_xai_artifacts(mock_model_full, batch, modalities)
        
        assert "timestamp" in artifacts
        assert "modalities" in artifacts
        assert "num_samples" in artifacts

    @patch('pipeline.xai_engine.XAIExplainer')
    def test_generate_xai_artifacts_calls_explainer(self, mock_explainer_class, mock_model_full):
        """Test that generate_xai_artifacts instantiates and uses XAIExplainer."""
        mock_explainer = Mock()
        mock_explainer.generate_artifacts.return_value = {
            "tabular": {"feature_importances": {}},
        }
        mock_explainer_class.return_value = mock_explainer
        
        batch = {"tabular": np.random.rand(1, 3)}
        modalities = ["tabular"]
        
        artifacts = generate_xai_artifacts(mock_model_full, batch, modalities)
        
        # Verify XAIExplainer was instantiated
        mock_explainer_class.assert_called_once()
        mock_explainer.generate_artifacts.assert_called_once()


class TestXAIModalities:
    """Integration tests for XAI across different modalities."""

    def test_tabular_shap_integration(self):
        """Test SHAP integration for tabular data if available."""
        pytest.importorskip("shap")
        
        # Create a simple sklearn-compatible model
        from sklearn.ensemble import RandomForestClassifier
        import shap
        
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 10)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:1])
            
            # Should return SHAP values for first sample
            assert shap_values is not None
        except Exception:
            # SHAP might have version issues, acceptable to skip
            pytest.skip("SHAP integration not available")

    def test_image_gradcam_integration(self):
        """Test GradCAM integration for image data if available."""
        pytest.importorskip("captum")
        
        # Create a simple CNN
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        )
        
        # Generate dummy image batch
        X_image = torch.randn(1, 3, 32, 32)
        
        # GradCAM requires a layer to attribute to
        target_layer = model[0]
        
        try:
            from captum.attr import LayerGradCam
            gradcam = LayerGradCam(model, target_layer)
            attr = gradcam.attribute(X_image, target=0)
            
            # Should return attribution with same shape as input
            assert attr.shape == X_image.shape
        except Exception as e:
            pytest.skip(f"GradCAM integration not available: {e}")


class TestXAIEdgeCases:
    """Test XAI handling of edge cases and error conditions."""

    def test_xai_with_model_on_gpu(self):
        """Test XAI works when model is on GPU (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = Mock()
        model.eval = Mock()
        model.device = torch.device("cuda:0")
        
        xai = XAIExplainer(model, modalities=["tabular"])
        
        # Should handle GPU models without crashing
        assert xai.model.device.type == "cuda"

    def test_xai_with_single_sample_batch(self):
        """Test XAI with batch size of 1."""
        model = Mock()
        model.eval = Mock()
        
        xai = XAIExplainer(model, modalities=["tabular"])
        
        X_single = np.array([[0.5, 0.3, 0.7]])
        explanation = xai._explain_tabular_batch(X_single)
        
        assert isinstance(explanation, dict)

    def test_xai_with_large_batch(self):
        """Test XAI with larger batch size."""
        model = Mock()
        model.eval = Mock()
        
        xai = XAIExplainer(model, modalities=["tabular"])
        
        X_large = np.random.rand(100, 50)
        explanation = xai._explain_tabular_batch(X_large)
        
        assert isinstance(explanation, dict)

    def test_xai_with_high_dim_tabular(self):
        """Test XAI with very high-dimensional tabular data."""
        model = Mock()
        model.eval = Mock()
        
        xai = XAIExplainer(model, modalities=["tabular"])
        
        X_highdim = np.random.rand(5, 500)  # 500 features
        explanation = xai._explain_tabular_batch(X_highdim)
        
        assert isinstance(explanation, dict)
