"""Tests for feature attribution."""

import torch
import pytest
import numpy as np

from depression_companion.models.feature_attribution import (
    GradientAttributor,
    AttentionAnalyzer,
)
from depression_companion.models.multi_task_model import MultiTaskDepressionModel


class TestGradientAttributor:
    """Test gradient-based attribution."""
    
    @pytest.fixture
    def model(self) -> MultiTaskDepressionModel:
        return MultiTaskDepressionModel(
            audio_dim=128,
            text_dim=256,
            hidden_dim=64,
            use_cross_modal=False,
        )
    
    @pytest.fixture
    def attributor(self, model: MultiTaskDepressionModel) -> GradientAttributor:
        return GradientAttributor(model)
    
    def test_saliency_map(self, attributor: GradientAttributor) -> None:
        """Test saliency map computation."""
        audio = torch.randn(1, 128)
        text = torch.randn(1, 256)
        
        result = attributor.compute_saliency(audio, text)
        
        assert "audio_saliency" in result
        assert "text_saliency" in result
        assert result["audio_saliency"].shape == (128,)
        assert result["text_saliency"].shape == (256,)
        # Saliency should be non-negative
        assert (result["audio_saliency"] >= 0).all()
    
    def test_integrated_gradients(self, attributor: GradientAttributor) -> None:
        """Test integrated gradients computation."""
        audio = torch.randn(1, 128)
        text = torch.randn(1, 256)
        
        result = attributor.compute_integrated_gradients(
            audio, text, steps=10,
        )
        
        assert "audio_integrated_gradients" in result
        assert "text_integrated_gradients" in result
        assert result["audio_integrated_gradients"].shape == (128,)
    
    def test_specific_target_class(self, attributor: GradientAttributor) -> None:
        """Test attribution for specific target class."""
        audio = torch.randn(1, 128)
        text = torch.randn(1, 256)
        
        result_0 = attributor.compute_saliency(audio, text, target_class=0)
        result_1 = attributor.compute_saliency(audio, text, target_class=1)
        
        # Different classes should have different saliency
        assert not torch.allclose(
            torch.tensor(result_0["audio_saliency"]),
            torch.tensor(result_1["audio_saliency"]),
        )


class TestAttentionAnalyzer:
    """Test attention weight analysis."""
    
    def test_feature_importance(self) -> None:
        """Test feature importance from attention weights."""
        attention = torch.rand(8, 10, 15)  # (B, seq_q, seq_k)
        
        importance = AttentionAnalyzer.compute_feature_importance(attention)
        
        assert len(importance) == 15
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_feature_importance_with_names(self) -> None:
        """Test feature importance with named features."""
        attention = torch.rand(4, 6, 5)
        feature_names = ["pitch", "energy", "mfcc", "jitter", "shimmer"]
        
        importance = AttentionAnalyzer.compute_feature_importance(
            attention, feature_names,
        )
        
        assert list(importance.keys()) == feature_names
    
    def test_most_attended_features(self) -> None:
        """Test finding most attended features."""
        attention = torch.rand(2, 4, 8)
        feature_names = [f"feat_{i}" for i in range(8)]
        
        top_features = AttentionAnalyzer.find_most_attended_features(
            attention, top_k=3, feature_names=feature_names,
        )
        
        assert len(top_features) == 3
        assert top_features[0][1] >= top_features[1][1] >= top_features[2][1]
    
    def test_get_attention_heatmap(self) -> None:
        """Test heatmap conversion."""
        attention = torch.rand(6, 10)
        heatmap = AttentionAnalyzer.get_attention_heatmap(attention)
        
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (6, 10)