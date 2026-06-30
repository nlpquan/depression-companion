"""Tests for multi-task learning model."""

import torch
import pytest

from depression_companion.models.multi_task_model import (
    MultiTaskHead,
    MultiTaskDepressionModel,
    UncertaintyWeightedLoss,
)


class TestUncertaintyWeightedLoss:
    """Test uncertainty-weighted loss."""
    
    def test_initialization(self) -> None:
        """Test loss creation."""
        loss_fn = UncertaintyWeightedLoss(num_tasks=3)
        assert len(loss_fn.log_vars) == 3
    
    def test_forward(self) -> None:
        """Test loss computation."""
        loss_fn = UncertaintyWeightedLoss(num_tasks=2)
        
        losses = [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0, requires_grad=True),
        ]
        
        total = loss_fn(losses)
        assert total.item() > 0
    
    def test_learnable_parameters(self) -> None:
        """Test that log variances are learnable."""
        loss_fn = UncertaintyWeightedLoss(num_tasks=2)
        
        # Should have 2 learnable parameters
        params = list(loss_fn.parameters())
        assert len(params) == 1
        assert params[0].shape == (2,)


class TestMultiTaskHead:
    """Test multi-task prediction head."""
    
    @pytest.fixture
    def head(self) -> MultiTaskHead:
        return MultiTaskHead(input_dim=512, hidden_dim=128)
    
    def test_initialization(self, head: MultiTaskHead) -> None:
        """Test head creation."""
        assert head is not None
    
    def test_forward(self, head: MultiTaskHead) -> None:
        """Test forward pass."""
        features = torch.randn(8, 512)
        
        output = head(features)
        
        assert "depression_logits" in output
        assert "depression_probs" in output
        assert "phq8_score" in output
        assert "anxiety_logits" in output
        assert "anxiety_probs" in output
        assert "mood_score" in output
        
        assert output["depression_logits"].shape == (8, 2)
        assert output["phq8_score"].shape == (8,)
        assert output["anxiety_logits"].shape == (8, 2)
        assert output["mood_score"].shape == (8,)
    
    def test_mood_score_range(self, head: MultiTaskHead) -> None:
        """Test that mood score is in [-1, 1]."""
        features = torch.randn(16, 512)
        output = head(features)
        
        assert output["mood_score"].min() >= -1.0
        assert output["mood_score"].max() <= 1.0
    
    def test_probs_sum_to_one(self, head: MultiTaskHead) -> None:
        """Test that classification probabilities sum to 1."""
        features = torch.randn(4, 512)
        output = head(features)
        
        dep_sums = output["depression_probs"].sum(dim=-1)
        anx_sums = output["anxiety_probs"].sum(dim=-1)
        
        assert torch.allclose(dep_sums, torch.ones(4), atol=1e-5)
        assert torch.allclose(anx_sums, torch.ones(4), atol=1e-5)


class TestMultiTaskDepressionModel:
    """Test complete multi-task model."""
    
    @pytest.fixture
    def model(self) -> MultiTaskDepressionModel:
        return MultiTaskDepressionModel(
            audio_dim=256,
            text_dim=512,
            hidden_dim=128,
            use_cross_modal=True,
        )
    
    def test_forward(self, model: MultiTaskDepressionModel) -> None:
        """Test forward pass."""
        audio = torch.randn(4, 256)
        text = torch.randn(4, 512)
        
        output = model(audio, text)
        
        assert "depression_logits" in output
        assert "phq8_score" in output
        assert "anxiety_logits" in output
        assert "mood_score" in output
    
    def test_loss_computation(self, model: MultiTaskDepressionModel) -> None:
        """Test loss computation."""
        audio = torch.randn(4, 256)
        text = torch.randn(4, 512)
        
        predictions = model(audio, text)
        
        targets = {
            "depression_label": torch.randint(0, 2, (4,)),
            "phq8_score": torch.rand(4) * 24,  # PHQ-8 range 0-24
            "anxiety_label": torch.randint(0, 2, (4,)),
            "mood_score": torch.rand(4) * 2 - 1,  # Range [-1, 1]
        }
        
        losses = model.compute_loss(predictions, targets)
        
        assert "total_loss" in losses
        assert "depression_loss" in losses
        assert "phq8_loss" in losses
        assert "anxiety_loss" in losses
        assert "mood_loss" in losses
        assert losses["total_loss"].item() > 0
    
    def test_without_cross_modal(self) -> None:
        """Test model without cross-modal attention."""
        model = MultiTaskDepressionModel(
            audio_dim=256,
            text_dim=512,
            hidden_dim=128,
            use_cross_modal=False,
        )
        
        audio = torch.randn(4, 256)
        text = torch.randn(4, 512)
        
        output = model(audio, text)
        assert "depression_logits" in output