"""Tests for mood forecasting."""

import torch
import pytest

from depression_companion.models.forecasting import (
    MoodForecaster,
    MoodStabilityPredictor,
)


class TestMoodForecaster:
    """Test mood forecasting model."""
    
    @pytest.fixture
    def model(self) -> MoodForecaster:
        return MoodForecaster(
            input_dim=8,
            hidden_dim=64,
            forecast_horizons=[1, 3, 7],
        )
    
    def test_forward(self, model: MoodForecaster) -> None:
        """Test forward pass."""
        x = torch.randn(4, 14, 8)  # 14 days, 8 features
        output = model(x)
        
        assert "forecasts" in output
        assert "uncertainties" in output
        assert output["forecasts"]["day_1"].shape == (4,)
        assert output["forecasts"]["day_3"].shape == (4,)
        assert output["forecasts"]["day_7"].shape == (4,)
    
    def test_trajectory(self, model: MoodForecaster) -> None:
        """Test trajectory prediction with confidence intervals."""
        x = torch.randn(2, 14, 8)
        trajectory = model.predict_trajectory(x)
        
        assert "trajectory" in trajectory
        assert "lower_bound" in trajectory
        assert "upper_bound" in trajectory
        
        # Lower bound should be less than trajectory
        for horizon in [1, 3, 7]:
            assert (trajectory["lower_bound"][f"day_{horizon}"] <= 
                    trajectory["trajectory"][f"day_{horizon}"]).all()
    
    def test_loss(self, model: MoodForecaster) -> None:
        """Test loss computation."""
        x = torch.randn(4, 14, 8)
        predictions = model(x)
        
        targets = {
            "day_1": torch.rand(4),
            "day_3": torch.rand(4),
            "day_7": torch.rand(4),
        }
        
        losses = model.compute_loss(predictions, targets)
        
        assert "total_loss" in losses
        assert not torch.isnan(losses["total_loss"])


class TestMoodStabilityPredictor:
    """Test mood stability predictor."""
    
    def test_forward(self) -> None:
        model = MoodStabilityPredictor(input_dim=20, num_classes=3)
        x = torch.randn(4, 20)
        output = model(x)
        
        assert "stability_logits" in output
        assert "volatility" in output
        assert "trend" in output
        assert output["trend"].min() >= -1 and output["trend"].max() <= 1