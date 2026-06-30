"""Tests for early warning system."""

import numpy as np
import torch
import pytest

from depression_companion.models.early_warning import (
    EarlyWarningSystem,
    RelapseRiskClassifier,
)


class TestEarlyWarningSystem:
    """Test early warning detection."""
    
    @pytest.fixture
    def ews(self) -> EarlyWarningSystem:
        return EarlyWarningSystem(baseline_window=14)
    
    def test_compute_baseline(self, ews: EarlyWarningSystem) -> None:
        """Test baseline computation."""
        historical = {
            "sleep_disruption": np.random.randn(30),
            "social_withdrawal": np.random.randn(30),
        }
        
        baselines = ews.compute_baseline(historical)
        
        assert "sleep_disruption" in baselines
        assert "mean" in baselines["sleep_disruption"]
        assert "std" in baselines["sleep_disruption"]
    
    def test_detect_no_warnings(self, ews: EarlyWarningSystem) -> None:
        """Test detection when everything is normal."""
        np.random.seed(42)
        historical = {
            "sleep_disruption": np.random.randn(30) * 0.1 + 0.5,
        }
        
        baselines = ews.compute_baseline(historical)
        # Set current exactly at baseline mean — no deviation, no warning
        current = {"sleep_disruption": baselines["sleep_disruption"]["mean"]}
        
        result = ews.detect_warnings(current, baselines)
        
        assert result["num_active_warnings"] == 0
        assert result["risk_level"] == "low"
    
    def test_detect_warning(self, ews: EarlyWarningSystem) -> None:
        """Test detection of warning signals."""
        historical = {
            "sleep_disruption": np.ones(30) * 0.5,
            "speech_energy_drop": np.ones(30) * 0.6,
        }
        
        baselines = ews.compute_baseline(historical)
        
        # Current values significantly below baseline
        current = {
            "sleep_disruption": 0.1,  # Much lower than baseline
            "speech_energy_drop": 0.6,  # Normal
        }
        
        result = ews.detect_warnings(current, baselines)
        
        # Should detect sleep disruption warning
        assert result["num_active_warnings"] >= 1
        assert result["risk_score"] > 0
    
    def test_risk_levels(self, ews: EarlyWarningSystem) -> None:
        """Test that risk levels are assigned correctly."""
        historical = {
            "sleep_disruption": np.ones(30) * 0.5,
            "social_withdrawal": np.ones(30) * 0.5,
            "speech_energy_drop": np.ones(30) * 0.5,
        }
        
        baselines = ews.compute_baseline(historical)
        
        # All indicators abnormal
        current = {
            "sleep_disruption": 0.0,
            "social_withdrawal": 0.0,
            "speech_energy_drop": 0.0,
        }
        
        result = ews.detect_warnings(current, baselines)
        
        # Should be high risk
        assert result["risk_level"] == "high"
        assert result["risk_score"] > 0.5


class TestRelapseRiskClassifier:
    """Test relapse risk classifier."""
    
    def test_forward(self) -> None:
        model = RelapseRiskClassifier(input_dim=20, num_classes=3)
        x = torch.randn(4, 20)
        output = model(x)
        
        assert "logits" in output
        assert "probs" in output
        assert "risk_score" in output
        assert output["risk_score"].min() >= 0 and output["risk_score"].max() <= 1