"""Tests for audio model."""

import torch
import pytest

from depression_companion.models.audio_model import Wav2Vec2Classifier


class TestWav2Vec2Classifier:
    """Test Wav2Vec2 classifier."""
    
    def test_initialization(self) -> None:
        """Test model creation."""
        model = Wav2Vec2Classifier(
            pretrained_model="facebook/wav2vec2-base",
            num_classes=2,
        )
        assert model is not None
    
    def test_forward_pass(self) -> None:
        """Test forward pass with synthetic data."""
        model = Wav2Vec2Classifier(num_classes=2, hidden_dim=128)
        model.eval()
        
        # Create fake batch
        batch_size = 4
        audio_length = 16000  # 1 second at 16kHz
        batch = {
            "audio": torch.randn(batch_size, audio_length),
        }
        
        with torch.no_grad():
            output = model(batch)
        
        assert "logits" in output
        assert "embeddings" in output
        assert "probs" in output
        assert output["logits"].shape == (batch_size, 2)
        assert output["probs"].shape == (batch_size, 2)
    
    def test_freeze_encoder(self) -> None:
        """Test that freezing encoder works."""
        model = Wav2Vec2Classifier(
            num_classes=2,
            freeze_encoder=True,
        )
        
        # If transformers is available, check frozen params
        if model._has_transformers:
            frozen_params = sum(
                1 for p in model.wav2vec2.parameters() if not p.requires_grad
            )
            total_params = sum(1 for p in model.wav2vec2.parameters())
            assert frozen_params == total_params
    
    def test_output_probs_sum_to_one(self) -> None:
        """Test that probabilities sum to 1."""
        model = Wav2Vec2Classifier(num_classes=2)
        model.eval()
        
        batch = {"audio": torch.randn(2, 16000)}
        
        with torch.no_grad():
            output = model(batch)
        
        probs = output["probs"]
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)