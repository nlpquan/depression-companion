"""Tests for text model."""

import torch
import pytest

from depression_companion.models.text_model import BERTClassifier


class TestBERTClassifier:
    """Test BERT classifier."""
    
    def test_initialization(self) -> None:
        """Test model creation."""
        model = BERTClassifier(
            pretrained_model="bert-base-uncased",
            num_classes=2,
        )
        assert model is not None
    
    def test_forward_pass(self) -> None:
        """Test forward pass with synthetic data."""
        model = BERTClassifier(num_classes=2, hidden_dim=128)
        model.eval()
        
        batch = {
            "input_ids": torch.randint(0, 30000, (4, 128)),
            "attention_mask": torch.ones(4, 128),
        }
        
        with torch.no_grad():
            output = model(batch)
        
        assert "logits" in output
        assert "embeddings" in output
        assert "probs" in output
        assert output["logits"].shape == (4, 2)
    
    def test_forward_pass_without_transformers(self) -> None:
        """Test forward pass when transformers not available."""
        model = BERTClassifier(num_classes=2)
        # Force no transformers
        model._has_transformers = False
        model.eval()
        
        batch = {
            "input_ids": torch.randint(0, 30000, (2, 64)),
        }
        
        with torch.no_grad():
            output = model(batch)
        
        assert output["logits"].shape == (2, 2)
    
    def test_freeze_encoder(self) -> None:
        """Test that freezing encoder works."""
        model = BERTClassifier(
            num_classes=2,
            freeze_encoder=True,
        )
        
        if model._has_transformers:
            frozen_params = sum(
                1 for p in model.bert.parameters() if not p.requires_grad
            )
            total_params = sum(1 for p in model.bert.parameters())
            assert frozen_params == total_params