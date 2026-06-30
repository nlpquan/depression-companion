"""Tests for temporal models."""

import torch
import pytest

from depression_companion.models.temporal_model import (
    PositionalEncoding,
    TemporalAttention,
    BiLSTMTemporalModel,
    MultiHeadTemporalAttention,
)


class TestPositionalEncoding:
    """Test positional encoding."""
    
    def test_shape(self) -> None:
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(4, 50, 64)
        output = pe(x)
        assert output.shape == (4, 50, 64)
    
    def test_different_positions(self) -> None:
        pe = PositionalEncoding(d_model=32)
        x = torch.zeros(1, 10, 32)
        output = pe(x)
        # Different positions should have different encodings
        assert not torch.allclose(output[0, 0], output[0, 1])


class TestTemporalAttention:
    """Test temporal attention."""
    
    def test_shape(self) -> None:
        attention = TemporalAttention(hidden_dim=128)
        x = torch.randn(4, 20, 128)
        context, weights = attention(x)
        assert context.shape == (4, 128)
        assert weights.shape == (4, 20)
    
    def test_weights_sum_to_one(self) -> None:
        attention = TemporalAttention(hidden_dim=64)
        x = torch.randn(2, 15, 64)
        _, weights = attention(x)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    
    def test_mask(self) -> None:
        attention = TemporalAttention(hidden_dim=64)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10)
        mask[:, -3:] = 0  # Mask last 3 positions
        _, weights = attention(x, mask)
        assert torch.allclose(weights[:, -3:], torch.zeros(2, 3), atol=1e-5)


class TestBiLSTMTemporalModel:
    """Test BiLSTM temporal model."""
    
    def test_forward(self) -> None:
        model = BiLSTMTemporalModel(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )
        x = torch.randn(4, 30, 10)
        output = model(x, return_attention=True)
        
        assert "features" in output
        assert "hidden_states" in output
        assert "attention_weights" in output
        assert output["features"].shape == (4, 128)  # hidden*2 for bidirectional


class TestMultiHeadTemporalAttention:
    """Test multi-head temporal attention."""
    
    def test_forward(self) -> None:
        attention = MultiHeadTemporalAttention(
            input_dim=64,
            num_heads=4,
            head_dim=32,
        )
        x = torch.randn(2, 20, 64)
        output, weights = attention(x)
        
        assert output.shape == (2, 20, 64)
        assert weights.shape == (2, 20, 20)
    
    def test_attention_mask(self) -> None:
        attention = MultiHeadTemporalAttention(input_dim=32, num_heads=2, head_dim=16)
        x = torch.randn(2, 8, 32)
        mask = torch.ones(2, 8)
        mask[:, -2:] = 0
        
        output, weights = attention(x, mask)
        # Masked positions should have zero attention to them
        assert torch.allclose(weights[:, :, -2:].sum(), torch.tensor(0.0), atol=1e-5)