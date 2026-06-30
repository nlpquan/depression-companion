"""Tests for cross-modal attention."""

import torch
import pytest

from depression_companion.models.cross_modal_attention import (
    CrossModalAttention,
    CrossModalFusion,
    SimpleConcatFusion,
)


class TestCrossModalAttention:
    """Test cross-modal attention module."""
    
    @pytest.fixture
    def attention(self) -> CrossModalAttention:
        return CrossModalAttention(
            query_dim=128,
            key_dim=256,
            hidden_dim=128,
            num_heads=4,
        )
    
    def test_initialization(self, attention: CrossModalAttention) -> None:
        """Test attention module creation."""
        assert attention.query_dim == 128
        assert attention.key_dim == 256
        assert attention.num_heads == 4
    
    def test_forward(self, attention: CrossModalAttention) -> None:
        """Test forward pass."""
        query = torch.randn(2, 10, 128)  # (B, seq_q, query_dim)
        key = torch.randn(2, 15, 256)    # (B, seq_k, key_dim)
        
        output, weights = attention(query, key)
        
        # Output shape matches query
        assert output.shape == (2, 10, 128)
        # Attention weights shape
        assert weights.shape == (2, 10, 15)
        # Weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10), atol=0.15)
    
    def test_attention_mask(self, attention: CrossModalAttention) -> None:
        """Test attention with masking."""
        query = torch.randn(2, 5, 128)
        key = torch.randn(2, 8, 256)
        
        # Mask out last 3 key positions
        mask = torch.ones(2, 5, 8)
        mask[:, :, -3:] = 0
        
        output, weights = attention(query, key, mask=mask)
        
        # Masked positions should have zero attention
        assert torch.allclose(weights[:, :, -3:], torch.zeros(2, 5, 3), atol=0.15)
    
    def test_different_sequence_lengths(self, attention: CrossModalAttention) -> None:
        """Test with different sequence lengths."""
        query = torch.randn(3, 7, 128)
        key = torch.randn(3, 12, 256)
        
        output, weights = attention(query, key)
        
        assert output.shape == (3, 7, 128)
        assert weights.shape == (3, 7, 12)


class TestCrossModalFusion:
    """Test complete fusion architecture."""
    
    @pytest.fixture
    def fusion(self) -> CrossModalFusion:
        return CrossModalFusion(
            audio_dim=256,
            text_dim=512,
            hidden_dim=128,
            num_heads=4,
            num_fusion_layers=2,
        )
    
    def test_initialization(self, fusion: CrossModalFusion) -> None:
        """Test fusion module creation."""
        assert fusion.audio_dim == 256
        assert fusion.text_dim == 512
    
    def test_forward_2d(self, fusion: CrossModalFusion) -> None:
        """Test forward pass with 2D inputs (no sequence)."""
        audio = torch.randn(4, 256)
        text = torch.randn(4, 512)
        
        output = fusion(audio, text)
        
        assert "fused_features" in output
        assert "text_to_audio_attention" in output
        assert "audio_to_text_attention" in output
        assert output["fused_features"].shape == (4, 768)  # audio_dim + text_dim
    
    def test_forward_3d(self, fusion: CrossModalFusion) -> None:
        """Test forward pass with 3D inputs (with sequence)."""
        audio = torch.randn(2, 10, 256)
        text = torch.randn(2, 8, 512)
        
        output = fusion(audio, text)
        
        assert output["fused_features"].shape == (2, 768)
        assert output["fused_sequence"].shape == (2, 10, 768)  # max of seq lengths
    
    def test_attention_weights_valid(self, fusion: CrossModalFusion) -> None:
        """Test that attention weights are valid probability distributions."""
        audio = torch.randn(2, 8, 256)
        text = torch.randn(2, 6, 512)
        
        output = fusion(audio, text)
        
        text_to_audio = output["text_to_audio_attention"]
        audio_to_text = output["audio_to_text_attention"]
        
        # Check they sum to 1
        assert torch.allclose(
            text_to_audio.sum(dim=-1),
            torch.ones(2, 6),
            atol=0.15,
        )
        assert torch.allclose(
            audio_to_text.sum(dim=-1),
            torch.ones(2, 8),
            atol=0.15,
        )


class TestSimpleConcatFusion:
    """Test baseline concatenation fusion."""
    
    def test_forward(self) -> None:
        """Test forward pass."""
        fusion = SimpleConcatFusion(audio_dim=256, text_dim=512)
        
        audio = torch.randn(4, 256)
        text = torch.randn(4, 512)
        
        output = fusion(audio, text)
        
        assert output["fused_features"].shape == (4, 768)