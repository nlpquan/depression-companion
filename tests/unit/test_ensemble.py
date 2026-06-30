"""Tests for ensemble model."""

import torch
import pytest

from depression_companion.models.audio_model import Wav2Vec2Classifier
from depression_companion.models.text_model import BERTClassifier
from depression_companion.models.ensemble import MultimodalEnsemble


class TestMultimodalEnsemble:
    """Test multimodal ensemble."""
    
    @pytest.fixture
    def audio_model(self) -> Wav2Vec2Classifier:
        """Create audio model fixture."""
        return Wav2Vec2Classifier(num_classes=2, hidden_dim=64)
    
    @pytest.fixture
    def text_model(self) -> BERTClassifier:
        """Create text model fixture."""
        return BERTClassifier(num_classes=2, hidden_dim=64)
    
    def test_average_fusion(
        self, audio_model: Wav2Vec2Classifier, text_model: BERTClassifier
    ) -> None:
        """Test average fusion method."""
        ensemble = MultimodalEnsemble(
            audio_model=audio_model,
            text_model=text_model,
            fusion_method="average",
        )
        ensemble.eval()
        
        audio_batch = {"audio": torch.randn(4, 16000)}
        text_batch = {"input_ids": torch.randint(0, 30000, (4, 128))}
        
        with torch.no_grad():
            output = ensemble(audio_batch, text_batch)
        
        assert "logits" in output
        assert "probs" in output
        assert output["logits"].shape == (4, 2)
    
    def test_weighted_fusion(
        self, audio_model: Wav2Vec2Classifier, text_model: BERTClassifier
    ) -> None:
        """Test weighted fusion method."""
        ensemble = MultimodalEnsemble(
            audio_model=audio_model,
            text_model=text_model,
            fusion_method="weighted",
        )
        ensemble.eval()
        
        audio_batch = {"audio": torch.randn(4, 16000)}
        text_batch = {"input_ids": torch.randint(0, 30000, (4, 128))}
        
        with torch.no_grad():
            output = ensemble(audio_batch, text_batch)
        
        assert "audio_weight" in output
        assert "text_weight" in output
        assert output["audio_weight"] is not None
    
    def test_learnable_fusion(
        self, audio_model: Wav2Vec2Classifier, text_model: BERTClassifier
    ) -> None:
        """Test learnable fusion method."""
        ensemble = MultimodalEnsemble(
            audio_model=audio_model,
            text_model=text_model,
            fusion_method="learnable",
        )
        ensemble.eval()
        
        audio_batch = {"audio": torch.randn(4, 16000)}
        text_batch = {"input_ids": torch.randint(0, 30000, (4, 128))}
        
        with torch.no_grad():
            output = ensemble(audio_batch, text_batch)
        
        assert output["logits"].shape == (4, 2)
    
    def test_probs_sum_to_one(
        self, audio_model: Wav2Vec2Classifier, text_model: BERTClassifier
    ) -> None:
        """Test that ensemble probabilities sum to 1."""
        ensemble = MultimodalEnsemble(
            audio_model=audio_model,
            text_model=text_model,
            fusion_method="average",
        )
        ensemble.eval()
        
        audio_batch = {"audio": torch.randn(2, 16000)}
        text_batch = {"input_ids": torch.randint(0, 30000, (2, 128))}
        
        with torch.no_grad():
            output = ensemble(audio_batch, text_batch)
        
        sums = output["probs"].sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)