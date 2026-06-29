"""Tests for the TextProcessor."""

import numpy as np
import pytest

from depression_companion.config import TextConfig
from depression_companion.exceptions import ValidationError
from depression_companion.pipeline.text_processor import TextProcessor
from depression_companion.pipeline.base import TextFeatures


class TestTextProcessor:
    """Test text processing functionality."""
    
    def test_initialization(self, text_config: TextConfig) -> None:
        """Test processor initializes correctly."""
        processor = TextProcessor(text_config)
        assert processor.max_length == 128
        assert processor.model_name == "bert-base-uncased"
        assert not processor.is_initialized
    
    def test_validate_text(self, text_config: TextConfig, sample_text: str) -> None:
        """Test validation of text input."""
        processor = TextProcessor(text_config)
        assert processor.validate_input(sample_text) is True
    
    def test_validate_empty_text(self, text_config: TextConfig) -> None:
        """Test validation fails for empty text."""
        processor = TextProcessor(text_config)
        with pytest.raises(ValidationError, match="empty"):
            processor.validate_input("")
    
    def test_validate_whitespace_text(self, text_config: TextConfig) -> None:
        """Test validation fails for whitespace-only text."""
        processor = TextProcessor(text_config)
        with pytest.raises(ValidationError, match="empty"):
            processor.validate_input("   \n\t  ")
    
    def test_validate_too_long(self, text_config: TextConfig) -> None:
        """Test validation fails for excessively long text."""
        processor = TextProcessor(text_config)
        long_text = "x" * 10001
        with pytest.raises(ValidationError, match="too long"):
            processor.validate_input(long_text)
    
    def test_validate_invalid_type(self, text_config: TextConfig) -> None:
        """Test validation fails for non-string input."""
        processor = TextProcessor(text_config)
        with pytest.raises(ValidationError, match="Expected"):
            processor.validate_input(123)  # type: ignore
    
    def test_process(self, text_config: TextConfig, sample_text: str) -> None:
        """Test full text processing."""
        processor = TextProcessor(text_config)
        result = processor.process(sample_text)
        
        assert isinstance(result, TextFeatures)
        assert result.embeddings.ndim == 1
        assert len(result.tokens) > 0
        assert result.metadata["char_count"] == len(sample_text)
        assert result.metadata["word_count"] > 0
    
    def test_linguistic_features(self, text_config: TextConfig, sample_text: str) -> None:
        """Test linguistic feature extraction."""
        processor = TextProcessor(text_config)
        features = processor._extract_linguistic_features(sample_text)
        
        assert features.shape == (10,)
        assert features.dtype == np.float32
        
        # Word count should be positive
        assert features[0] > 0
        
        # Type-token ratio should be between 0 and 1
        assert 0 <= features[3] <= 1
    
    def test_sentiment_features(self, text_config: TextConfig, sample_text: str) -> None:
        """Test sentiment feature extraction."""
        processor = TextProcessor(text_config)
        
        # Depressive text
        dep_features = processor._extract_sentiment(sample_text)
        assert dep_features.shape == (4,)
        
        # Neutral text
        neutral_features = processor._extract_sentiment(
            "Today was a good day. I feel happy and productive."
        )
        
        # Depressive text should have higher negative sentiment
        assert dep_features[0] > neutral_features[0]  # neg score
    
    def test_keyword_features_depressive(
        self, text_config: TextConfig, sample_text: str
    ) -> None:
        """Test keyword extraction for depressive text."""
        processor = TextProcessor(text_config)
        features = processor._extract_keywords(sample_text)
        
        assert features.shape == (5,)
        # Should detect depression keywords
        assert features[0] > 0  # depression_count
        # Should detect sleep keywords
        assert features[2] > 0  # sleep_count
    
    def test_keyword_features_crisis(
        self, text_config: TextConfig, sample_text_crisis: str
    ) -> None:
        """Test keyword extraction for crisis text."""
        processor = TextProcessor(text_config)
        features = processor._extract_keywords(sample_text_crisis)
        
        # Should detect crisis keywords
        assert features[3] > 0  # crisis_count
        assert features[4] == 1  # crisis_phrase_present
    
    def test_keyword_features_neutral(
        self, text_config: TextConfig, sample_text_neutral: str
    ) -> None:
        """Test keyword extraction for neutral text."""
        processor = TextProcessor(text_config)
        features = processor._extract_keywords(sample_text_neutral)
        
        # Should have zero or very low counts
        assert features[3] == 0  # crisis_count
        assert features[4] == 0  # crisis_phrase_present