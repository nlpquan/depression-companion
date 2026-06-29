"""Tests for the AudioProcessor."""

import numpy as np
import pytest

from depression_companion.config import AudioConfig
from depression_companion.exceptions import ValidationError, AudioProcessingError
from depression_companion.pipeline.audio_processor import AudioProcessor
from depression_companion.pipeline.base import AudioFeatures


class TestAudioProcessor:
    """Test audio processing functionality."""

    def test_initialization(self, audio_config: AudioConfig) -> None:
        """Test processor initializes correctly."""
        processor = AudioProcessor(audio_config)
        assert processor.sample_rate == 16000
        assert processor.max_duration == 10.0
        assert not processor.is_initialized

    def test_validate_audio_file(
        self, audio_config: AudioConfig, sample_audio_file: str
    ) -> None:
        """Test validation of audio file input."""
        processor = AudioProcessor(audio_config)
        assert processor.validate_input(sample_audio_file) is True

    def test_validate_audio_array(
        self, audio_config: AudioConfig, sample_audio: np.ndarray
    ) -> None:
        """Test validation of numpy array input."""
        processor = AudioProcessor(audio_config)
        assert processor.validate_input(sample_audio) is True

    def test_validate_nonexistent_file(self, audio_config: AudioConfig) -> None:
        """Test validation fails for nonexistent file."""
        processor = AudioProcessor(audio_config)
        with pytest.raises(ValidationError, match="not found"):
            processor.validate_input("nonexistent.wav")

    def test_validate_invalid_type(self, audio_config: AudioConfig) -> None:
        """Test validation fails for invalid input type."""
        processor = AudioProcessor(audio_config)
        with pytest.raises(ValidationError, match="Expected"):
            processor.validate_input(42)  # type: ignore

    def test_process_from_file(
        self, audio_config: AudioConfig, sample_audio_file: str
    ) -> None:
        """Test full processing from audio file."""
        processor = AudioProcessor(audio_config)
        result = processor.process(sample_audio_file)

        assert isinstance(result, AudioFeatures)
        assert result.sample_rate == 16000
        assert result.duration > 0
        assert result.embeddings.ndim == 1  # Flattened feature vector

        # Check metadata
        assert "feature_names" in result.metadata
        assert "processing_time" in result.metadata

    def test_process_from_array(
        self, audio_config: AudioConfig, sample_audio: np.ndarray
    ) -> None:
        """Test processing from numpy array."""
        processor = AudioProcessor(audio_config)
        result = processor.process(sample_audio)

        assert isinstance(result, AudioFeatures)
        assert result.duration > 0
        assert result.embeddings.shape[0] > 0

    def test_process_short_audio(self, audio_config: AudioConfig) -> None:
        """Test that very short audio is rejected."""
        processor = AudioProcessor(audio_config)
        short_audio = np.zeros(100)  # Too short

        with pytest.raises(AudioProcessingError, match="too short"):
            processor.process(short_audio)

    def test_stereo_to_mono(self, audio_config: AudioConfig) -> None:
        """Test stereo audio is converted to mono."""
        processor = AudioProcessor(audio_config)
        stereo = np.random.randn(2, 32000).astype(np.float32)

        result = processor.process(stereo)
        assert isinstance(result, AudioFeatures)

    def test_feature_extraction(
        self, audio_config: AudioConfig, sample_audio: np.ndarray
    ) -> None:
        """Test that features are extracted correctly."""
        processor = AudioProcessor(audio_config)
        features = processor._extract_features(sample_audio, 16000)

        assert "mfcc" in features
        assert "mel_spectrogram" in features
        assert "spectral" in features
        assert "prosody" in features
        assert "pitch" in features

        # Check shapes
        assert features["mfcc"].shape[1] == 13  # 13 MFCC coefficients
        assert features["mel_spectrogram"].shape[1] == 40  # 40 mel bands

    def test_embedding_size_consistent(
        self, audio_config: AudioConfig, sample_audio_file: str
    ) -> None:
        """Test that embedding size is consistent for same config."""
        processor = AudioProcessor(audio_config)

        result1 = processor.process(sample_audio_file)
        result2 = processor.process(sample_audio_file)

        assert result1.embeddings.shape == result2.embeddings.shape
