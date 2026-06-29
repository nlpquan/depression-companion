"""Shared fixtures for all tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from depression_companion.config import AppConfig, AudioConfig, TextConfig


@pytest.fixture
def sample_config() -> AppConfig:
    """Create a sample AppConfig for testing."""
    # ⭐ FIX: Pass a single dictionary, not multiple arguments
    config_dict = {
        "model": {
            "audio": {
                "sample_rate": 16000,
                "max_duration": 10.0,
                "feature_type": "wav2vec2",
                "pretrained_model": "facebook/wav2vec2-base-960h",
            },
            "text": {
                "max_length": 128,
                "feature_type": "bert",
                "pretrained_model": "bert-base-uncased",
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
            },
            "evaluation": {
                "metrics": ["accuracy", "f1"],
                "cv_folds": 3,
                "test_split": 0.2,
                "random_seed": 42,
            },
        },
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
            "datasets": {},
        },
        "logging": {
            "level": "DEBUG",
            "format": "{time} | {level} | {message}",
            "rotation": "1 MB",
            "retention": "7 days",
            "file": "logs/test.log",
        },
    }
    return AppConfig(**config_dict)  # ⭐ Unpack the dict as keyword arguments


@pytest.fixture
def audio_config(sample_config: AppConfig) -> AudioConfig:
    """Extract audio config for testing."""
    return sample_config.model.audio


@pytest.fixture
def text_config(sample_config: AppConfig) -> TextConfig:
    """Extract text config for testing."""
    return sample_config.model.text


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate synthetic audio for testing (1 second of 440Hz tone)."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave with some noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(len(t))
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_file(sample_audio: np.ndarray) -> Generator[Path, None, None]:
    """Create a temporary WAV file with synthetic audio."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, sample_audio, 16000)
        tmp_path = Path(tmp.name)

    yield tmp_path

    # Cleanup
    if tmp_path.exists():
        os.unlink(tmp_path)


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing - contains depressive language markers."""
    return (
        "I've been feeling really down lately. Nothing seems to matter anymore. "
        "I can't sleep at night and I'm tired all the time. Sometimes I feel "
        "hopeless and worthless. I don't enjoy things I used to love."
    )


@pytest.fixture
def sample_text_neutral() -> str:
    """Neutral sample text for testing."""
    return (
        "Today was a regular day. I went to work, had lunch with colleagues, "
        "and then came home. The weather was nice. I watched a movie in the evening."
    )


@pytest.fixture
def sample_text_crisis() -> str:
    """Sample text with crisis indicators for safety testing."""
    return (
        "I can't take this anymore. I want to end my life. Nothing helps "
        "and nobody understands. I've been thinking about suicide."
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
