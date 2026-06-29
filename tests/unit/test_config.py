"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from depression_companion.config import AppConfig, load_config


class TestAppConfig:
    """Test configuration validation and defaults."""

    def test_default_config(self) -> None:
        """Test creating config with defaults."""
        config = AppConfig()

        assert config.model.audio.sample_rate == 16000
        assert config.model.text.max_length == 512
        assert config.model.training.batch_size == 16
        assert config.model.evaluation.cv_folds == 5

    def test_config_validation(self) -> None:
        """Test config with custom values."""
        config = AppConfig(
            model={
                "audio": {"sample_rate": 22050},
                "text": {"max_length": 256},
            }
        )

        assert config.model.audio.sample_rate == 22050
        assert config.model.text.max_length == 256
        # Unchanged defaults
        assert config.model.audio.feature_type == "wav2vec2"

    def test_invalid_config(self) -> None:
        """Test that invalid config raises error."""
        with pytest.raises(Exception):
            AppConfig(model={"audio": {"sample_rate": "not_a_number"}})

    def test_security_config_defaults(self) -> None:
        """Test security config has correct defaults."""
        config = AppConfig()

        assert ".wav" in config.security.allowed_extensions
        assert config.security.data_retention_days == 30


class TestLoadConfig:
    """Test loading config from YAML file."""

    def test_load_valid_yaml(self) -> None:
        """Test loading a valid YAML config file."""
        yaml_content = """
model:
  audio:
    sample_rate: 22050
  text:
    max_length: 128
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write(yaml_content)
            tmp_path = Path(tmp.name)

        try:
            config = load_config(tmp_path)
            assert config.model.audio.sample_rate == 22050
            assert config.model.text.max_length == 128
        finally:
            tmp_path.unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test that loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_load_invalid_yaml(self) -> None:
        """Test that malformed YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("invalid: [unclosed bracket")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(tmp_path)
        finally:
            tmp_path.unlink()
