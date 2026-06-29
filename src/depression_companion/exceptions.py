"""Custom exceptions for the Depression Companion system."""

from typing import Optional, Any


class DepressionCompanionError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(DepressionCompanionError):
    """Configuration-related errors."""

    pass


class DataLoadError(DepressionCompanionError):
    """Data loading and validation errors."""

    pass


class AudioProcessingError(DepressionCompanionError):
    """Audio processing pipeline errors."""

    pass


class TextProcessingError(DepressionCompanionError):
    """Text processing pipeline errors."""

    pass


class ModelError(DepressionCompanionError):
    """Model inference or training errors."""

    pass


class ValidationError(DepressionCompanionError):
    """Input validation errors."""

    pass


class PipelineError(DepressionCompanionError):
    """Pipeline orchestration errors."""

    pass
