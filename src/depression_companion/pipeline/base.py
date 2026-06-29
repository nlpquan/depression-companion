"""Abstract base classes for the processing pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

import numpy as np
import torch

# ---- Type Variables ----

T = TypeVar("T")  # Input type
U = TypeVar("U")  # Output type


# ---- Data Classes ----


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""

    embeddings: np.ndarray
    sample_rate: int
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextFeatures:
    """Container for extracted text features."""

    embeddings: np.ndarray
    tokens: list[str]
    attention_mask: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalFeatures:
    """Container for combined audio + text features."""

    audio: AudioFeatures
    text: TextFeatures
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Complete processing result with predictions and metadata."""

    input_id: str
    features: MultimodalFeatures
    predictions: dict[str, float]
    confidence: float
    processing_time: float
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- Abstract Base Classes ----


class BaseProcessor(ABC, Generic[T, U]):
    """Abstract base class for all processors.

    All processors must implement:
    - process: Main processing method
    - validate_input: Input validation
    - preprocess: Optional preprocessing step
    """

    def __init__(self, config: Any, device: Optional[torch.device] = None):
        """Initialize processor.

        Args:
            config: Configuration object for this processor.
            device: Torch device for computation.
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._is_initialized = False

    @abstractmethod
    def process(self, input_data: T) -> U:
        """Process input and return result.

        Args:
            input_data: Input data to process.

        Returns:
            Processed output.
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: T) -> bool:
        """Validate input before processing.

        Args:
            input_data: Input data to validate.

        Returns:
            True if valid.

        Raises:
            ValidationError: If input is invalid.
        """
        pass

    def preprocess(self, input_data: T) -> T:
        """Optional preprocessing step. Override in subclasses.

        Args:
            input_data: Raw input data.

        Returns:
            Preprocessed input data.
        """
        return input_data

    def initialize(self) -> None:
        """Initialize processor resources. Override in subclasses."""
        self._is_initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if processor is initialized."""
        return self._is_initialized


class BasePipeline(ABC):
    """Abstract base class for the complete pipeline.

    Coordinates multiple processors for end-to-end processing.
    """

    def __init__(self, config: Any):
        """Initialize pipeline.

        Args:
            config: Full application configuration.
        """
        self.config = config
        self._processors: dict[str, BaseProcessor] = {}

    @abstractmethod
    def run(
        self,
        input_id: str,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
    ) -> ProcessingResult:
        """Run the complete pipeline.

        Args:
            input_id: Unique identifier for this input.
            audio_path: Path to audio file (optional).
            text: Text input (optional).

        Returns:
            ProcessingResult with all outputs.
        """
        pass

    def add_processor(self, name: str, processor: BaseProcessor) -> None:
        """Register a processor in the pipeline.

        Args:
            name: Processor name.
            processor: Processor instance.
        """
        self._processors[name] = processor

    def get_processor(self, name: str) -> BaseProcessor:
        """Get a registered processor.

        Args:
            name: Processor name.

        Returns:
            Processor instance.

        Raises:
            KeyError: If processor not found.
        """
        if name not in self._processors:
            raise KeyError(f"Processor '{name}' not registered")
        return self._processors[name]
