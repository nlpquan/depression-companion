"""Pipeline orchestrator that coordinates audio and text processing."""

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger

from depression_companion.config import AppConfig
from depression_companion.exceptions import PipelineError
from depression_companion.pipeline.audio_processor import AudioProcessor
from depression_companion.pipeline.base import (
    BasePipeline,
    MultimodalFeatures,
    ProcessingResult,
)
from depression_companion.pipeline.text_processor import TextProcessor


class DepressionDetectionPipeline(BasePipeline):
    """Complete pipeline for depression detection from multimodal input.
    
    Coordinates audio and text processing, feature fusion, and prediction.
    """
    
    def __init__(self, config: AppConfig):
        """Initialize the depression detection pipeline.
        
        Args:
            config: Full application configuration.
        """
        super().__init__(config)
        
        # Initialize processors
        self.audio_processor = AudioProcessor(config.model.audio)
        self.text_processor = TextProcessor(config.model.text)
        
        # Register processors
        self.add_processor("audio", self.audio_processor)
        self.add_processor("text", self.text_processor)
        
        logger.info("DepressionDetectionPipeline initialized")
    
    def run(
        self,
        input_id: str,
        audio_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
    ) -> ProcessingResult:
        """Run the complete pipeline.
        
        At least one of audio_path or text must be provided.
        
        Args:
            input_id: Unique identifier for this input.
            audio_path: Path to audio file (optional).
            text: Text input (optional).
            
        Returns:
            ProcessingResult with all outputs.
            
        Raises:
            PipelineError: If neither audio nor text is provided.
        """
        start_time = time.time()
        errors = []
        
        if audio_path is None and text is None:
            raise PipelineError("At least one of audio_path or text must be provided")
        
        logger.info(f"Running pipeline for input: {input_id}")
        
        # Process audio
        audio_features = None
        if audio_path is not None:
            try:
                audio_features = self.audio_processor.process(audio_path)
                logger.info(f"Audio processed: {audio_features.duration:.2f}s")
            except Exception as e:
                errors.append(f"Audio processing error: {str(e)}")
                logger.error(f"Audio processing failed: {e}")
        
        # Process text
        text_features = None
        if text is not None:
            try:
                text_features = self.text_processor.process(text)
                logger.info(f"Text processed: {text_features.metadata.get('word_count', 0)} words")
            except Exception as e:
                errors.append(f"Text processing error: {str(e)}")
                logger.error(f"Text processing failed: {e}")
        
        # Create multimodal features (handle missing modalities)
        if audio_features is None and text_features is None:
            raise PipelineError("Both audio and text processing failed")
        
        # Use empty features for missing modalities
        if audio_features is None:
            logger.warning("No audio features available, using zeros")
            audio_features = self.audio_processor.process(
                np.zeros(16000)  # 1 second of silence
            )
        
        if text_features is None:
            logger.warning("No text features available, using zeros")
            text_features = self._create_zero_text_features()
        
        multimodal = MultimodalFeatures(
            audio=audio_features,
            text=text_features,
            metadata={
                "audio_provided": audio_path is not None,
                "text_provided": text is not None,
            },
        )
        
        # Generate predictions (placeholder - replaced in Phase 2)
        predictions = self._predict(multimodal)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            input_id=input_id,
            features=multimodal,
            predictions=predictions,
            confidence=predictions.get("confidence", 0.0),
            processing_time=processing_time,
            errors=errors,
            metadata={
                "total_processing_time": processing_time,
                "has_errors": len(errors) > 0,
            },
        )
    
    def _create_zero_text_features(self):
        """Create placeholder text features when text is not provided.
        Returns:
        TextFeatures object with zero embeddings.
        """
        from depression_companion.pipeline.base import TextFeatures
        
        return TextFeatures(
        embeddings=np.zeros(384, dtype=np.float32),
        tokens=[],  # Required field - empty list
        attention_mask=None,
        metadata={
            "word_count": 0,
            "placeholder": True,
        },
    )
    
    def _predict(self, features: MultimodalFeatures) -> dict[str, float]:
        """Generate predictions from multimodal features.
        
        This is a placeholder. In Phase 2, this will use the trained model.
        
        Args:
            features: Multimodal features.
            
        Returns:
            Dictionary of prediction scores.
        """
        # Simple heuristic-based prediction for now
        # Will be replaced with actual model in Phase 2
        
        audio_energy = np.mean(np.abs(features.audio.embeddings))
        text_embedding = features.text.embeddings
        
        # Simple baseline: check sentiment and keyword features
        # text_embedding indices: [8] = negation_ratio, 
        # sentiment starts at index 10
        neg_sentiment = text_embedding[10] if len(text_embedding) > 10 else 0
        pos_sentiment = text_embedding[12] if len(text_embedding) > 12 else 0
        crisis_count = text_embedding[17] if len(text_embedding) > 17 else 0
        
        # Heuristic score
        depression_score = (
            neg_sentiment * 0.4 +
            (1 - pos_sentiment) * 0.2 +
            (1 - min(audio_energy * 10, 1)) * 0.3 +
            min(crisis_count * 0.3, 0.3)
        )
        
        depression_score = min(max(depression_score, 0.0), 1.0)
        
        return {
            "depression_score": float(depression_score),
            "anxiety_score": float(depression_score * 0.8),
            "mood_score": float(1.0 - depression_score),
            "confidence": 0.5,  # Low confidence for heuristic
        }