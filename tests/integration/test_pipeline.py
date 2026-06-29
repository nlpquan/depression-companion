"""Integration tests for the complete pipeline."""

import time

import numpy as np

from depression_companion.config import AppConfig
from depression_companion.pipeline.orchestrator import DepressionDetectionPipeline


class TestPipelineIntegration:
    """End-to-end pipeline integration tests."""
    
    def test_full_pipeline_with_realistic_input(
        self,
        sample_config: AppConfig,
        sample_audio_file: str,
        sample_text: str,
    ) -> None:
        """Test complete pipeline with realistic multimodal input."""
        pipeline = DepressionDetectionPipeline(sample_config)
        
        start = time.time()
        result = pipeline.run(
            input_id="integration_test_001",
            audio_path=sample_audio_file,
            text=sample_text,
        )
        elapsed = time.time() - start
        
        # Should process in reasonable time
        assert elapsed < 10, f"Pipeline took {elapsed:.2f}s"
        
        # Check result structure
        assert result.input_id == "integration_test_001"
        assert result.features.audio.embeddings.shape[0] > 0
        assert result.features.text.embeddings.shape[0] > 0
        assert result.confidence >= 0
        assert result.processing_time > 0
        assert len(result.errors) == 0
    
    def test_multiple_runs_consistent(
        self,
        sample_config: AppConfig,
        sample_text: str,
    ) -> None:
        """Test that pipeline produces consistent results."""
        pipeline = DepressionDetectionPipeline(sample_config)
        
        result1 = pipeline.run(input_id="run1", text=sample_text)
        result2 = pipeline.run(input_id="run2", text=sample_text)
        
        # Results should be identical for same input
        assert result1.predictions == result2.predictions
    
    def test_pipeline_with_long_audio(
        self,
        sample_config: AppConfig,
    ) -> None:
        """Test pipeline handles longer audio files."""
        pipeline = DepressionDetectionPipeline(sample_config)
        
        # Create 5 seconds of audio
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        long_audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        
        result = pipeline.run(
            input_id="long_audio_test",
            audio_path=long_audio,
            text="I feel tired today.",
        )
        
        assert result.features.audio.duration > 1.0
        assert result.features.audio.duration <= 10.0  # max_duration