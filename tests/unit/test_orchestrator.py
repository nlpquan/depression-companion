"""Tests for the pipeline orchestrator."""

import pytest

from depression_companion.config import AppConfig
from depression_companion.exceptions import PipelineError
from depression_companion.pipeline.base import ProcessingResult
from depression_companion.pipeline.orchestrator import DepressionDetectionPipeline


class TestDepressionDetectionPipeline:
    """Test pipeline orchestration."""
    
    def test_initialization(self, sample_config: AppConfig) -> None:
        """Test pipeline initializes correctly."""
        pipeline = DepressionDetectionPipeline(sample_config)
        
        assert "audio" in pipeline._processors
        assert "text" in pipeline._processors
    
    def test_run_with_both_modalities(
        self,
        sample_config: AppConfig,
        sample_audio_file: str,
        sample_text: str,
    ) -> None:
        """Test pipeline with both audio and text."""
        pipeline = DepressionDetectionPipeline(sample_config)
        result = pipeline.run(
            input_id="test_001",
            audio_path=sample_audio_file,
            text=sample_text,
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.input_id == "test_001"
        assert result.processing_time > 0
        assert "depression_score" in result.predictions
        assert 0 <= result.predictions["depression_score"] <= 1
        
        # Should have no errors
        assert len(result.errors) == 0
    
    def test_run_with_audio_only(
        self,
        sample_config: AppConfig,
        sample_audio_file: str,
    ) -> None:
        """Test pipeline with only audio input."""
        pipeline = DepressionDetectionPipeline(sample_config)
        result = pipeline.run(
            input_id="test_002",
            audio_path=sample_audio_file,
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.features.audio is not None
        assert result.features.text is not None  # Filled with zeros
    
    def test_run_with_text_only(
        self,
        sample_config: AppConfig,
        sample_text: str,
    ) -> None:
        """Test pipeline with only text input."""
        pipeline = DepressionDetectionPipeline(sample_config)
        result = pipeline.run(
            input_id="test_003",
            text=sample_text,
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.features.text is not None
        assert result.features.audio is not None  # Filled with silence
    
    def test_run_no_input(self, sample_config: AppConfig) -> None:
        """Test that pipeline fails with no input."""
        pipeline = DepressionDetectionPipeline(sample_config)
        
        with pytest.raises(PipelineError, match="At least one"):
            pipeline.run(input_id="test_004")
    
    def test_predictions_are_bounded(
        self,
        sample_config: AppConfig,
        sample_text: str,
    ) -> None:
        """Test that predictions are in valid range [0, 1]."""
        pipeline = DepressionDetectionPipeline(sample_config)
        result = pipeline.run(input_id="test_005", text=sample_text)
        
        for key, value in result.predictions.items():
            if key != "confidence":
                assert 0 <= value <= 1, f"{key} = {value} is out of bounds"
    
    def test_depressive_text_scores_higher(
        self,
        sample_config: AppConfig,
        sample_text: str,
        sample_text_neutral: str,
    ) -> None:
        """Test that depressive text gets higher depression scores."""
        pipeline = DepressionDetectionPipeline(sample_config)
        
        dep_result = pipeline.run(input_id="dep", text=sample_text)
        neutral_result = pipeline.run(input_id="neutral", text=sample_text_neutral)
        
        assert (
            dep_result.predictions["depression_score"]
            > neutral_result.predictions["depression_score"]
        )