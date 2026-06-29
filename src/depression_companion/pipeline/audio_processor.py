"""Audio processing pipeline for feature extraction."""

import time
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
import torchaudio
from loguru import logger

from depression_companion.config import AudioConfig
from depression_companion.exceptions import AudioProcessingError, ValidationError
from depression_companion.pipeline.base import AudioFeatures, BaseProcessor


class AudioProcessor(BaseProcessor[Union[str, Path, np.ndarray], AudioFeatures]):
    """Extract audio features from speech recordings.

    Supports:
    - MFCC, mel spectrogram, spectral features via librosa
    - Wav2Vec2 embeddings via transformers (Phase 2)
    - Voice activity detection
    - Audio quality checks
    """

    def __init__(self, config: AudioConfig, device: Optional[torch.device] = None):
        """Initialize audio processor.

        Args:
            config: Audio processing configuration.
            device: Torch device.
        """
        super().__init__(config, device)
        self.sample_rate = config.sample_rate
        self.max_duration = config.max_duration

    def validate_input(self, input_data: Union[str, Path, np.ndarray]) -> bool:
        """Validate audio input.

        Args:
            input_data: Audio file path or numpy array.

        Returns:
            True if valid.

        Raises:
            ValidationError: If input is invalid.
        """
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.exists():
                raise ValidationError(f"Audio file not found: {path}")
            if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a"}:
                raise ValidationError(f"Unsupported audio format: {path.suffix}")
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim not in {1, 2}:
                raise ValidationError(
                    f"Audio array must be 1D or 2D, got shape: {input_data.shape}"
                )
        else:
            raise ValidationError(
                f"Expected str, Path, or np.ndarray, got: {type(input_data)}"
            )
        return True

    def load_audio(
        self, input_data: Union[str, Path, np.ndarray]
    ) -> tuple[np.ndarray, int]:
        """Load audio and resample to target sample rate.

        Args:
            input_data: Audio file path or numpy array.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        if isinstance(input_data, np.ndarray):
            audio = input_data
            if audio.ndim == 2:
                audio = audio.mean(axis=0)  # Convert stereo to mono
            sr = self.sample_rate  # Assume array is already at target rate
        else:
            audio, sr = librosa.load(
                str(input_data),
                sr=self.sample_rate,
                mono=True,
                duration=self.max_duration,
            )

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        return audio, self.sample_rate

    def process(self, input_data: Union[str, Path, np.ndarray]) -> AudioFeatures:
        """Extract audio features from input.

        Args:
            input_data: Audio file path or numpy array.

        Returns:
            AudioFeatures with extracted features.

        Raises:
            AudioProcessingError: If processing fails.
        """
        start_time = time.time()

        try:
            self.validate_input(input_data)
            audio, sr = self.load_audio(input_data)

            # Check minimum duration
            duration = len(audio) / sr
            if duration < 0.5:
                raise AudioProcessingError(
                    f"Audio too short: {duration:.2f}s (minimum 0.5s)"
                )

            logger.info(f"Processing audio: {duration:.2f}s, sr={sr}")

            # Extract features
            features_dict = self._extract_features(audio, sr)

            # Combine all features into single embedding vector
            embeddings = self._combine_features(features_dict)

            processing_time = time.time() - start_time

            return AudioFeatures(
                embeddings=embeddings,
                sample_rate=sr,
                duration=duration,
                metadata={
                    "feature_names": list(features_dict.keys()),
                    "processing_time": processing_time,
                    "original_duration": duration,
                },
            )

        except ValidationError:
            raise
        except Exception as e:
            raise AudioProcessingError(f"Audio processing failed: {str(e)}")

    def _extract_features(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """Extract multiple audio feature sets.

        Args:
            audio: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary of feature name to feature array.
        """
        features = {}

        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features["mfcc"] = mfcc.T  # Shape: (time, 13)

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        features["mel_spectrogram"] = mel_spec.T  # Shape: (time, 40)

        # Spectral features (aggregated)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

        features["spectral"] = np.concatenate(
            [
                spectral_centroid.T,
                spectral_rolloff.T,
                spectral_bandwidth.T,
                zero_crossing_rate.T,
            ],
            axis=1,
        )  # Shape: (time, 4)

        # Prosodic features
        rms_energy = librosa.feature.rms(y=audio)
        features["prosody"] = rms_energy.T  # Shape: (time, 1)

        # Pitch (F0) contour
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr, fill_na=0)
        features["pitch"] = f0.reshape(-1, 1)  # Shape: (time, 1)

        return features

    def _combine_features(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Combine multiple feature sets into a single embedding.

        Uses statistical functionals (mean, std, percentiles) to aggregate
        time-varying features into fixed-length vectors.

        Args:
            features: Dictionary of feature arrays.

        Returns:
            Combined embedding vector.
        """
        aggregated = []

        for name, feat_array in features.items():
            # Compute statistical functionals
            mean = np.mean(feat_array, axis=0)
            std = np.std(feat_array, axis=0)
            min_val = np.min(feat_array, axis=0)
            max_val = np.max(feat_array, axis=0)
            p25 = np.percentile(feat_array, 25, axis=0)
            p75 = np.percentile(feat_array, 75, axis=0)

            # Concatenate all statistics for this feature
            aggregated.append(np.concatenate([mean, std, min_val, max_val, p25, p75]))

        return np.concatenate(aggregated)
