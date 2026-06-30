"""PyTorch datasets for depression detection."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from depression_companion.data.preprocessing import augment_audio
from torch.nn.utils.rnn import pad_sequence


class DepressionAudioDataset(Dataset):
    """Dataset for audio-based depression detection (DAIC-WOZ).
    
    Handles loading audio features and labels with optional augmentation.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        augment: bool = False,
        max_duration: float = 30.0,
        sample_rate: int = 16000,
    ):
        """Initialize dataset.
        
        Args:
            data_path: Path to audio features directory or DataFrame.
            metadata_path: Path to metadata CSV with labels.
            split: One of 'train', 'val', 'test'.
            augment: Whether to apply augmentation (training only).
            max_duration: Maximum audio duration in seconds.
            sample_rate: Target sample rate.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.augment = augment and split == "train"
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        # Load metadata
        if metadata_path:
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = self._create_dummy_metadata()
        
        # Filter by split
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        
        # Extract labels
        self.labels = self.metadata["phq8_score"].values if "phq8_score" in self.metadata.columns else np.zeros(len(self.metadata))
        
        # Binarize for classification
        self.binary_labels = (self.labels >= 10).astype(int)  # PHQ-8 >= 10 indicates depression
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with audio features, labels, and metadata.
        """
        row = self.metadata.iloc[idx]
        
        # Load audio features (or raw audio)
        audio = self._load_audio(row)
        
        # Pad or truncate
        audio = self._pad_audio(audio)
        
        # Augment if training
        if self.augment:
            audio = augment_audio(
                audio.numpy(),
                self.sample_rate,
            )
            audio = torch.from_numpy(audio).float() 
        
        return {
            "audio": audio,
            "label": torch.tensor(self.binary_labels[idx], dtype=torch.long),
            "phq8_score": torch.tensor(self.labels[idx], dtype=torch.float32),
            "participant_id": row.get("participant_id", str(idx)),
        }
    
    def _load_audio(self, row: pd.Series) -> torch.Tensor:
        """Load audio from file or generate synthetic for testing.
        
        Args:
            row: Metadata row.
            
        Returns:
            Audio tensor.
        """
        file_path = self.data_path / f"{row['participant_id']}.npy"
        
        if file_path.exists():
            audio = np.load(file_path)
            return torch.from_numpy(audio.astype(np.float32))
        else:
            # Generate synthetic audio for development
            duration = min(self.max_duration, 5.0)
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.sin(2 * np.pi * 220 * t) * 0.3
            return torch.from_numpy(audio.astype(np.float32))
    
    def _pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad or truncate audio to fixed length.
        
        Args:
            audio: Audio tensor.
            
        Returns:
            Padded/truncated audio tensor.
        """
        if len(audio) > self.max_samples:
            # Random crop during training, center crop otherwise
            if self.split == "train":
                start = np.random.randint(0, len(audio) - self.max_samples)
            else:
                start = (len(audio) - self.max_samples) // 2
            return audio[start:start + self.max_samples]
        else:
            # Pad with zeros
            padded = torch.zeros(self.max_samples)
            padded[:len(audio)] = audio
            return padded
    
    def _create_dummy_metadata(self) -> pd.DataFrame:
        """Create dummy metadata for testing."""
        import pandas as pd
        
        return pd.DataFrame({
            "participant_id": [f"P{i:03d}" for i in range(100)],
            "phq8_score": np.random.randint(0, 24, 100),
            "split": np.random.choice(["train", "val", "test"], 100, p=[0.7, 0.15, 0.15]),
        })


class DepressionTextDataset(Dataset):
    """Dataset for text-based depression detection.
    
    Supports Reddit mental health data and ESConv counseling data.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        max_length: int = 512,
    ):
        """Initialize dataset.
        
        Args:
            data_path: Path to text data directory or file.
            metadata_path: Path to metadata CSV.
            split: One of 'train', 'val', 'test'.
            max_length: Maximum sequence length.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        
        # Load data
        self.texts, self.labels = self._load_data(metadata_path)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with text, labels, and metadata.
        """
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "idx": idx,
        }
    
    def _load_data(
        self, metadata_path: Optional[Path] = None
    ) -> tuple[list[str], list[int]]:
        """Load text data and labels.
        
        Args:
            metadata_path: Path to metadata CSV.
            
        Returns:
            Tuple of (texts, labels).
        """
        if metadata_path and Path(metadata_path).exists():
            df = pd.read_csv(metadata_path)
            df = df[df["split"] == self.split]
            return df["text"].tolist(), df["label"].tolist()
        else:
            # Generate synthetic data for development
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> tuple[list[str], list[int]]:
        """Create dummy text data for testing."""
        depressive_texts = [
            "I've been feeling really down and hopeless lately.",
            "Nothing seems to matter anymore. I'm tired all the time.",
            "I can't concentrate on anything and I feel worthless.",
        ]
        neutral_texts = [
            "Today was a regular day. I went to work and came home.",
            "The weather is nice today. I might go for a walk.",
            "Just finished a good book. Looking forward to the weekend.",
        ]
        
        # Repeat to create enough samples
        n_samples = 50
        texts = (depressive_texts * (n_samples // 2))[:n_samples]
        labels = [1] * (n_samples // 2) + [0] * (n_samples // 2)
        
        return texts, labels

def collate_audio_batch(batch):
    """Collate function that pads audio to max length in batch."""
    audio_list = [item['audio'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    phq8 = torch.stack([item['phq8_score'] for item in batch])
    
    audio_padded = pad_sequence(audio_list, batch_first=True)  # [batch, time]
    
    return {
        'audio': audio_padded,
        'label': labels,
        'phq8_score': phq8,
    }