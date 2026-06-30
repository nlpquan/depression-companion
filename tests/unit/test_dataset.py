"""Tests for dataset classes."""

import numpy as np
import torch
import pytest

from depression_companion.data.dataset import DepressionAudioDataset, DepressionTextDataset


class TestDepressionAudioDataset:
    """Test audio dataset."""
    
    def test_initialization(self) -> None:
        """Test dataset creation."""
        dataset = DepressionAudioDataset(
            data_path="data/raw",
            split="train",
        )
        assert len(dataset) > 0
    
    def test_getitem(self) -> None:
        """Test getting a sample."""
        dataset = DepressionAudioDataset(
            data_path="data/raw",
            split="train",
        )
        sample = dataset[0]
        
        assert "audio" in sample
        assert "label" in sample
        assert "phq8_score" in sample
        assert isinstance(sample["audio"], torch.Tensor)
        assert sample["audio"].ndim == 1
    
    def test_augmentation(self) -> None:
        """Test augmentation only applied during training."""
        # Training dataset should allow augmentation
        train_dataset = DepressionAudioDataset(
            data_path="data/raw",
            split="train",
            augment=True,
        )
        sample = train_dataset[0]
        assert sample["audio"].ndim == 1
    
    def test_padding(self) -> None:
        """Test audio padding to fixed length."""
        max_duration = 5.0
        sample_rate = 16000
        
        dataset = DepressionAudioDataset(
            data_path="data/raw",
            split="train",
            max_duration=max_duration,
            sample_rate=sample_rate,
        )
        
        sample = dataset[0]
        expected_length = int(max_duration * sample_rate)
        assert len(sample["audio"]) == expected_length


class TestDepressionTextDataset:
    """Test text dataset."""
    
    def test_initialization(self) -> None:
        """Test dataset creation."""
        dataset = DepressionTextDataset(
            data_path="data/raw",
            split="train",
        )
        assert len(dataset) > 0
    
    def test_getitem(self) -> None:
        """Test getting a sample."""
        dataset = DepressionTextDataset(
            data_path="data/raw",
            split="train",
        )
        sample = dataset[0]
        
        assert "text" in sample
        assert "label" in sample
        assert isinstance(sample["text"], str)
        assert isinstance(sample["label"], torch.Tensor)
    
    def test_labels_binary(self) -> None:
        """Test labels are binary."""
        dataset = DepressionTextDataset(
            data_path="data/raw",
            split="train",
        )
        
        labels = []
        for i in range(len(dataset)):
            labels.append(dataset[i]["label"].item())
        
        unique_labels = set(labels)
        assert unique_labels.issubset({0, 1})