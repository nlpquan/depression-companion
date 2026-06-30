"""Data preprocessing for depression datasets."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from loguru import logger


def create_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test split.
    
    Args:
        df: Input DataFrame.
        label_col: Name of label column.
        test_size: Proportion for test set.
        val_size: Proportion for validation set (from training).
        random_state: Random seed.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )
    
    # Second split: validation from training
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df[label_col],
        random_state=random_state,
    )
    
    logger.info(
        f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df


def create_time_based_split(
    df: pd.DataFrame,
    timestamp_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create time-based split (no data leakage).
    
    Args:
        df: Input DataFrame with timestamp column.
        timestamp_col: Name of timestamp column.
        test_size: Proportion for test set (most recent).
        val_size: Proportion for validation set.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]
    
    logger.info(
        f"Time-based split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df


def compute_class_weights(
    labels: np.ndarray,
) -> np.ndarray:
    """Compute class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels.
        
    Returns:
        Array of class weights (inverse frequency).
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    
    logger.info(f"Class distribution: {dict(zip(classes, counts))}")
    logger.info(f"Class weights: {dict(zip(classes, weights))}")
    
    return weights


def augment_audio(
    audio: np.ndarray,
    sample_rate: int,
    augmentation_prob: float = 0.5,
) -> np.ndarray:
    """Apply SpecAugment-style audio augmentation.
    
    Args:
        audio: Audio signal.
        sample_rate: Sample rate.
        augmentation_prob: Probability of applying each augmentation.
        
    Returns:
        Augmented audio.
    """
    augmented = audio.copy()
    
    # Time stretching (±10%)
    if np.random.random() < augmentation_prob:
        rate = np.random.uniform(0.9, 1.1)
        from librosa.effects import time_stretch
        augmented = time_stretch(augmented, rate=rate)
    
    # Pitch shifting (±2 semitones)
    if np.random.random() < augmentation_prob:
        steps = np.random.randint(-2, 3)
        from librosa.effects import pitch_shift
        augmented = pitch_shift(augmented, sr=sample_rate, n_steps=steps)
    
    # Add background noise
    if np.random.random() < augmentation_prob:
        noise_level = np.random.uniform(0.001, 0.01)
        noise = np.random.randn(len(augmented))
        augmented = augmented + noise_level * noise
    
    # Volume perturbation
    if np.random.random() < augmentation_prob:
        gain = np.random.uniform(0.8, 1.2)
        augmented = augmented * gain
    
    return augmented