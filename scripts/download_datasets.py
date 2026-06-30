"""Download and prepare depression detection datasets.

Datasets:
- DAIC-WOZ: Clinical interviews for depression detection
- Reddit Self-Reported Depression: Reddit posts with depression labels
- ESConv: Emotional Support Conversations
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger
from tqdm import tqdm


def setup_logging() -> None:
    """Configure logging."""
    logger.add(
        "logs/dataset_download.log",
        rotation="10 MB",
        level="INFO",
    )


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.
    
    Args:
        url: Download URL.
        dest_path: Destination path.
        desc: Progress bar description.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest_path, "wb") as f, tqdm(
        desc=desc,
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def prepare_daic_woz(
    output_dir: Path,
    download: bool = True,
) -> pd.DataFrame:
    """Prepare DAIC-WOZ dataset.
    
    DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz) contains
    clinical interviews with depression labels (PHQ-8 scores).
    
    Args:
        output_dir: Output directory for processed data.
        download: Whether to attempt download (requires credentials).
        
    Returns:
        DataFrame with processed metadata.
    """
    logger.info("Preparing DAIC-WOZ dataset...")
    
    raw_dir = output_dir / "raw" / "daic_woz"
    processed_dir = output_dir / "processed" / "daic_woz"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if download:
        logger.warning(
            "DAIC-WOZ requires approval from USC. "
            "Visit: https://dcapswoz.ict.usc.edu/"
        )
        logger.info("Creating placeholder structure for development...")
    
    # Create synthetic data for development
    np.random.seed(42)
    n_participants = 189  # Actual DAIC-WOZ size
    
    metadata = pd.DataFrame({
        "participant_id": [f"P{i:03d}" for i in range(n_participants)],
        "gender": np.random.choice(["M", "F"], n_participants),
        "phq8_score": np.random.randint(0, 24, n_participants),
        "depression_diagnosis": np.random.choice([0, 1], n_participants, p=[0.7, 0.3]),
        "interview_duration": np.random.uniform(300, 900, n_participants),
    })
    
    # Add split assignment
    metadata["split"] = np.random.choice(
        ["train", "val", "test"],
        n_participants,
        p=[0.7, 0.15, 0.15],
    )
    
    # Save metadata
    metadata_path = processed_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata: {metadata_path}")
    
    # Generate synthetic audio features for development
    logger.info("Generating synthetic audio features...")
    features_dir = processed_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    for pid in metadata["participant_id"]:
        # Random audio features (simulating Wav2Vec2 embeddings)
        duration = np.random.uniform(5, 30)
        n_samples = int(duration * 16000)
        features = np.random.randn(n_samples).astype(np.float32)
        np.save(features_dir / f"{pid}.npy", features)
    
    logger.info(f"Generated features for {n_participants} participants")
    
    return metadata


def prepare_reddit_depression(
    output_dir: Path,
    download: bool = True,
) -> pd.DataFrame:
    """Prepare Reddit depression dataset.
    
    Uses the Self-Reported Mental Health Diagnoses dataset or
    creates synthetic data for development.
    
    Args:
        output_dir: Output directory for processed data.
        download: Whether to attempt download.
        
    Returns:
        DataFrame with processed data.
    """
    logger.info("Preparing Reddit depression dataset...")
    
    processed_dir = output_dir / "processed" / "reddit_depression"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic data for development
    np.random.seed(42)
    n_samples = 5000
    
    depressive_texts = [
        "I've been feeling really down and hopeless lately. Nothing seems to bring me joy anymore.",
        "Can't sleep, can't eat, just want to stay in bed all day. Everything feels pointless.",
        "I feel so worthless and empty. It's like I'm just going through the motions.",
        "The anxiety is overwhelming. I can't stop worrying about everything.",
        "I'm so tired of feeling this way. Nothing helps. I just want it to stop.",
        "Every day feels like a struggle. I don't see the point in anything anymore.",
        "I push everyone away because I feel like a burden. They'd be better off without me.",
        "My mind won't stop racing. I can't focus on anything and I'm exhausted.",
    ]
    
    neutral_texts = [
        "Had a good day at work today. Finished a project I've been working on.",
        "The weather is beautiful today. Thinking of going for a hike this weekend.",
        "Just finished reading a great book. Anyone have recommendations for similar ones?",
        "Made a new recipe for dinner tonight. Turned out pretty well!",
        "Looking forward to the game this weekend. Should be a good match.",
        "Started learning guitar recently. It's challenging but fun.",
        "Had a nice catch-up with an old friend today. Always good to reconnect.",
        "Finally organized my workspace. Feels so much better now.",
    ]
    
    # Generate samples
    texts = []
    labels = []
    
    for i in range(n_samples):
        if i < n_samples * 0.4:  # 40% depressive
            texts.append(np.random.choice(depressive_texts))
            labels.append(1)
        else:  # 60% neutral
            texts.append(np.random.choice(neutral_texts))
            labels.append(0)
    
    # Add timestamps for time-based splitting
    import datetime
    base_date = datetime.datetime(2020, 1, 1)
    timestamps = [
        base_date + datetime.timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
        )
        for _ in range(n_samples)
    ]
    
    metadata = pd.DataFrame({
        "text": texts,
        "label": labels,
        "timestamp": sorted(timestamps),
    })
    
    # Time-based split
    metadata = metadata.sort_values("timestamp")
    n_test = int(n_samples * 0.2)
    n_val = int(n_samples * 0.15)
    
    metadata["split"] = "train"
    metadata.iloc[-(n_test + n_val):-n_test, metadata.columns.get_loc("split")] = "val"
    metadata.iloc[-n_test:, metadata.columns.get_loc("split")] = "test"
    
    # Save
    metadata_path = processed_dir / "posts.csv"
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved {n_samples} posts to {metadata_path}")
    
    return metadata


def prepare_esconv(output_dir: Path) -> pd.DataFrame:
    """Prepare ESConv (Emotional Support Conversations) dataset.
    
    Args:
        output_dir: Output directory for processed data.
        
    Returns:
        DataFrame with processed conversations.
    """
    logger.info("Preparing ESConv dataset...")
    
    processed_dir = output_dir / "processed" / "esconv"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic counseling conversations for development
    conversations = [
        {
            "conversation_id": "conv_001",
            "seeker_text": "I've been feeling really down lately. Nothing seems to help.",
            "supporter_text": "I hear you. It sounds like you're going through a difficult time. Can you tell me more about what's been happening?",
            "emotion": "sadness",
            "strategy": "reflection",
        },
        {
            "conversation_id": "conv_002",
            "seeker_text": "I'm so anxious about my exams. I can't sleep or concentrate.",
            "supporter_text": "That sounds really stressful. Many people struggle with test anxiety. Have you tried any relaxation techniques?",
            "emotion": "anxiety",
            "strategy": "normalization",
        },
        {
            "conversation_id": "conv_003",
            "seeker_text": "I feel like nobody understands what I'm going through.",
            "supporter_text": "It can be really isolating when you feel that way. I want you to know that your feelings are valid.",
            "emotion": "loneliness",
            "strategy": "validation",
        },
    ]
    
    df = pd.DataFrame(conversations)
    
    metadata_path = processed_dir / "conversations.json"
    df.to_json(metadata_path, orient="records", indent=2)
    logger.info(f"Saved {len(df)} conversations to {metadata_path}")
    
    return df


def main() -> None:
    """Download and prepare all datasets."""
    parser = argparse.ArgumentParser(description="Download depression datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["daic_woz", "reddit", "esconv"],
        choices=["daic_woz", "reddit", "esconv", "all"],
        help="Datasets to prepare",
    )
    args = parser.parse_args()
    
    setup_logging()
    output_dir = Path(args.output_dir)
    
    if "all" in args.datasets or "daic_woz" in args.datasets:
        prepare_daic_woz(output_dir)
    
    if "all" in args.datasets or "reddit" in args.datasets:
        prepare_reddit_depression(output_dir)
    
    if "all" in args.datasets or "esconv" in args.datasets:
        prepare_esconv(output_dir)
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()