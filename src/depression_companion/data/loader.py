"""Data loading utilities for various dataset formats."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from loguru import logger

from depression_companion.exceptions import DataLoadError


class DataLoader:
    """Unified data loader supporting multiple formats."""

    SUPPORTED_FORMATS = {"csv", "json", "jsonl", "parquet"}

    @staticmethod
    def load(
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from file into DataFrame.

        Args:
            path: Path to data file.
            format: File format. Auto-detected if None.
            **kwargs: Additional arguments passed to pandas loader.

        Returns:
            DataFrame with loaded data.

        Raises:
            DataLoadError: If file not found or format unsupported.
        """
        path = Path(path)

        if not path.exists():
            raise DataLoadError(f"File not found: {path}")

        # Auto-detect format
        if format is None:
            format = path.suffix.lstrip(".")

        if format not in DataLoader.SUPPORTED_FORMATS:
            raise DataLoadError(
                f"Unsupported format: {format}. "
                f"Supported: {DataLoader.SUPPORTED_FORMATS}"
            )

        logger.info(f"Loading {format} file: {path}")

        try:
            if format == "csv":
                return pd.read_csv(path, **kwargs)
            elif format == "json":
                return pd.read_json(path, **kwargs)
            elif format == "jsonl":
                records = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
                return pd.DataFrame(records)
            elif format == "parquet":
                return pd.read_parquet(path, **kwargs)
        except Exception as e:
            raise DataLoadError(f"Failed to load {path}: {str(e)}")

    @staticmethod
    def save(
        df: pd.DataFrame,
        path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Save DataFrame to file.

        Args:
            df: DataFrame to save.
            path: Output path.
            format: File format. Auto-detected if None.
            **kwargs: Additional arguments passed to pandas saver.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = path.suffix.lstrip(".")

        logger.info(f"Saving {format} file: {path}")

        if format == "csv":
            df.to_csv(path, index=False, **kwargs)
        elif format == "json":
            df.to_json(path, orient="records", **kwargs)
        elif format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")
        elif format == "parquet":
            df.to_parquet(path, index=False, **kwargs)
