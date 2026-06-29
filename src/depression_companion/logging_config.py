"""Logging configuration using Loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from depression_companion.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure Loguru logger with application settings.

    Args:
        config: Logging configuration from AppConfig.
    """
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=config.level,
        colorize=True,
    )

    # Add file handler
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level=config.level,
        rotation=config.rotation,
        retention=config.retention,
        compression="zip",
    )

    logger.info(f"Logging configured: level={config.level}, file={config.file}")


def get_logger(name: str):
    """Get a logger instance for a specific module.

    Args:
        name: Module name (usually __name__).

    Returns:
        Logger bound with module context.
    """
    return logger.bind(name=name)
