"""File and path utility functions with Windows compatibility."""

from pathlib import Path
from typing import Optional, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root.
    """
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (pyproject.toml)")


def resolve_path(path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    """Resolve a path relative to project root or another directory.

    Args:
        path: Path to resolve.
        relative_to: Base directory. If None, uses project root.

    Returns:
        Resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute():
        return path

    if relative_to is None:
        relative_to = get_project_root()

    return (relative_to / path).resolve()


def safe_filename(filename: str) -> str:
    """Create a safe filename for all operating systems.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename.
    """
    # Replace characters that are problematic on Windows
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    return filename
