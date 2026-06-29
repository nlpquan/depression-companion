# setup.py (at root C:\AI-Projects\depression-companion\setup.py)
from setuptools import setup, find_packages

setup(
    name="depression-companion",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.35.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.4.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "soundfile>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "mypy>=1.7.0",
            "ruff>=0.1.0",
            "pre-commit>=3.5.0",
            "coverage>=7.3.0",
        ],
    },
)