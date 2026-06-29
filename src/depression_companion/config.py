"""Configuration management using Pydantic and YAML."""

from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---- Model Configs ----

class AudioConfig(BaseModel):
    """Audio processing configuration."""
    sample_rate: int = 16000
    max_duration: float = 30.0
    feature_type: str = "wav2vec2"
    pretrained_model: str = "facebook/wav2vec2-base-960h"


class TextConfig(BaseModel):
    """Text processing configuration."""
    max_length: int = 512
    feature_type: str = "bert"
    pretrained_model: str = "bert-base-uncased"


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 16
    learning_rate: float = 2.0e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0


class EvaluationConfig(BaseModel):
    """Evaluation settings."""
    metrics: List[str] = ["accuracy", "f1", "auc", "mae", "rmse"]
    cv_folds: int = 5
    test_split: float = 0.2
    random_seed: int = 42


class ModelConfig(BaseModel):
    """Complete model configuration."""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ---- Data Config ----

class DatasetConfig(BaseModel):
    """Individual dataset configuration."""
    path: str
    format: str = "csv"


class DataConfig(BaseModel):
    """Data paths and sources."""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)


# ---- Logging Config ----

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "10 MB"
    retention: str = "30 days"
    file: str = "logs/depression_companion.log"


# ---- Monitoring Config ----

class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""
    tracking_uri: str = "mlruns"
    experiment_name: str = "depression_companion"


class WandbConfig(BaseModel):
    """Weights & Biases configuration."""
    project: str = "depression-companion"
    entity: Optional[str] = None
    enabled: bool = False


class MonitoringConfig(BaseModel):
    """Experiment tracking configuration."""
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


# ---- API Config ----

class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000"]
    rate_limit: int = 100


# ---- Security Config ----

class SecurityConfig(BaseModel):
    """Security and data handling configuration."""
    max_file_size: str = "10MB"
    allowed_extensions: List[str] = [".wav", ".mp3", ".flac"]
    data_retention_days: int = 30


# ---- Main Config ----

class AppConfig(BaseModel):
    """Root application configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


class Settings(BaseSettings):
    """Settings loaded from YAML and environment variables."""
    model_config = SettingsConfigDict(
        env_prefix="DEPRESSION_COMPANION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )
    
    config_path: Path = Path("config/config.yaml")
    environment: str = "development"
    debug: bool = False
    
    # Override individual settings via env vars (optional)
    data_raw_path: Optional[str] = None
    api_port: Optional[int] = None
    logging_level: Optional[str] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default.
        
    Returns:
        Validated AppConfig object.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
        ValidationError: If config schema is invalid.
    """
    if config_path is None:
        config_path = Path("config/config.yaml")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    return AppConfig(**raw_config)


def load_settings() -> Settings:
    """Load settings from environment and .env file."""
    return Settings()