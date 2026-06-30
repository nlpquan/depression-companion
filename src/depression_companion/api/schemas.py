"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---- Request Schemas ----

class TextAnalysisRequest(BaseModel):
    """Request for text-only analysis."""
    text: str = Field(..., min_length=1, max_length=5000, description="User text input")
    user_id: Optional[str] = Field(None, description="Anonymous user identifier")


class AudioAnalysisRequest(BaseModel):
    """Request for audio analysis (metadata only, file uploaded separately)."""
    user_id: Optional[str] = None
    duration: Optional[float] = Field(None, description="Expected audio duration in seconds")


class MultimodalAnalysisRequest(BaseModel):
    """Request for combined audio + text analysis."""
    text: str = Field(..., min_length=1, max_length=5000)
    user_id: Optional[str] = None
    include_forecast: bool = Field(False, description="Include mood forecast")
    include_attention: bool = Field(False, description="Include attention visualization data")


class ChatRequest(BaseModel):
    """Request for RAG-powered chat."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


class MoodLogRequest(BaseModel):
    """Request to log mood data."""
    user_id: str
    mood_score: float = Field(..., ge=0, le=1, description="Mood score (0-1)")
    notes: Optional[str] = None
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    anxiety_level: Optional[float] = Field(None, ge=0, le=1)


# ---- Response Schemas ----

class DepressionScores(BaseModel):
    """Depression analysis scores."""
    depression_score: float = Field(..., ge=0, le=1)
    anxiety_score: float = Field(..., ge=0, le=1)
    mood_score: float = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    id: str
    timestamp: datetime
    scores: DepressionScores
    processing_time_ms: float
    features_extracted: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """RAG chat response."""
    message: str
    sources: list[dict] = Field(default_factory=list)
    is_crisis: bool = False
    crisis_resources: Optional[str] = None
    conversation_id: str


class ForecastResponse(BaseModel):
    """Mood forecast response."""
    trajectory: dict[str, float]  # day_1, day_3, day_7
    lower_bound: dict[str, float]
    upper_bound: dict[str, float]
    risk_level: str
    active_warnings: list[dict]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float