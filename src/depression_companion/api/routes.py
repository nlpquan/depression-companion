"""FastAPI route definitions."""

import time
import datetime
import uuid
import os
import requests
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from loguru import logger

from depression_companion.api.schemas import (
    AnalysisResponse,
    AudioAnalysisRequest,
    ChatRequest,
    ChatResponse,
    DepressionScores,
    ForecastResponse,
    HealthCheckResponse,
    MoodLogRequest,
    MultimodalAnalysisRequest,
    TextAnalysisRequest,
)

router = APIRouter()

# Store active WebSocket connections
active_connections: dict[str, WebSocket] = {}

# HuggingFace Inference API (free tier: 30K requests/month)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # GOOD - reads from Railway env var
SENTIMENT_API = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"

@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=True,
        uptime_seconds=time.time(),
    )


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest) -> AnalysisResponse:
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Analyzing text: {len(request.text)} chars")
        
        # Real ML analysis with fallback
        scores = await _real_text_analysis(request.text)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            id=analysis_id,
            timestamp=datetime.now(timezone.utc),
            scores=DepressionScores(
                depression_score=scores["depression"],
                anxiety_score=scores["anxiety"],
                mood_score=scores["mood"],
                confidence=scores.get("confidence", 0.75),
            ),
            processing_time_ms=processing_time,
            features_extracted={
                "word_count": len(request.text.split()),
                "char_count": len(request.text),
                "model": scores.get("model", "heuristic"),
            },
            warnings=[],
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    user_id: Optional[str] = None,
) -> AnalysisResponse:
    """Analyze audio for depression indicators.
    
    Args:
        file: Uploaded audio file.
        user_id: Optional user identifier.
        
    Returns:
        Analysis response.
    """
    analysis_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file type
        if file.content_type not in [
            "audio/wav", "audio/mpeg", "audio/flac", "audio/x-wav"
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file.content_type}",
            )
        
        # Read file
        audio_bytes = await file.read()
        logger.info(f"Received audio: {len(audio_bytes)} bytes")
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            id=analysis_id,
            timestamp=datetime.now(timezone.utc),
            scores=DepressionScores(
                depression_score=0.45,
                anxiety_score=0.35,
                mood_score=0.55,
                confidence=0.65,
            ),
            processing_time_ms=processing_time,
            features_extracted={
                "file_size_bytes": len(audio_bytes),
                "format": file.content_type,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/multimodal", response_model=AnalysisResponse)
async def analyze_multimodal(
    request: MultimodalAnalysisRequest,
    file: Optional[UploadFile] = File(None),
) -> AnalysisResponse:
    """Analyze both text and audio for comprehensive assessment.
    
    Args:
        request: Multimodal analysis request.
        file: Optional audio file.
        
    Returns:
        Analysis response.
    """
    analysis_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Process both modalities
        text_scores = await _real_text_analysis(request.text)
        
        audio_scores = None
        if file:
            audio_scores = await _mock_audio_analysis(file)
        
        # Fusion
        if audio_scores:
            depression_score = 0.6 * text_scores["depression"] + 0.4 * audio_scores["depression"]
            confidence = 0.85
        else:
            depression_score = text_scores["depression"]
            confidence = 0.75
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            id=analysis_id,
            timestamp=datetime.now(timezone.utc),
            scores=DepressionScores(
                depression_score=depression_score,
                anxiety_score=depression_score * 0.85,
                mood_score=1.0 - depression_score,
                confidence=confidence,
            ),
            processing_time_ms=processing_time,
            features_extracted={
                "text_analyzed": True,
                "audio_analyzed": file is not None,
                "fusion_method": "weighted_average" if file else "text_only",
            },
        )
        
    except Exception as e:
        logger.error(f"Multimodal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """RAG-powered mental health chat.
    
    Args:
        request: Chat request with user message.
        
    Returns:
        Chat response with CBT-informed reply.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        # In production, this calls the RAG system
        from depression_companion.models.safety_classifier import SafetyClassifier
        
        safety = SafetyClassifier()
        safety_result = safety.check(request.message)
        
        if safety_result["is_crisis"]:
            return ChatResponse(
                message=safety_result["response_message"],
                is_crisis=True,
                crisis_resources=(
                    "National Suicide Prevention Lifeline: 988\n"
                    "Crisis Text Line: Text HOME to 741741"
                ),
                conversation_id=conversation_id,
            )
        
        # Mock RAG response
        response_templates = [
            "I hear you. Can you tell me more about what's been going on?",
            "That sounds really challenging. How long have you been feeling this way?",
            "Thank you for sharing. Let's explore some coping strategies together.",
            "I understand this is difficult. Have you noticed any patterns in these feelings?",
        ]
        
        import random
        response = random.choice(response_templates)
        
        return ChatResponse(
            message=response,
            sources=[
                {"title": "Cognitive Restructuring", "relevance": 0.85},
                {"title": "Behavioral Activation", "relevance": 0.72},
            ],
            conversation_id=conversation_id,
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mood/log")
async def log_mood(request: MoodLogRequest) -> dict:
    """Log a mood entry.
    
    Args:
        request: Mood log request.
        
    Returns:
        Confirmation with entry ID.
    """
    entry_id = str(uuid.uuid4())
    
    logger.info(f"Mood logged: user={request.user_id}, score={request.mood_score}")
    
    return {
        "id": entry_id,
        "status": "logged",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/forecast/{user_id}", response_model=ForecastResponse)
async def get_forecast(user_id: str) -> ForecastResponse:
    """Get mood forecast for a user.
    
    Args:
        user_id: User identifier.
        
    Returns:
        Forecast with trajectory and risk assessment.
    """
    # Mock forecast
    return ForecastResponse(
        trajectory={"day_1": 0.62, "day_3": 0.58, "day_7": 0.55},
        lower_bound={"day_1": 0.55, "day_3": 0.48, "day_7": 0.40},
        upper_bound={"day_1": 0.69, "day_3": 0.68, "day_7": 0.70},
        risk_level="low",
        active_warnings=[],
    )


# ---- WebSocket ----

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time analysis updates.
    
    Args:
        websocket: WebSocket connection.
        user_id: User identifier.
    """
    await websocket.accept()
    active_connections[user_id] = websocket
    
    logger.info(f"WebSocket connected: {user_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Process real-time data
            if data.get("type") == "audio_chunk":
                # Process audio chunk in real-time
                await websocket.send_json({
                    "type": "interim_result",
                    "energy": 0.6,
                    "speech_detected": True,
                })
            elif data.get("type") == "text_input":
                # Real-time text analysis
                await websocket.send_json({
                    "type": "sentiment_update",
                    "sentiment": "neutral",
                    "confidence": 0.8,
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_id}")
        active_connections.pop(user_id, None)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.pop(user_id, None)


# ---- Real analysis functions ----

async def _real_text_analysis(text: str) -> dict:
    """Real sentiment analysis via HuggingFace Inference API.
    
    Falls back to heuristic if API is unavailable.
    """
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set, using mock analysis")
        return _heuristic_text_analysis(text)
    
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": text[:500]}
        
        response = requests.post(SENTIMENT_API, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        # Parse sentiment scores: labels are LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        sentiment_map = {}
        for item in result[0]:
            label = item["label"].lower()
            sentiment_map[label] = item["score"]
        
        negative_score = sentiment_map.get("negative", sentiment_map.get("label_0", 0.0))
        positive_score = sentiment_map.get("positive", sentiment_map.get("label_2", 0.0))
        
        # Map to depression/anxiety/mood
        depression_score = round(negative_score, 2)
        anxiety_score = round(negative_score * 0.85, 2)
        mood_score = round(positive_score, 2)
        
        logger.info(f"Real analysis: depression={depression_score}, mood={mood_score}")
        
        return {
            "depression": depression_score,
            "anxiety": anxiety_score,
            "mood": mood_score,
            "confidence": max(sentiment_map.values()) if sentiment_map else 0.5,
            "model": "roberta-sentiment",
        }
        
    except Exception as e:
        logger.warning(f"HF API failed, falling back to heuristic: {e}")
        return _heuristic_text_analysis(text)


def _heuristic_text_analysis(text: str) -> dict:
    """Keyword-based fallback analysis."""
    text_lower = text.lower()
    
    depression_keywords = {
        "hopeless": 0.25, "worthless": 0.25, "suicidal": 0.35,
        "tired": 0.1, "exhausted": 0.15, "lonely": 0.15,
        "anxious": 0.15, "overwhelmed": 0.15, "numb": 0.2,
        "crying": 0.15, "struggling": 0.1, "pain": 0.15,
        "dark": 0.1, "empty": 0.15, "sad": 0.1,
        "depressed": 0.2, "down": 0.1, "terrible": 0.15,
    }
    
    depression_score = 0.1
    for word, weight in depression_keywords.items():
        if word in text_lower:
            depression_score += weight
    
    depression_score = min(depression_score, 0.95)
    
    return {
        "depression": round(depression_score, 2),
        "anxiety": round(depression_score * 0.85, 2),
        "mood": round(1.0 - depression_score, 2),
    }


async def _mock_audio_analysis(file: UploadFile) -> dict:
    """Mock audio analysis for development."""
    return {
        "depression": 0.4,
        "anxiety": 0.35,
    }