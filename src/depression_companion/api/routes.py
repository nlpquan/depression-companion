"""FastAPI route definitions."""

import time
import datetime
import uuid
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
    """Analyze text for depression indicators.
    
    Args:
        request: Text analysis request.
        
    Returns:
        Analysis response with scores.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        # This would call the pipeline in production
        # For now, return mock analysis
        logger.info(f"Analyzing text: {len(request.text)} chars")
        
        # Mock scores based on text length and content
        text_lower = request.text.lower()
        
        # Simple heuristic for demo
        depression_indicators = sum(
            1 for word in ["sad", "depressed", "hopeless", "tired", "worthless"]
            if word in text_lower
        )
        
        depression_score = min(0.3 + depression_indicators * 0.15, 0.95)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            id=analysis_id,
            timestamp=datetime.now(timezone.utc),
            scores=DepressionScores(
                depression_score=depression_score,
                anxiety_score=depression_score * 0.8,
                mood_score=1.0 - depression_score,
                confidence=0.75,
            ),
            processing_time_ms=processing_time,
            features_extracted={
                "word_count": len(request.text.split()),
                "char_count": len(request.text),
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
        text_scores = await _mock_text_analysis(request.text)
        
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


# ---- Mock analysis functions ----

async def _mock_text_analysis(text: str) -> dict:
    """Mock text analysis for development."""
    text_lower = text.lower()
    depression_words = sum(
        1 for word in ["sad", "depressed", "hopeless", "tired", "worthless", "down"]
        if word in text_lower
    )
    return {
        "depression": min(0.3 + depression_words * 0.12, 0.95),
        "anxiety": min(0.2 + depression_words * 0.1, 0.9),
    }


async def _mock_audio_analysis(file: UploadFile) -> dict:
    """Mock audio analysis for development."""
    return {
        "depression": 0.4,
        "anxiety": 0.35,
    }