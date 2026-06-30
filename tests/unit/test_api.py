"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from depression_companion.api.main import app
from depression_companion.api.schemas import (
    AnalysisResponse,
    ChatResponse,
    HealthCheckResponse,
)


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data


class TestTextAnalysis:
    """Test text analysis endpoint."""
    
    def test_valid_text(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze/text",
            json={"text": "I've been feeling really down and hopeless lately."},
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "scores" in data
        assert data["scores"]["depression_score"] > 0
    
    def test_empty_text(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze/text",
            json={"text": ""},
        )
        assert response.status_code == 422  # Validation error
    
    def test_too_long_text(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze/text",
            json={"text": "x" * 5001},
        )
        assert response.status_code == 422
    
    def test_depressive_text_scores_higher(self, client: TestClient) -> None:
        dep_response = client.post(
            "/api/v1/analyze/text",
            json={"text": "I am so depressed and hopeless. Nothing matters anymore."},
        )
        neutral_response = client.post(
            "/api/v1/analyze/text",
            json={"text": "Today was a nice day. I went for a walk and enjoyed the weather."},
        )
        
        dep_score = dep_response.json()["scores"]["depression_score"]
        neutral_score = neutral_response.json()["scores"]["depression_score"]
        
        assert dep_score > neutral_score


class TestChat:
    """Test chat endpoint."""
    
    def test_valid_chat(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/chat",
            json={"message": "I've been feeling really down lately."},
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "conversation_id" in data
    
    def test_crisis_detection(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/chat",
            json={"message": "I want to kill myself."},
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["is_crisis"] is True
        assert "988" in data.get("crisis_resources", "")


class TestMoodLogging:
    """Test mood logging endpoint."""
    
    def test_log_mood(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/mood/log",
            json={
                "user_id": "test_user",
                "mood_score": 0.7,
                "notes": "Feeling okay today",
            },
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "logged"
        assert "id" in data
    
    def test_invalid_mood_score(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/mood/log",
            json={
                "user_id": "test_user",
                "mood_score": 1.5,  # Invalid: > 1
            },
        )
        assert response.status_code == 422


class TestForecast:
    """Test forecast endpoint."""
    
    def test_get_forecast(self, client: TestClient) -> None:
        response = client.get("/api/v1/forecast/test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert "trajectory" in data
        assert "day_1" in data["trajectory"]
        assert "day_3" in data["trajectory"]
        assert "day_7" in data["trajectory"]
        assert "risk_level" in data