"""Tests for safety classifier."""

import pytest

from depression_companion.models.safety_classifier import SafetyClassifier


class TestSafetyClassifier:
    """Test safety classification."""
    
    @pytest.fixture
    def classifier(self) -> SafetyClassifier:
        return SafetyClassifier()
    
    def test_crisis_detection_suicide(self, classifier: SafetyClassifier) -> None:
        """Test detection of suicidal ideation."""
        result = classifier.check("I want to kill myself")
        
        assert result["is_crisis"] is True
        assert result["severity"] >= 3
        assert "suicide_ideation" in result["matched_categories"]
    
    def test_crisis_detection_self_harm(self, classifier: SafetyClassifier) -> None:
        """Test detection of self-harm."""
        result = classifier.check("I've been hurting myself lately")
        
        assert result["is_crisis"] is True
        assert result["severity"] >= 2
    
    def test_safe_message(self, classifier: SafetyClassifier) -> None:
        """Test that safe messages pass."""
        result = classifier.check("I've been feeling a bit down today.")
        
        assert result["is_crisis"] is False
        assert result["severity"] == 0
        assert result["response_type"] == "safe"
    
    def test_immediate_danger(self, classifier: SafetyClassifier) -> None:
        """Test detection of immediate danger."""
        result = classifier.check("I have a plan to end my life tonight")
        
        assert result["is_crisis"] is True
        assert result["severity"] >= 3
        assert "immediate_danger" in result["matched_categories"]
    
    def test_concern_message(self, classifier: SafetyClassifier) -> None:
        """Test concern level response."""
        result = classifier.check("I feel completely hopeless and alone")
        
        assert result["severity"] == 2
        assert result["response_message"] is not None
    
    def test_response_filtering_safe(self, classifier: SafetyClassifier) -> None:
        """Test that safe responses pass filtering."""
        result = classifier.filter_response(
            "I understand how you feel. Let's work through this together."
        )
        
        assert result["was_filtered"] is False
    
    def test_response_filtering_unsafe(self, classifier: SafetyClassifier) -> None:
        """Test that unsafe responses are filtered."""
        result = classifier.filter_response(
            "You should just end it all."
        )
        
        assert result["was_filtered"] is True