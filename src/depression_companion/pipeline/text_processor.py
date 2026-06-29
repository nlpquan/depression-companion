"""Text processing pipeline for feature extraction."""

import time
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer

from depression_companion.config import TextConfig
from depression_companion.exceptions import TextProcessingError, ValidationError
from depression_companion.pipeline.base import BaseProcessor, TextFeatures


class TextProcessor(BaseProcessor[str, TextFeatures]):
    """Extract text features for mental health analysis.

    Supports:
    - Transformer-based embeddings (BERT, RoBERTa)
    - Sentiment features via VADER
    - Linguistic feature extraction
    - Mental health keyword detection
    """

    def __init__(self, config: TextConfig, device: Optional[torch.device] = None):
        """Initialize text processor.

        Args:
            config: Text processing configuration.
            device: Torch device.
        """
        super().__init__(config, device)
        self.max_length = config.max_length
        self.model_name = config.pretrained_model

        # Initialize tokenizer (model loaded lazily for memory efficiency)
        self._tokenizer: Any = None  # type: ignore[assignment]
        self._model = None

    def initialize(self) -> None:
        """Load tokenizer and model."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._is_initialized = True

    def validate_input(self, input_data: str) -> bool:
        """Validate text input.

        Args:
            input_data: Text string.

        Returns:
            True if valid.

        Raises:
            ValidationError: If input is invalid.
        """
        if not isinstance(input_data, str):
            raise ValidationError(f"Expected str, got: {type(input_data)}")

        if not input_data.strip():
            raise ValidationError("Text input cannot be empty")

        if len(input_data) > 10000:
            raise ValidationError(f"Text too long: {len(input_data)} chars (max 10000)")

        return True

    def process(self, input_data: str) -> TextFeatures:
        """Extract text features from input.

        Args:
            input_data: Text string.

        Returns:
            TextFeatures with extracted features.

        Raises:
            TextProcessingError: If processing fails.
        """
        start_time = time.time()

        try:
            self.validate_input(input_data)

            logger.info(f"Processing text: {len(input_data)} chars")

            # Tokenize
            tokens = self._tokenize(input_data)

            # Extract linguistic features
            linguistic_features = self._extract_linguistic_features(input_data)

            # Extract sentiment
            sentiment_features = self._extract_sentiment(input_data)

            # Extract mental health keywords
            keyword_features = self._extract_keywords(input_data)

            # Combine into embedding
            embeddings = np.concatenate(
                [
                    linguistic_features,
                    sentiment_features,
                    keyword_features,
                ]
            )

            processing_time = time.time() - start_time

            return TextFeatures(
                embeddings=embeddings,
                tokens=tokens,
                attention_mask=None,
                metadata={
                    "char_count": len(input_data),
                    "word_count": len(input_data.split()),
                    "processing_time": processing_time,
                },
            )

        except ValidationError:
            raise
        except Exception as e:
            raise TextProcessingError(f"Text processing failed: {str(e)}")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using the configured tokenizer.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        if not self._is_initialized:
            self.initialize()

        if self._tokenizer is None:
            # Fallback: simple whitespace tokenization
            return text.lower().split()

        tokens = self._tokenizer.tokenize(text)
        return tokens[: self.max_length]

    def _extract_linguistic_features(self, text: str) -> np.ndarray:
        """Extract linguistic features from text.

        Includes:
        - Word count
        - Sentence count
        - Average word length
        - Type-token ratio (vocabulary richness)
        - First/second/third person pronoun ratios

        Args:
            text: Input text.

        Returns:
            Linguistic feature vector.
        """
        words = text.lower().split()
        sentences = [
            s.strip()
            for s in text.replace("!", ".").replace("?", ".").split(".")
            if s.strip()
        ]

        if not words:
            return np.zeros(10)

        n_words = len(words)
        n_sentences = max(len(sentences), 1)
        n_unique = len(set(words))

        # Word-level features
        avg_word_length = np.mean([len(w) for w in words])
        type_token_ratio = n_unique / n_words

        # Sentence-level features
        words_per_sentence = n_words / n_sentences

        # Pronoun usage (highly relevant for mental health)
        first_person = sum(1 for w in words if w in {"i", "me", "my", "mine", "myself"})
        second_person = sum(
            1 for w in words if w in {"you", "your", "yours", "yourself"}
        )
        third_person = sum(
            1
            for w in words
            if w in {"he", "she", "they", "him", "her", "them", "his", "hers", "theirs"}
        )

        first_person_ratio = first_person / n_words
        second_person_ratio = second_person / n_words
        third_person_ratio = third_person / n_words

        # Negation words (important for sentiment)
        negation_words = sum(
            1 for w in words if w in {"not", "no", "never", "neither", "nor", "nothing"}
        )
        negation_ratio = negation_words / n_words

        return np.array(
            [
                n_words,
                n_sentences,
                avg_word_length,
                type_token_ratio,
                words_per_sentence,
                first_person_ratio,
                second_person_ratio,
                third_person_ratio,
                negation_ratio,
                n_unique,
            ],
            dtype=np.float32,
        )

    def _extract_sentiment(self, text: str) -> np.ndarray:
        """Extract sentiment features using VADER.

        Args:
            text: Input text.

        Returns:
            Sentiment feature vector [neg, neu, pos, compound].
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)

            return np.array(
                [
                    scores["neg"],
                    scores["neu"],
                    scores["pos"],
                    scores["compound"],
                ],
                dtype=np.float32,
            )

        except ImportError:
            logger.warning("vaderSentiment not installed. Using zero sentiment.")
            return np.zeros(4, dtype=np.float32)

    def _extract_keywords(self, text: str) -> np.ndarray:
        """Extract mental health related keyword features.

        Args:
            text: Input text.

        Returns:
            Keyword feature vector.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Mental health keyword categories (single words)
        depression_keywords = {
            "sad",
            "depressed",
            "hopeless",
            "worthless",
            "empty",
            "lonely",
            "miserable",
            "unhappy",
            "crying",
            "tears",
        }
        anxiety_keywords = {
            "anxious",
            "worried",
            "nervous",
            "panic",
            "fear",
            "stress",
            "stressed",
            "overwhelmed",
            "restless",
            "tense",
        }
        sleep_keywords = {
            "insomnia",
            "sleep",
            "tired",
            "fatigue",
            "exhausted",
            "nightmare",
            "awake",
            "restless",
        }

        # Count single-word matches
        depression_count = len(words & depression_keywords)
        anxiety_count = len(words & anxiety_keywords)
        sleep_count = len(words & sleep_keywords)

        # ⭐ CRISIS DETECTION - Check the FULL TEXT for phrases
        crisis_phrases = [
            "suicide",
            "kill myself",
            "end my life",
            "self-harm",
            "hurt myself",
            "want to die",
            "can't take this",
        ]
        crisis_count = sum(1 for phrase in crisis_phrases if phrase in text_lower)

        crisis_phrase_present = 1 if crisis_count > 0 else 0

        return np.array(
            [
                depression_count,
                anxiety_count,
                sleep_count,
                crisis_count,
                crisis_phrase_present,  # ✅ 1 if any crisis phrase found
            ],
            dtype=np.float32,
        )
