"""Tests for knowledge retrieval."""

import pytest

from depression_companion.rag.knowledge_base import CBTKnowledgeBase
from depression_companion.rag.retriever import SimpleRetriever


class TestCBTKnowledgeBase:
    """Test CBT knowledge base."""
    
    @pytest.fixture
    def kb(self) -> CBTKnowledgeBase:
        return CBTKnowledgeBase()
    
    def test_initialization(self, kb: CBTKnowledgeBase) -> None:
        """Test knowledge base creation."""
        assert len(kb.protocols) > 0
    
    def test_search(self, kb: CBTKnowledgeBase) -> None:
        """Test keyword search."""
        results = kb.search("negative thoughts", top_k=3)
        assert len(results) > 0
        assert any("thought" in r["title"].lower() for r in results)
    
    def test_get_by_category(self, kb: CBTKnowledgeBase) -> None:
        """Test category filtering."""
        results = kb.get_by_category("behavioral")
        assert len(results) > 0
        assert all(r["category"] == "behavioral" for r in results)
    
    def test_get_by_suitability(self, kb: CBTKnowledgeBase) -> None:
        """Test suitability filtering."""
        results = kb.get_by_suitability("depression")
        assert len(results) > 0
        assert all("depression" in r["suitable_for"] for r in results)
    
    def test_to_documents(self, kb: CBTKnowledgeBase) -> None:
        """Test document conversion."""
        docs = kb.to_documents()
        assert len(docs) == len(kb.protocols)
        assert all(isinstance(d, str) for d in docs)


class TestSimpleRetriever:
    """Test simple retriever."""
    
    @pytest.fixture
    def retriever(self) -> SimpleRetriever:
        kb = CBTKnowledgeBase()
        return SimpleRetriever(kb)
    
    def test_initialization(self, retriever: SimpleRetriever) -> None:
        """Test retriever creation."""
        assert len(retriever.documents) > 0
    
    def test_retrieve(self, retriever: SimpleRetriever) -> None:
        """Test document retrieval."""
        results = retriever.retrieve("I can't sleep at night", top_k=3)
        
        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]
        assert results[0]["score"] > 0
    
    def test_retrieve_empty_query(self, retriever: SimpleRetriever) -> None:
        """Test retrieval with empty query."""
        results = retriever.retrieve("", top_k=3)
        assert len(results) == 0
    
    def test_retrieve_relevance(self, retriever: SimpleRetriever) -> None:
        """Test that results are relevant to query."""
        results = retriever.retrieve("anxiety and panic attacks", top_k=3)
        
        # Should find anxiety-related content
        contents = " ".join([r["content"] for r in results]).lower()
        assert any(word in contents for word in ["anxiety", "panic", "grounding", "exposure"])