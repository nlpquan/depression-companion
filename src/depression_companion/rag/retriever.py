"""Vector-based retrieval for CBT knowledge."""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class SimpleRetriever:
    """Simple retriever using keyword matching and TF-IDF-like scoring.
    
    In production, replace with ChromaDB or FAISS for vector search.
    """
    
    def __init__(self, knowledge_base):
        """Initialize retriever.
        
        Args:
            knowledge_base: CBTKnowledgeBase instance.
        """
        self.kb = knowledge_base
        self.documents = knowledge_base.to_documents()
        self._build_index()
        
        logger.info(f"SimpleRetriever initialized with {len(self.documents)} documents")
    
    def _build_index(self) -> None:
        """Build simple word frequency index."""
        self.word_index: dict[str, set[int]] = {}
        
        for doc_id, doc in enumerate(self.documents):
            words = set(doc.lower().split())
            for word in words:
                if word not in self.word_index:
                    self.word_index[word] = set()
                self.word_index[word].add(doc_id)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve relevant documents.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries with 'content' and 'score'.
        """
        query_words = set(query.lower().split())
        
        # Score each document
        scores = np.zeros(len(self.documents))
        
        for word in query_words:
            if word in self.word_index:
                for doc_id in self.word_index[word]:
                    # TF-IDF-like: more weight for rare words
                    idf = np.log(len(self.documents) / len(self.word_index[word]))
                    scores[doc_id] += idf
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self.documents[idx],
                    "protocol": self.kb.protocols[idx],
                    "score": float(scores[idx]),
                })
        
        return results


class ChromaDBRetriever:
    """ChromaDB-based vector retriever for production use.
    
    Requires: pip install chromadb sentence-transformers
    """
    
    def __init__(
        self,
        knowledge_base,
        collection_name: str = "cbt_protocols",
        persist_directory: str = "data/chroma_db",
    ):
        """Initialize ChromaDB retriever.
        
        Args:
            knowledge_base: CBTKnowledgeBase instance.
            collection_name: ChromaDB collection name.
            persist_directory: Directory for persistent storage.
        """
        self.kb = knowledge_base
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Use sentence-transformers for embeddings
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                )
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                )
                self._populate_collection()
                logger.info(f"Created new collection: {collection_name}")
            
            self._has_chromadb = True
            
        except ImportError:
            logger.warning("chromadb not installed. Using SimpleRetriever instead.")
            self._has_chromadb = False
            self.fallback = SimpleRetriever(knowledge_base)
    
    def _populate_collection(self) -> None:
        """Populate ChromaDB collection with CBT protocols."""
        documents = self.kb.to_documents()
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [
            {"title": p["title"], "category": p.get("category", "")}
            for p in self.kb.protocols
        ]
        
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to collection")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve relevant documents using vector search.
        
        Args:
            query: Search query.
            top_k: Number of results.
            
        Returns:
            List of results with content, protocol, and score.
        """
        if not self._has_chromadb:
            return self.fallback.retrieve(query, top_k)
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i] if results["distances"] else 0.0,
            })
        
        return retrieved