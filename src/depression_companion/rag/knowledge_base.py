"""CBT knowledge base management.

Stores and manages structured Cognitive Behavioral Therapy protocols
for retrieval-augmented generation.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger


class CBTKnowledgeBase:
    """Structured knowledge base of CBT protocols and techniques."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize knowledge base.
        
        Args:
            data_path: Path to CBT protocols JSON file.
        """
        self.protocols: list[dict] = []
        
        if data_path and Path(data_path).exists():
            self.load(data_path)
        else:
            self._load_default_protocols()
        
        logger.info(f"CBTKnowledgeBase loaded with {len(self.protocols)} protocols")
    
    def load(self, data_path: Path) -> None:
        """Load protocols from JSON file.
        
        Args:
            data_path: Path to JSON file.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.protocols = json.load(f)
    
    def _load_default_protocols(self) -> None:
        """Load built-in default CBT protocols."""
        self.protocols = [
            {
                "id": "cbt_001",
                "title": "Challenging Negative Thoughts",
                "category": "cognitive",
                "description": "Identify and challenge automatic negative thoughts using evidence-based techniques",
                "steps": [
                    "Identify the negative thought",
                    "Examine the evidence for and against",
                    "Generate a balanced alternative thought",
                    "Rate your belief in the new thought",
                ],
                "suitable_for": ["depression", "anxiety", "negative_thoughts"],
            },
            {
                "id": "cbt_002",
                "title": "Behavioral Activation",
                "category": "behavioral",
                "description": "Increase engagement in positive activities to improve mood",
                "steps": [
                    "Track daily activities and mood",
                    "Identify patterns between activities and mood",
                    "Schedule pleasant activities",
                    "Gradually increase activity levels",
                ],
                "suitable_for": ["depression"],
            },
            {
                "id": "cbt_003",
                "title": "Grounding Techniques for Anxiety",
                "category": "behavioral",
                "description": "Use sensory grounding to manage anxiety and panic attacks",
                "steps": [
                    "5-4-3-2-1 senses exercise",
                    "Deep breathing: 4-7-8 technique",
                    "Progressive muscle relaxation",
                    "Focus on present moment",
                ],
                "suitable_for": ["anxiety", "panic", "ptsd"],
            },
            {
                "id": "cbt_004",
                "title": "Sleep Hygiene Protocol",
                "category": "behavioral",
                "description": "Improve sleep quality through behavioral changes",
                "steps": [
                    "Maintain consistent sleep schedule",
                    "Create relaxing bedtime routine",
                    "Limit screen time before bed",
                    "Avoid caffeine after 2pm",
                ],
                "suitable_for": ["depression", "anxiety", "insomnia"],
            },
        ]
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Simple keyword search over protocols.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of matching protocols.
        """
        query_lower = query.lower()
        scored = []
        
        for protocol in self.protocols:
            # Score based on keyword matches
            score = 0
            
            # Check title
            if any(word in protocol["title"].lower() for word in query_lower.split()):
                score += 3
            
            # Check description
            if any(word in protocol["description"].lower() for word in query_lower.split()):
                score += 2
            
            # Check steps
            for step in protocol.get("steps", []):
                if any(word in step.lower() for word in query_lower.split()):
                    score += 1
            
            # Check suitable_for tags
            for tag in protocol.get("suitable_for", []):
                if tag.replace("_", " ") in query_lower:
                    score += 3
            
            if score > 0:
                scored.append((score, protocol))
        
        # Sort by score and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [protocol for _, protocol in scored[:top_k]]
    
    def get_by_category(self, category: str) -> list[dict]:
        """Get all protocols in a category.
        
        Args:
            category: Category name.
            
        Returns:
            List of protocols in the category.
        """
        return [p for p in self.protocols if p.get("category") == category]
    
    def get_by_suitability(self, condition: str) -> list[dict]:
        """Get protocols suitable for a specific condition.
        
        Args:
            condition: Mental health condition (e.g., 'depression').
            
        Returns:
            List of suitable protocols.
        """
        return [
            p for p in self.protocols
            if condition in p.get("suitable_for", [])
        ]
    
    def to_documents(self) -> list[str]:
        """Convert protocols to text documents for embedding.
        
        Returns:
            List of text representations of each protocol.
        """
        documents = []
        for protocol in self.protocols:
            doc = (
                f"Title: {protocol['title']}\n"
                f"Category: {protocol.get('category', '')}\n"
                f"Description: {protocol['description']}\n"
                f"Steps: {' '.join(protocol.get('steps', []))}\n"
                f"Suitable for: {', '.join(protocol.get('suitable_for', []))}"
            )
            documents.append(doc)
        return documents