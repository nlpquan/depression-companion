"""RAG response generator for mental health support.

Combines retrieved CBT knowledge with LLM generation to produce
clinically-informed, empathetic responses.
"""

from typing import Optional

from loguru import logger


class RAGGenerator:
    """Generate responses using retrieval-augmented generation.
    
    Retrieves relevant CBT protocols and uses them as context for generation.
    """
    
    def __init__(
        self,
        retriever,
        model_name: str = "mistral",
        safety_classifier=None,
    ):
        """Initialize RAG generator.
        
        Args:
            retriever: Document retriever instance.
            model_name: Name of the LLM to use.
            safety_classifier: Safety classifier for crisis detection.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.safety_classifier = safety_classifier
        
        logger.info(f"RAGGenerator initialized with model={model_name}")
    
    def generate(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 3,
    ) -> dict:
        """Generate a response using RAG.
        
        Args:
            user_message: User's message.
            conversation_history: Previous conversation turns.
            top_k: Number of documents to retrieve.
            
        Returns:
            Dictionary with response, sources, and safety flags.
        """
        # Step 1: Safety check
        safety_result = self._check_safety(user_message)
        
        if safety_result.get("is_crisis", False):
            return {
                "response": self._get_crisis_response(),
                "sources": [],
                "is_crisis": True,
                "safety_check": safety_result,
            }
        
        # Step 2: Retrieve relevant CBT knowledge
        retrieved_docs = self.retriever.retrieve(user_message, top_k=top_k)
        
        # Step 3: Build prompt with context
        prompt = self._build_prompt(user_message, retrieved_docs, conversation_history)
        
        # Step 4: Generate response (placeholder for actual LLM call)
        response = self._generate_response(prompt, retrieved_docs)
        
        return {
            "response": response,
            "sources": [
                {
                    "title": doc.get("protocol", doc.get("metadata", {})).get("title", "Unknown"),
                    "content": doc["content"][:200] + "...",
                    "score": doc["score"],
                }
                for doc in retrieved_docs
            ],
            "is_crisis": False,
            "safety_check": safety_result,
        }
    
    def _check_safety(self, message: str) -> dict:
        """Check message for safety concerns.
        
        Args:
            message: User message.
            
        Returns:
            Safety check results.
        """
        if self.safety_classifier:
            return self.safety_classifier.check(message)
        
        # Simple keyword-based check
        crisis_keywords = [
            "suicide", "kill myself", "end my life", "self-harm",
            "hurt myself", "want to die", "better off dead",
        ]
        
        message_lower = message.lower()
        is_crisis = any(keyword in message_lower for keyword in crisis_keywords)
        
        return {
            "is_crisis": is_crisis,
            "risk_level": "high" if is_crisis else "low",
            "matched_keywords": [kw for kw in crisis_keywords if kw in message_lower],
        }
    
    def _get_crisis_response(self) -> str:
        """Get crisis response message.
        
        Returns:
            Crisis support message with resources.
        """
        return (
            "I'm concerned about what you're sharing. Your safety is important. "
            "Please reach out to a crisis support service:\n\n"
            "• National Suicide Prevention Lifeline: 988 or 1-800-273-8255\n"
            "• Crisis Text Line: Text HOME to 741741\n"
            "• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "If you're in immediate danger, please call emergency services (911 in the US) "
            "or go to your nearest emergency room. You don't have to go through this alone."
        )
    
    def _build_prompt(
        self,
        user_message: str,
        retrieved_docs: list[dict],
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Build prompt with retrieved context.
        
        Args:
            user_message: User message.
            retrieved_docs: Retrieved documents.
            conversation_history: Previous turns.
            
        Returns:
            Formatted prompt string.
        """
        # Format retrieved context
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            protocol = doc.get("protocol", {})
            context_parts.append(
                f"[Protocol {i+1}] {protocol.get('title', '')}\n"
                f"{protocol.get('description', '')}\n"
                f"Steps: {' → '.join(protocol.get('steps', []))}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build conversation history
        history_str = ""
        if conversation_history:
            for turn in conversation_history[-5:]:  # Last 5 turns
                history_str += f"User: {turn.get('user', '')}\n"
                history_str += f"Assistant: {turn.get('assistant', '')}\n\n"
        
        # System prompt
        system_prompt = (
            "You are a supportive mental health companion trained in CBT techniques. "
            "Use the provided CBT protocols to inform your responses. "
            "Be empathetic, non-judgmental, and encouraging. "
            "Never diagnose or prescribe. "
            "If someone expresses suicidal thoughts, direct them to crisis resources immediately."
        )
        
        prompt = (
            f"{system_prompt}\n\n"
            f"Available CBT Protocols:\n{context}\n\n"
            f"Conversation History:\n{history_str}\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )
        
        return prompt
    
    def _generate_response(
        self,
        prompt: str,
        retrieved_docs: list[dict],
    ) -> str:
        """Generate response using LLM.
        
        This is a placeholder. In production, this would call Mistral-7B or similar.
        
        Args:
            prompt: Formatted prompt.
            retrieved_docs: Retrieved documents for context.
            
        Returns:
            Generated response.
        """
        # Placeholder: Use template-based responses from retrieved protocols
        if retrieved_docs:
            top_protocol = retrieved_docs[0].get("protocol", {})
            technique = top_protocol.get("title", "active listening")
            
            templates = {
                "Cognitive Restructuring": (
                    "I hear you expressing some challenging thoughts. Let's try to "
                    "examine them together. What evidence do you see that supports "
                    "this thought? And what might be some evidence against it?"
                ),
                "Behavioral Activation": (
                    "When we're feeling low, even small activities can make a "
                    "difference. Is there one tiny thing you used to enjoy that "
                    "you could try today, even just for 5 minutes?"
                ),
                "Mindfulness and Grounding": (
                    "It sounds like you're feeling overwhelmed right now. Let's "
                    "try a quick grounding exercise together. Can you name 3 things "
                    "you can see around you right now?"
                ),
                "Sleep Hygiene Protocol": (
                    "Sleep difficulties often go hand-in-hand with how we're "
                    "feeling emotionally. Would you be open to exploring some "
                    "small changes to your evening routine?"
                ),
            }
            
            for key, response in templates.items():
                if key in technique:
                    return response
        
        # Default empathetic response
        return (
            "Thank you for sharing that with me. It takes courage to express "
            "these feelings. Can you tell me more about what's been going on?"
        )