"""
LLM Client for Testing
Supports both mock and real API calls
Author: Mira Mamdoh
"""

import time
import random
from typing import Dict, Any, Optional


class MockLLMClient:
    """Mock LLM client for testing without API costs"""
    
    def __init__(self, model_name: str = "mock-gpt-4"):
        self.model_name = model_name
        self.call_count = 0
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Generate mock response with realistic behavior
        """
        self.call_count += 1
        time.sleep(0.1)  # Simulate API latency
        
        # Mock responses based on prompt content
        response_text = self._generate_mock_response(prompt, temperature)
        
        return {
            "text": response_text,
            "model": self.model_name,
            "temperature": temperature,
            "tokens_used": len(response_text.split()),
            "latency_ms": random.randint(100, 500)
        }
    
    def _generate_mock_response(self, prompt: str, temperature: float) -> str:
        """Generate contextually relevant mock responses"""
        
        # Empty prompt
        if not prompt or prompt.strip() == "":
            return ""
        
        # Very short prompt
        if len(prompt) < 5:
            return "I need more context to provide a helpful answer."
        
        # EDA-related
        if "eda" in prompt.lower() or "electronic" in prompt.lower():
            responses = [
                "EDA (Electronic Design Automation) tools help engineers design and verify electronic circuits and systems. They automate complex design tasks and ensure accuracy.",
                "Electronic Design Automation tools are software applications that assist in designing electronic systems like integrated circuits and printed circuit boards.",
                "EDA tools streamline the process of designing electronic components by providing simulation, verification, and layout capabilities."
            ]
            # Add temperature-based variation
            if temperature > 0.8:
                return random.choice(responses) + " These tools are essential for modern chip design."
            return random.choice(responses)
        
        # AI/GenAI related
        if "ai" in prompt.lower() or "context" in prompt.lower() or "llm" in prompt.lower():
            responses = [
                "Context memory in conversational AI refers to the system's ability to retain and reference previous messages in a conversation. This enables more coherent and contextually relevant responses.",
                "In conversational AI, context memory allows the model to maintain awareness of earlier parts of the conversation, creating a more natural dialogue experience.",
                "Context memory works by storing conversation history and using it to inform subsequent responses, making the AI appear more coherent and human-like."
            ]
            return random.choice(responses)
        
        # Temperature parameter
        if "temperature" in prompt.lower() and ("parameter" in prompt.lower() or "llm" in prompt.lower()):
            responses = [
                "The temperature parameter in LLMs controls randomness in text generation. Lower values (near 0) produce more deterministic outputs, while higher values increase creativity and variation.",
                "Temperature affects the probability distribution over possible next tokens. Higher temperature leads to more diverse, creative responses, while lower temperature produces more focused, deterministic outputs.",
                "In LLM responses, temperature regulates the level of randomness. A temperature of 0 produces the most likely response consistently, while higher values introduce more variability."
            ]
            return random.choice(responses)
        
        # Testing related
        if "test" in prompt.lower():
            responses = [
                "Unit testing focuses on individual components in isolation, while integration testing verifies how multiple components work together. Both are crucial for software quality.",
                "The key difference is scope: unit tests validate single functions or methods, whereas integration tests check interactions between multiple modules or systems.",
                "Unit testing targets small, isolated pieces of code. Integration testing examines how these pieces interact as a cohesive system."
            ]
            return random.choice(responses)
        
        # Regression testing
        if "regression" in prompt.lower():
            responses = [
                "Regression testing in AI systems ensures that new updates or changes don't break existing functionality. It's crucial because AI models can exhibit unexpected behavior changes.",
                "In AI systems, regression testing detects unintended changes in model behavior after updates. This is especially important for non-deterministic systems like LLMs.",
                "Regression testing verifies that AI system updates maintain expected behavior. For LLMs, this involves semantic similarity checks rather than exact output matching."
            ]
            return random.choice(responses)
        
        # Default response with variation based on temperature
        base_responses = [
            f"This is a response to: '{prompt[:50]}...' - The answer involves multiple considerations and contextual factors.",
            f"Regarding '{prompt[:50]}...', there are several aspects to consider in providing a comprehensive answer.",
            f"To address '{prompt[:50]}...', we need to examine the key concepts and their practical implications."
        ]
        
        response = random.choice(base_responses)
        
        # Add temperature-based variation
        if temperature > 0.8:
            response += " Let me elaborate further with additional context and examples."
        
        return response


class LLMClient:
    """
    Real LLM client wrapper (can be extended for OpenAI, Gemini, etc.)
    For now, uses mock for demonstration
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", use_mock: bool = True):
        self.model = model
        self.use_mock = use_mock
        
        if use_mock:
            self.client = MockLLMClient(model_name=model)
        else:
            # In production, initialize real API client here
            # import openai
            # self.client = openai.Client(api_key=api_key)
            raise NotImplementedError("Real API client not implemented. Use use_mock=True")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> Dict[str, Any]:
        """Generate response using configured client"""
        return self.client.generate(prompt, temperature, max_tokens)
    
    def generate_multiple(self, prompt: str, n: int = 10, temperature: float = 0.7) -> list:
        """
        Generate multiple responses for reliability testing
        """
        responses = []
        for i in range(n):
            result = self.generate(prompt, temperature)
            responses.append(result)
        return responses