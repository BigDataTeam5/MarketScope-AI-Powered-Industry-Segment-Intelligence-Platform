"""
LiteLLM Service Module for MarketScope
Initializes LiteLLM with configuration from Config class
"""
import os
import litellm
from typing import List, Dict, Any, Optional, Callable
from langchain.tools import BaseTool
from .config import Config

class LiteLLMService:
    """Service class to handle LiteLLM configuration and operations"""
    
    def __init__(self):
        """Initialize LiteLLM service with default settings"""
        self.current_model = "gpt-4o"  # Hardcoded to gpt-4o
        self.temperature = Config.DEFAULT_TEMPERATURE
        self._configure_api_keys()
    
    def _configure_api_keys(self):
        """Configure API keys for all supported providers"""
        if Config.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        if Config.ANTHROPIC_API_KEY:
            os.environ["ANTHROPIC_API_KEY"] = Config.ANTHROPIC_API_KEY
        if Config.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        if Config.GROK_API_KEY:
            os.environ["XAI_API_KEY"] = Config.GROK_API_KEY
        if Config.DEEPSEEK_API_KEY:
            os.environ["DEEPSEEK_API_KEY"] = Config.DEEPSEEK_API_KEY
    
    def get_model(self, model_name: str = None) -> str:
        """Get the appropriate LiteLLM model name for the selected model"""
        return "gpt-4o"  # Always return gpt-4o regardless of input
    
    def __call__(self, messages, model=None, temperature=None, **kwargs):
        """Call the LLM service with configured settings"""
        model = model or self.current_model
        temperature = temperature or self.temperature
        model_name = self.get_model(model)

        try:
            response = litellm.completion(
                model=model_name,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            # Properly extract message content from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message
                else:
                    return str(response.choices[0])
            return str(response)
        except Exception as e:
            raise Exception(f"LiteLLM call failed: {str(e)}")

# Create a singleton instance
litellm_service = LiteLLMService()