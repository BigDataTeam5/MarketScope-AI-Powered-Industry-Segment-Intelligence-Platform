"""
LiteLLM Service Module for MarketScope
Initializes LiteLLM with configuration from Config class
"""
import os
import litellm
from typing import List, Dict, Any

# Import the Config class from the same directory
from .config import Config

class LiteLLMService:
    """Service class to handle LiteLLM configuration and operations"""
    
    def __init__(self):
        """Initialize LiteLLM with configuration from Config class"""
        self.initialize()
    
    def initialize(self):
        """Initialize LiteLLM with settings from Config"""
        # Get LiteLLM parameters
        params = Config.get_litellm_params()
        
        # Set cache configuration
        litellm.cache = params["cache"]
        if litellm.cache:
            litellm.cache_params = params["cache_params"]
        
        # Set API keys in environment variables for LiteLLM to use
        for provider, api_key in params["api_key"].items():
            if api_key:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
    
    async def generate_completion(self, 
                           model_name: str, 
                           messages: List[Dict[str, str]], 
                           temperature: float = None,
                           max_tokens: int = None) -> Dict[str, Any]:
        """
        Generate completion using LiteLLM
        
        Args:
            model_name: Short model name from Config.MODEL_CONFIGS
            messages: List of message dictionaries with role and content
            temperature: Temperature for generation (defaults to Config.DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (defaults to model's max_output_tokens)
            
        Returns:
            Dictionary containing the completion result
        """
        # Use default temperature if not specified
        if temperature is None:
            temperature = Config.DEFAULT_TEMPERATURE
            
        # Get full model name formatted for LiteLLM
        litellm_model = Config.get_litellm_model_name(model_name)
        
        # Get model config
        model_config = Config.get_model_config(model_name)
        
        # Use model's max output tokens if not specified
        if max_tokens is None:
            max_tokens = model_config.get("max_output_tokens", 2048)
        
        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
        except Exception as e:
            # Log the error
            print(f"Error generating completion: {str(e)}")
            raise

# Create a singleton instance
litellm_service = LiteLLMService()