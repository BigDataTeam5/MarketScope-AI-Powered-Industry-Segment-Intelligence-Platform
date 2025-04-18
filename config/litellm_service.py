"""
LiteLLM Service Configuration
"""
import os
import logging
from typing import Any, Dict, List, Optional
import litellm
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("litellm_service")

def get_llm_model():
    """Get the configured LLM model"""
    try:
        # Configure LiteLLM with OpenAI key
        litellm.api_key = Config.OPENAI_API_KEY

        # Create a wrapper class that matches LangChain's interface
        class LiteLLMWrap:
            def __init__(self):
                self.model = Config.DEFAULT_MODEL
                self.temperature = Config.TEMPERATURE if hasattr(Config, 'TEMPERATURE') else 0.7
                self.max_tokens = Config.MAX_TOKENS if hasattr(Config, 'MAX_TOKENS') else 1000
                
            def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
                try:
                    # Process messages to match LiteLLM format
                    formatted_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            formatted_messages.append(msg)
                        else:
                            # Handle LangChain message objects
                            formatted_messages.append({
                                "role": msg.type if hasattr(msg, "type") else "user",
                                "content": msg.content if hasattr(msg, "content") else str(msg)
                            })

                    # Call LiteLLM with the original completion function that's available in your version
                    response = litellm.completion(
                        model=self.model,
                        messages=formatted_messages,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens)
                    )
                    
                    # Extract the response content
                    if response and hasattr(response, 'choices') and len(response.choices) > 0:
                        content = response.choices[0].message.content
                    else:
                        content = "No response generated"
                    
                    # Return in a format compatible with both LangChain and direct use
                    return {
                        "type": "assistant",
                        "content": content
                    }
                    
                except Exception as e:
                    logger.error(f"Error in LiteLLM call: {str(e)}")
                    return {
                        "type": "assistant",
                        "content": f"Error generating response: {str(e)}"
                    }

        return LiteLLMWrap()
        
    except Exception as e:
        logger.error(f"Error initializing LLM model: {str(e)}")
        
        # Return a mock LLM for development/testing
        class MockLLM:
            def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
                return {
                    "type": "assistant",
                    "content": "This is a mock response since the LLM is not available."
                }
        
        return MockLLM()
