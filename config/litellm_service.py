"""
LiteLLM service integration for MarketScope
Provides standardized access to LLM models
"""
import os
import logging
from typing import Optional, Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("litellm_service")

# Import Config for default model settings
try:
    from config.config import Config
except ImportError:
    # Fallback if Config is not available
    class Config:
        DEFAULT_MODEL = "gpt-4o"
        DEFAULT_TEMPERATURE = 0.3
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_llm_model(model_name: Optional[str] = None, temperature: Optional[float] = None):
    """
    Get a language model instance using LiteLLM
    
    Args:
        model_name: Name of the model to use (defaults to Config.DEFAULT_MODEL)
        temperature: Temperature setting (defaults to Config.DEFAULT_TEMPERATURE)
        
    Returns:
        LangChain compatible model
    """
    try:
# Import LiteLLM and related packages
        try:
            import litellm
            from langchain_community.llms.litellm import LiteLLM
        except ImportError:
            try:
                import litellm
                # Fallback for older langchain versions
                from langchain.llms.litellm import LiteLLM
            except ImportError:
                # Try to install the package if it's missing
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-community", "litellm"])
                import litellm
                from langchain_community.llms.litellm import LiteLLM
        
        # Set default API keys
        litellm.openai_api_key = os.getenv("OPENAI_API_KEY", Config.OPENAI_API_KEY)
        litellm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", Config.ANTHROPIC_API_KEY)
        litellm.google_api_key = os.getenv("GOOGLE_API_KEY", Config.GOOGLE_API_KEY)
        
        # Set model name and temperature
        model = model_name or Config.DEFAULT_MODEL
        temp = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE
        
        # Create and return LiteLLM instance
        return LiteLLM(
            model=model,
            temperature=temp,
            streaming=True
        )
    
    except ImportError as e:
        logger.error(f"Error importing LiteLLM: {str(e)}")
        logger.warning("Falling back to mock LLM for development")
        
        # Fallback to mock LLM for development
        from langchain_community.llms.fake import FakeListLLM
        
        return FakeListLLM(
            responses=["I'm a mock LLM model for development. LiteLLM could not be imported."]
        )
    except Exception as e:
        logger.error(f"Error initializing LLM model: {str(e)}")
        
        # Return a more informative error message
        from langchain_community.llms.fake import FakeListLLM
        return FakeListLLM(
            responses=[f"Error initializing LLM model: {str(e)}. Please check your API keys and model configuration."]
        )

def get_embeddings_model():
    """
    Get an embeddings model
    
    Returns:
        Embeddings model instance
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY", Config.OPENAI_API_KEY)
        )
    except ImportError:
        logger.error("Error importing OpenAIEmbeddings")
        try:
            # Try older langchain version
            from langchain.embeddings.openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model_name="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY", Config.OPENAI_API_KEY)
            )
        except ImportError:
            # Return a mock embeddings model
            from langchain_community.embeddings.fake import FakeEmbeddings
            return FakeEmbeddings(size=1536)
