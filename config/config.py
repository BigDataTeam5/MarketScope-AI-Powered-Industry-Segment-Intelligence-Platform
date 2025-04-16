"""
Unified configuration for MarketScope
Consolidates all configuration settings in one place
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Unified configuration for the MarketScope application"""
    
    # AWS Credentials
    AWS_SERVER_PUBLIC_KEY = os.getenv("AWS_SERVER_PUBLIC_KEY")
    AWS_SERVER_SECRET_KEY = os.getenv("AWS_SERVER_SECRET_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
    
    # S3 Configuration
    BUCKET_NAME = os.getenv("BUCKET_NAME", "finalproject-product")
    S3_CHUNKS_PATH = os.getenv("S3_CHUNKS_PATH", "book-content/chunks/")
    S3_CHUNKS_FILE = os.getenv("S3_CHUNKS_FILE", "S3D7W4_Marketing_Management_chunks.json")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "marketscope-index")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Server Configuration
    MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
    API_PORT = int(os.getenv("API_PORT", "8001"))
    MCP_URL = f"http://localhost:{MCP_PORT}/sse"
    
    # Agent Configuration
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt4o")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    
    # Tool Configuration
    ENABLED_TOOLS = [
        "pinecone_search",
        "fetch_s3_chunk",
        "get_chunks_metadata",
        "get_all_retrieved_chunks",
        "analyze_market_segment",
        "generate_segment_strategy"
    ]
    
    # LiteLLM Configuration
    USE_LITELLM = os.getenv("USE_LITELLM", "True").lower() == "true"
    LITELLM_CACHE_ENABLED = os.getenv("LITELLM_CACHE_ENABLED", "True").lower() == "true"
    LITELLM_CACHE_TYPE = os.getenv("LITELLM_CACHE_TYPE", "redis")
    LITELLM_CACHE_HOST = os.getenv("LITELLM_CACHE_HOST", "localhost")
    LITELLM_CACHE_PORT = int(os.getenv("LITELLM_CACHE_PORT", "6379"))
    
    # Model Configurations with LiteLLM model names
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "gpt4o": {
            "name": "GPT-4o",
            "model": "gpt-4o",  # LiteLLM format
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "supports_images": True,
            "provider": "openai"
        },
        "gemini": {
            "name": "Gemini Flash",
            "model": "gemini-1.5-flash",  # LiteLLM format
            "max_input_tokens": 100000,
            "max_output_tokens": 4000,
            "supports_images": True,
            "provider": "google"
        },
        "deepseek": {
            "name": "DeepSeek",
            "model": "deepseek-reasoner",  # LiteLLM format
            "max_input_tokens": 16000,
            "max_output_tokens": 2048,
            "supports_images": False,
            "provider": "deepseek"
        },
        "claude": {
            "name": "Claude 3 Sonnet",
            "model": "claude-3-5-sonnet-20240620",  # LiteLLM format
            "max_input_tokens": 100000,
            "max_output_tokens": 4096,
            "supports_images": True,
            "provider": "anthropic"
        },
        "grok": {
            "name": "Grok",
            "model": "grok-2-latest",  # LiteLLM format
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "supports_images": True,
            "provider": "xai"
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return cls.MODEL_CONFIGS.get(model_name, cls.MODEL_CONFIGS["gpt4o"])
    
    @classmethod
    def get_litellm_model_name(cls, model_name: str) -> str:
        """Get the litellm formatted model name"""
        model_config = cls.get_model_config(model_name)
        provider = model_config.get("provider", "")
        model = model_config.get("model", "")
        
        if provider and model and cls.USE_LITELLM:
            return f"{provider}/{model}"
        return model
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names"""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate that required configuration is present
        
        Returns:
            List of missing configuration variables
        """
        required_vars = [
            "AWS_SERVER_PUBLIC_KEY", 
            "AWS_SERVER_SECRET_KEY",
            "PINECONE_API_KEY",
            "OPENAI_API_KEY"
        ]
        
        missing = [var for var in required_vars if not getattr(cls, var)]
        
        return missing
    
    @classmethod
    def get_litellm_params(cls) -> Dict[str, Any]:
        """Get parameters for LiteLLM configuration"""
        return {
            "cache": cls.LITELLM_CACHE_ENABLED,
            "cache_params": {
                "type": cls.LITELLM_CACHE_TYPE,
                "host": cls.LITELLM_CACHE_HOST,
                "port": cls.LITELLM_CACHE_PORT
            } if cls.LITELLM_CACHE_ENABLED else {},
            "api_key": {
                "openai": cls.OPENAI_API_KEY,
                "anthropic": cls.ANTHROPIC_API_KEY,
                "google": cls.GOOGLE_API_KEY,
                "xai": cls.GROK_API_KEY,
                "deepseek": cls.DEEPSEEK_API_KEY
            }
        }