from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Core settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_SERVER_PUBLIC_KEY = os.getenv("AWS_SERVER_PUBLIC_KEY")
AWS_SERVER_SECRET_KEY = os.getenv("AWS_SERVER_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_CHUNKS_PATH = os.getenv("S3_CHUNKS_PATH")
S3_CHUNKS_FILE = os.getenv("S3_CHUNKS_FILE")

# LangSmith settings
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")

# Server settings
MCP_SERVER_URL = "http://localhost:8000/sse"
API_SERVER_PORT = 8001

def setup_langsmith():
    """Set up LangSmith environment variables"""
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT