FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    curl \
    build-essential \
    git \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy requirements.txt for pip installation
COPY requirements.txt ./

# Create frontend requirements if it doesn't exist
RUN echo "# Frontend requirements" > frontend-requirements.txt
RUN echo "streamlit" >> frontend-requirements.txt
RUN echo "requests" >> frontend-requirements.txt
RUN echo "python-dotenv" >> frontend-requirements.txt
RUN echo "openai" >> frontend-requirements.txt
RUN echo "pandas" >> frontend-requirements.txt
RUN echo "numpy" >> frontend-requirements.txt
RUN echo "scikit-learn" >> frontend-requirements.txt
RUN echo "pinecone" >> frontend-requirements.txt
RUN echo "boto3" >> frontend-requirements.txt
RUN echo "litellm" >> frontend-requirements.txt

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r frontend-requirements.txt

# Copy all project code
COPY . .

# Fix line endings (esp. if files came from Windows)
RUN find /app -name "*.py" -exec dos2unix {} \;

# Optional: Precompile Python files to catch syntax errors
RUN python -m py_compile $(find /app -name "*.py")

# Expose backend service ports (MCP + optional APIs)
EXPOSE 8000 8001 8010 8011 8012 8013 8014 8501

# Create a script to run both services
RUN echo '#!/bin/bash\n\
python mcp_servers/run_all_servers.py &\n\
cd frontend && streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start_services.sh

RUN chmod +x /app/start_services.sh

# Start both services
CMD ["/app/start_services.sh"]
