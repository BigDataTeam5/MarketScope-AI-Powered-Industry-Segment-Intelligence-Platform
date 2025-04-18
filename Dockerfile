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

# Copy requirements first (for better caching)
COPY requirements.txt ./

# Create frontend requirements file
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

# Copy any existing code (this will take whatever is available)
COPY . .

# Create required directories if they don't exist
RUN mkdir -p /app/frontend/pages
RUN mkdir -p /app/agents
RUN mkdir -p /app/mcp_servers
RUN mkdir -p /app/config

# Create minimal frontend app if it doesn't exist
RUN if [ ! -f "/app/frontend/app.py" ]; then \
    echo "import streamlit as st" > /app/frontend/app.py && \
    echo "st.title('MarketScope AI Platform')" >> /app/frontend/app.py && \
    echo "st.write('Backend API URL: http://34.42.74.104:8000')" >> /app/frontend/app.py && \
    echo "st.info('Access the API documentation at http://34.42.74.104:8000/docs')" >> /app/frontend/app.py; \
fi

# Create a utils.py file if it doesn't exist
RUN if [ ! -f "/app/frontend/utils.py" ]; then \
    echo "import streamlit as st" > /app/frontend/utils.py && \
    echo "import requests" >> /app/frontend/utils.py && \
    echo "import sys, os" >> /app/frontend/utils.py && \
    echo "API_URL = 'http://34.42.74.104:8000'" >> /app/frontend/utils.py && \
    echo "def sidebar():" >> /app/frontend/utils.py && \
    echo "    with st.sidebar:" >> /app/frontend/utils.py && \
    echo "        st.title('MarketScope AI')" >> /app/frontend/utils.py; \
fi

# Create basic MCP server if it doesn't exist
RUN if [ ! -f "/app/mcp_servers/run_all_servers.py" ]; then \
    echo "import time" > /app/mcp_servers/run_all_servers.py && \
    echo "print('MCP Server started')" >> /app/mcp_servers/run_all_servers.py && \
    echo "while True:" >> /app/mcp_servers/run_all_servers.py && \
    echo "    time.sleep(60)" >> /app/mcp_servers/run_all_servers.py; \
fi

# List directories to verify
RUN echo "Contents of /app:" && ls -la /app && \
    echo "Contents of /app/frontend:" && ls -la /app/frontend

# Fix line endings (esp. if files came from Windows)
RUN find /app -name "*.py" -exec dos2unix {} \; 2>/dev/null || true

# Optional: Precompile Python files to catch syntax errors
RUN find /app -name "*.py" | xargs -n1 python -m py_compile 2>/dev/null || true

# Create start_services.sh script
RUN echo '#!/bin/bash' > /app/start_services.sh && \
    echo 'echo "Starting MCP servers..."' >> /app/start_services.sh && \
    echo 'python /app/mcp_servers/run_all_servers.py &' >> /app/start_services.sh && \
    echo 'MCP_PID=$!' >> /app/start_services.sh && \
    echo 'echo "MCP servers started with PID: $MCP_PID"' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo 'echo "Contents of current directory:"' >> /app/start_services.sh && \
    echo 'ls -la' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo 'echo "Contents of /app directory:"' >> /app/start_services.sh && \
    echo 'ls -la /app' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo 'echo "Contents of /app/frontend directory:"' >> /app/start_services.sh && \
    echo 'ls -la /app/frontend' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo 'echo "Starting Streamlit frontend..."' >> /app/start_services.sh && \
    echo 'cd /app/frontend && streamlit run app.py --server.port=8501 --server.address=0.0.0.0' >> /app/start_services.sh

# Make the script executable
RUN chmod +x /app/start_services.sh

# Expose backend service ports (MCP + optional APIs)
EXPOSE 8000 8001 8010 8011 8012 8013 8014 8501

# Start both services
CMD ["/app/start_services.sh"]