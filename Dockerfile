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

# Copy requirements file
COPY requirements.txt ./

# Create frontend requirements file
RUN echo "# Frontend requirements" > frontend-requirements.txt
RUN echo "streamlit==1.44.0" >> frontend-requirements.txt
RUN echo "requests==2.32.0" >> frontend-requirements.txt
RUN echo "python-dotenv==1.0.0" >> frontend-requirements.txt
RUN echo "pandas==2.2.0" >> frontend-requirements.txt
RUN echo "numpy==1.26.0" >> frontend-requirements.txt
RUN echo "matplotlib==3.8.0" >> frontend-requirements.txt
RUN echo "plotly==5.18.0" >> frontend-requirements.txt
RUN echo "scikit-learn==1.4.0" >> frontend-requirements.txt
RUN echo "litellm==1.17.0" >> frontend-requirements.txt
RUN echo "snowflake-connector-python==3.6.0" >> frontend-requirements.txt
RUN echo "fastapi==0.110.0" >> frontend-requirements.txt
RUN echo "boto3==1.34.0" >> frontend-requirements.txt
RUN echo "pydantic==1.10.8" >> frontend-requirements.txt
RUN echo "altair<5" >> frontend-requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r frontend-requirements.txt

# Install specific versions to fix compatibility issues
RUN pip install --no-cache-dir pydantic==1.10.8
RUN pip install --no-cache-dir pydantic-settings==2.0.3
RUN pip install --no-cache-dir openai==0.28.1

# Copy everything to the container
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

# Create start_services.sh script with improved Streamlit settings
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
    echo 'cd /app/frontend && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false' >> /app/start_services.sh

# Make the script executable
RUN chmod +x /app/start_services.sh

# Expose all necessary ports
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012 8013 8014 8015 8501

# Start both services
CMD ["/app/start_services.sh"]