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

# Copy requirements.txt for pip installation (much faster than Poetry for Docker builds)
COPY requirements.txt ./

# Install dependencies using pip (faster and more reliable in Docker)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Fix line endings (esp. if files came from Windows)
RUN find /app -name "*.py" -exec dos2unix {} \;

# Optional: Precompile Python files to catch syntax errors
RUN python -m py_compile $(find /app -name "*.py")

# Expose backend service ports (MCP + optional APIs)
EXPOSE 8000 8001 8010 8011 8012 8013 8014

# Start the unified MCP server
CMD ["python", "mcp_servers/run_all_servers.py"]
