FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Avoid creating virtualenvs inside container
RUN poetry config virtualenvs.create false

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies separately for cache efficiency
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-interaction --no-ansi

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
