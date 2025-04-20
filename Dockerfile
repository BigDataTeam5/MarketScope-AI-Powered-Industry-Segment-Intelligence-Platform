FROM python:3.10-slim

# Prevent Python from creating .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages: dos2unix (for line ending fixes), curl (to install Poetry)
RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false && \
    apt-get install -y --no-install-recommends \
    dos2unix \
    curl \
    build-essential \
    git \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to NOT create virtual environments
RUN poetry config virtualenvs.create false

# Set our working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy only pyproject.toml first
COPY pyproject.toml ./

# Generate a fresh lock file (this replaces the need for copying an existing one)
RUN poetry lock

# Install dependencies from the freshly generated lock file
RUN poetry install --no-root --no-interaction --no-ansi

# Make sure sentence-transformers is installed (often missing from dependencies)
RUN pip install sentence-transformers pinecone

# Now copy everything else into /app
COPY . /app

# Make sure the .env file is copied with proper permissions
COPY .env /app/.env
RUN chmod 600 /app/.env

# Convert all Python files from CRLF to LF (handles Windows line endings)
RUN find /app -name "*.py" -exec dos2unix {} \;

# Optional: check syntax for all .py files at build time but continue on error
RUN find /app -name "*.py" | grep -v "/mcp/" | xargs -n1 python -m py_compile || echo "Ignoring compile errors and continuing"

# Expose all backend service ports
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012 8013 8014 8015

# Start the unified MCP server with all services
CMD ["python", "mcp_servers/run_all_servers.py"]