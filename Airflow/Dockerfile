# Use Airflow base image
FROM apache/airflow:2.10.4

# Switch to root to install system dependencies
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gpg \
    git \
    pkg-config \
    build-essential \
    cmake \
    python3-dev \
    libpq-dev \
    libssl-dev \
    libonig-dev \
    libbz2-dev \
    libzstd-dev \
    liblzma-dev \
    ca-certificates \
    poppler-utils \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# RUN mkdir -p /opt/airflow/chroma_db && chown -R airflow:root /opt/airflow/chroma_db

USER airflow

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Copy filtered requirements
COPY Airflow/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt --prefer-binary
