FROM python:3.10-slim

# Install Java (required for ScienceWorld)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy package files
COPY pyproject.toml .
COPY agentenv_sciworld/ agentenv_sciworld/

# Install dependencies
RUN uv pip install --system -e .

# Expose port
EXPOSE 36003

# Run with CORS enabled for visualization
ENV VISUAL=true

CMD ["sciworld", "--host", "0.0.0.0", "--port", "36003"]
