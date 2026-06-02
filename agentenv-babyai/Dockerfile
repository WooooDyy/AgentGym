FROM python:3.10-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy package files
COPY pyproject.toml .
COPY agentenv_babyai/ agentenv_babyai/

# Install dependencies
RUN uv pip install --system -e .

# Expose port
EXPOSE 36002

# Run with CORS enabled for visualization
ENV VISUAL=true

CMD ["babyai", "--host", "0.0.0.0", "--port", "36002"]
