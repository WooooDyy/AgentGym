FROM python:3.10-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy package files
COPY pyproject.toml .
COPY agentenv_textcraft/ agentenv_textcraft/

# Install dependencies
RUN uv pip install --system -e .

# Expose port
EXPOSE 36001

# Run with CORS enabled for visualization
ENV VISUAL=true

CMD ["textcraft", "--host", "0.0.0.0", "--port", "36001"]
