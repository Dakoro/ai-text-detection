
# Use Python slim image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /

# Copy project files
COPY . ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache
RUN uv remove jupyter 

# Set default port
ENV PORT=8000
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD ["sh", "-c", "uv run src/server.py"]