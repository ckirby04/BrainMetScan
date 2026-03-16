# Stage 1: Builder — install dependencies using same base as runtime
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only production dependencies (exclude dev section from requirements)
COPY pyproject.toml .
COPY src/ src/
COPY run_server.py .
COPY configs/ configs/

RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime with CUDA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-distutils && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/lib/python3.11/site-packages /usr/lib/python3/dist-packages

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY run_server.py .
COPY configs/ configs/
COPY src/ src/

# Model directory will be mounted as volume
RUN mkdir -p /app/model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
