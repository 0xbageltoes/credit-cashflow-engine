# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

LABEL maintainer="cashflow-engine-team"
LABEL description="Credit Cashflow Engine Microservice"
LABEL version="1.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \  # Required for numpy
    curl \      # For health checks
    postgresql-client \ # For database backups
    supervisor \  # For process management
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Copy supervisor configuration
COPY supervisor.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directory
RUN mkdir -p /var/log && \
    touch /var/log/supervisord.log /var/log/api-err.log /var/log/api-out.log \
          /var/log/celery-worker-err.log /var/log/celery-worker-out.log \
          /var/log/celery-beat-err.log /var/log/celery-beat-out.log \
          /var/log/flower-err.log /var/log/flower-out.log \
          /var/log/prometheus-err.log /var/log/prometheus-out.log

# Create data directories
RUN mkdir -p /app/logs /app/data

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /var/log
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app:$PATH" \
    ENV="production" \
    WORKERS=4 \
    CELERY_CONCURRENCY=4

# Expose ports
EXPOSE 8000  # API
EXPOSE 5555  # Flower
EXPOSE 9090  # Prometheus

# Health check with improved parameters
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use supervisord for production
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# For development use only
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
