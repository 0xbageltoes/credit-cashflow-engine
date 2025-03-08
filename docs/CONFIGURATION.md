# Credit Cashflow Engine Configuration Guide

This document provides detailed information about configuring the Credit Cashflow Engine microservice for different environments, particularly for production use.

## Configuration Principles

The Credit Cashflow Engine follows these configuration principles:

1. **Environment-based configuration**: Different settings for development, testing, and production environments
2. **Secrets management**: Sensitive information is stored in environment variables
3. **Defaults for development**: Reasonable defaults are provided for development environments
4. **Explicit production settings**: Production settings require explicit configuration

## Configuration Sources

Configuration settings are loaded in the following order (later sources override earlier ones):

1. Default values in `app/core/config.py`
2. Environment variables
3. `.env` file (loaded through python-dotenv)

## Core Settings

### Application Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `ENV` | Environment (development, testing, production) | development | Yes |
| `VERSION` | Application version | 1.0.0 | No |
| `API_V1_STR` | API v1 prefix | /api/v1 | No |
| `WORKERS` | Number of Gunicorn workers | 1 | Yes |
| `LOG_LEVEL` | Logging level (debug, info, warning, error) | info | Yes |
| `SECRET_KEY` | Secret key for security features | development_secret_key | Yes |

### Database Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `SUPABASE_URL` | Supabase URL | None | Yes |
| `SUPABASE_KEY` | Supabase Key | None | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Service Role Key | None | Yes |
| `SUPABASE_JWT_SECRET` | Supabase JWT Secret | None | Yes |

### Redis Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `UPSTASH_REDIS_HOST` | Redis host | None | Yes |
| `UPSTASH_REDIS_PORT` | Redis port | None | Yes |
| `UPSTASH_REDIS_PASSWORD` | Redis password | None | Yes |

### Celery Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `CELERY_TASK_ALWAYS_EAGER` | Run tasks synchronously | True in dev, False in prod | No |
| `CELERY_WORKER_CONCURRENCY` | Number of Celery worker processes | 4 | Yes |
| `CELERY_TASK_TIME_LIMIT` | Task timeout in seconds | 1800 | No |
| `CELERY_TASK_MAX_RETRIES` | Maximum number of task retries | 3 | No |

### Performance Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `RATE_LIMIT_REQUESTS` | Max requests per window | 100 | Yes |
| `RATE_LIMIT_WINDOW` | Rate limit window in seconds | 3600 | Yes |
| `CACHE_TTL` | Cache time-to-live in seconds | 3600 | No |
| `BATCH_SIZE` | Size of batch operations | 1000 | No |
| `CALCULATION_THREAD_POOL_SIZE` | Thread pool size for calculations | 4 | Yes |

### Security Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `HASH_ALGORITHM` | JWT hash algorithm | HS256 | No |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token expiration | 30 | Yes |
| `SSL_VERIFICATION` | Verify SSL certificates | True in prod | No |

### Monitoring Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `SENTRY_DSN` | Sentry DSN for error tracking | None | Yes |
| `PROMETHEUS_ENABLED` | Enable Prometheus metrics | True | No |
| `LOGGING_JSON_FORMAT` | Use JSON format for logs | True | No |

### AWS Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `AWS_REGION` | AWS region | us-east-1 | Yes for AWS |
| `AWS_S3_BUCKET` | S3 bucket for backups | None | Yes for backups |

### Market Data API Settings

| Variable | Description | Default | Required in Production |
|----------|-------------|---------|------------------------|
| `FRED_API_KEY` | FRED API key | None | If using FRED |
| `BLOOMBERG_API_KEY` | Bloomberg API key | None | If using Bloomberg |

## Environment-Specific Configurations

### Development Environment

Development configuration focuses on ease of use and debugging:

```dotenv
ENV=development
LOG_LEVEL=debug
CELERY_TASK_ALWAYS_EAGER=true
```

### Testing Environment

Testing configuration is isolated and deterministic:

```dotenv
ENV=testing
LOG_LEVEL=info
CELERY_TASK_ALWAYS_EAGER=true
```

### Production Environment

Production configuration focuses on security, reliability, and performance:

```dotenv
ENV=production
LOG_LEVEL=info
WORKERS=4
CALCULATION_THREAD_POOL_SIZE=8
CELERY_WORKER_CONCURRENCY=8
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
SENTRY_DSN=https://your-sentry-project-key@sentry.io/your-project-id
```

## CORS Configuration

By default, CORS is configured to allow all origins in development, but should be restricted in production:

```dotenv
# Development
CORS_ORIGINS=*

# Production
CORS_ORIGINS=https://your-frontend-domain.com,https://admin.your-domain.com
```

## Supervisor Configuration

The application uses Supervisor to manage processes in production. The configuration is in `supervisor.conf` and controls:

- API server (Gunicorn/Uvicorn)
- Celery workers
- Celery beat scheduler
- Flower monitoring
- Prometheus server

## Docker Environment Variables

When running in Docker, environment variables can be passed via:

1. `.env` file with the `--env-file` option
2. Environment variables passed to the container
3. Docker Compose environment section

Example Docker Compose environment configuration:

```yaml
services:
  api:
    image: credit-cashflow-engine:latest
    environment:
      - ENV=production
      - WORKERS=4
      - CELERY_WORKER_CONCURRENCY=8
```

## Kubernetes ConfigMaps and Secrets

When deploying to Kubernetes, use:

- ConfigMaps for non-sensitive configuration
- Secrets for sensitive data

Example ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: credit-cashflow-engine-config
data:
  ENV: "production"
  WORKERS: "4"
  CALCULATION_THREAD_POOL_SIZE: "8"
```

Example Secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: credit-cashflow-engine-secrets
type: Opaque
data:
  SUPABASE_URL: <base64-encoded-value>
  SUPABASE_KEY: <base64-encoded-value>
  SECRET_KEY: <base64-encoded-value>
```

## Best Practices

1. **Never commit secrets to version control**
2. **Rotate secrets regularly**
3. **Use different secrets for different environments**
4. **Monitor configuration changes**
5. **Document all configuration options**
6. **Validate configuration at startup**
7. **Fail fast on misconfiguration**
