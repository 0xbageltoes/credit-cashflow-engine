# Credit Cashflow Engine Production Deployment Guide

This guide provides instructions for deploying the Credit Cashflow Engine microservice to a production environment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Deployment Options](#deployment-options)
4. [Monitoring and Logging](#monitoring-and-logging)
5. [Scaling](#scaling)
6. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker and Docker Compose
- AWS CLI configured with appropriate permissions
- Access to your production database
- TLS certificates for secure communication

## Environment Setup

### 1. Configure Environment Variables

Create a production `.env` file based on the `.env.example` template:

```bash
cp .env.example .env.production
```

Edit the `.env.production` file to include all required production settings:

- Set `ENV=production`
- Configure database credentials
- Set strong, unique secret keys
- Configure monitoring tools (Sentry, Prometheus)
- Set up appropriate rate limiting values
- Configure Redis for production use

### 2. Prepare Docker Images

Build the production Docker image:

```bash
docker build -t credit-cashflow-engine:latest .
```

For AWS ECR deployment:

```bash
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag credit-cashflow-engine:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/credit-cashflow-engine:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/credit-cashflow-engine:latest
```

## Deployment Options

### Option 1: Docker Compose (Single Host)

For smaller deployments on a single host:

1. Copy `docker-compose.prod.yml` to your server
2. Copy your `.env.production` file to the same location
3. Run the stack:

```bash
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

### Option 2: AWS ECS Deployment

For scalable cloud deployments:

1. Ensure your task definition (`task-definition.json`) is up-to-date
2. Use the provided CI/CD workflow in `.github/workflows/ci-cd.yml`
3. The workflow will:
   - Build and test the application
   - Push the Docker image to ECR
   - Deploy the updated task definition to ECS

To manually deploy to ECS:

```bash
aws ecs update-service --cluster your-cluster-name --service credit-cashflow-engine --force-new-deployment
```

### Option 3: Kubernetes Deployment

For Kubernetes-based deployments:

1. Use the Kubernetes manifests in the `k8s/` directory
2. Apply the configurations:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Monitoring and Logging

### Health Checks

The service provides a `/health` endpoint that returns the status of:
- API availability
- Database connection
- Redis connection
- Celery worker status

### Metrics

Prometheus metrics are available at the `/metrics` endpoint, including:
- Request counts and durations
- Error rates
- Memory and CPU usage
- Custom business metrics (cashflow calculations, etc.)

### Grafana Dashboard

1. Deploy Grafana alongside Prometheus
2. Import the provided dashboard from `grafana/provisioning/dashboards/dashboard.json`
3. Set up alerts for critical metrics

### Logging

Logs are output in JSON format for easy parsing by log aggregation tools:
- Application logs contain structured information
- Request logs include request ID and correlation ID for tracing
- Error logs include stack traces and contextual information

### Sentry Integration

Configure Sentry for error tracking:
1. Set `SENTRY_DSN` in your environment variables
2. Errors will be automatically captured and grouped
3. Performance monitoring is enabled to track slow requests

## Scaling

### Horizontal Scaling

- Increase the number of API containers to handle more traffic
- Increase the number of Celery workers for more parallel task processing
- In ECS, adjust the desired count of tasks
- In Kubernetes, adjust the replica count

### Vertical Scaling

- Increase CPU and memory allocations in task definitions
- Adjust worker concurrency via `CELERY_CONCURRENCY` env var
- Adjust calculation thread pool size via `CALCULATION_THREAD_POOL_SIZE` env var

## Backup and Disaster Recovery

### Database Backups

Automated backups are configured to run daily using the backup manager:

```python
from app.core.backup import BackupManager

backup = BackupManager()
backup.backup_database(
    db_name="your_db_name",
    db_user="your_db_user",
    db_password="your_db_password",
    db_host="your_db_host"
)
```

### Restoring from Backup

To restore from a backup:

```python
backup = BackupManager()
backup.restore_database(
    backup_key="backups/database/db_backup_yourdb_20250101120000.sql",
    db_name="your_db_name",
    db_user="your_db_user",
    db_password="your_db_password",
    db_host="your_db_host"
)
```

## Security Considerations

### Authentication and Authorization

- All API endpoints (except health and metrics) require authentication
- JWT tokens are used for authentication
- Ensure JWT secrets are properly managed and rotated

### Rate Limiting

Rate limiting is enabled by default:
- 100 requests per hour per IP/user by default
- Adjust `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW` for your needs

### Security Headers

Security headers are automatically added to all responses:
- Content Security Policy
- XSS Protection
- HSTS
- X-Frame-Options
- X-Content-Type-Options

### TLS/SSL

Always enable TLS in production:
- Use a reverse proxy (Nginx/Apache) for TLS termination
- Or configure TLS in your cloud load balancer

## Troubleshooting

### Common Issues

#### API Service Won't Start

Check the logs:
```bash
docker logs <container_id>
```

Common causes:
- Missing environment variables
- Database connection issues
- Redis connection issues

#### Slow Performance

- Check Prometheus metrics for bottlenecks
- Ensure enough memory is allocated
- Consider scaling horizontally for more capacity

#### High Error Rates

- Check Sentry for error details
- Review logs for patterns
- Check database connection pool settings

### Support

For additional support, contact the development team at:
- Email: cashflow-engine-support@example.com
- Slack: #cashflow-engine-support
