# Asset Classes API Production Deployment Guide

This guide provides detailed instructions for deploying the Asset Classes API module in a production environment with proper configuration, monitoring, and maintenance procedures.

## Prerequisites

- Access to production server(s) with Python 3.9+ installed
- Redis server (v6.0+) for caching and performance optimization
- Production database properly configured and secured
- HTTPS certificates for secure API access
- Access to monitoring and logging systems

## Environment Configuration

### Required Environment Variables

Create a `.env` file with the following variables (or set them in your environment):

```
# Core Configuration
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Security Settings
SECRET_KEY=<strong-random-key>
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
CORS_ORIGINS=https://app.your-domain.com,https://admin.your-domain.com

# Redis Configuration
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=<redis-password>
REDIS_SSL=True
REDIS_DATABASE=0
REDIS_TIMEOUT=5
REDIS_RETRY_COUNT=3
REDIS_RETRY_DELAY=0.5
REDIS_CONNECTION_POOL_SIZE=10
REDIS_MAX_CONNECTIONS=20

# Cache Settings
CACHE_TTL=3600
ENABLE_CACHING=True

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_DEFAULT=100  # requests per minute
RATE_LIMIT_BURST=150

# Monitoring
ENABLE_PROMETHEUS=True
ENABLE_SENTRY=True
SENTRY_DSN=<your-sentry-dsn>
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# Database
DATABASE_URL=<your-database-connection-string>
DB_CONNECTION_TIMEOUT=5
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

### Logging Configuration

Create a `logging.conf` file with proper production logging configuration:

```ini
[loggers]
keys=root,api,services,models

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=jsonFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[logger_api]
level=INFO
handlers=consoleHandler,fileHandler
qualname=app.api
propagate=0

[logger_services]
level=INFO
handlers=consoleHandler,fileHandler
qualname=app.services
propagate=0

[logger_models]
level=INFO
handlers=consoleHandler,fileHandler
qualname=app.models
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=jsonFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=handlers.TimedRotatingFileHandler
level=INFO
formatter=jsonFormatter
args=('/var/log/credit-cashflow-engine/api.log', 'midnight', 1, 30, 'utf-8')

[formatter_jsonFormatter]
class=app.core.logging.JsonFormatter
format=%(asctime)s %(name)s %(levelname)s %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

## Production Deployment Steps

### 1. Server Setup

1. Set up a dedicated server or container environment
2. Configure proper resource limits (CPU, memory, disk)
3. Set up firewall rules to restrict access to necessary ports only
4. Install required OS packages:
   ```bash
   apt-get update
   apt-get install -y build-essential python3-dev libssl-dev
   ```

### 2. Application Deployment

1. Clone the repository to a dedicated directory
   ```bash
   git clone https://github.com/your-org/credit-cashflow-engine.git /opt/credit-cashflow-engine
   ```

2. Set up a Python virtual environment
   ```bash
   cd /opt/credit-cashflow-engine
   python -m venv venv
   source venv/bin/activate
   ```

3. Install production dependencies
   ```bash
   pip install -r requirements/production.txt
   ```

4. Set environment variables
   ```bash
   cp .env.example .env
   # Edit .env with production values
   ```

5. Run database migrations
   ```bash
   alembic upgrade head
   ```

6. Set proper file permissions
   ```bash
   chmod -R 750 /opt/credit-cashflow-engine
   chmod 640 /opt/credit-cashflow-engine/.env
   ```

### 3. Web Server Configuration

#### Using Gunicorn and Nginx

1. Install Gunicorn
   ```bash
   pip install gunicorn
   ```

2. Create a Gunicorn configuration file (`gunicorn_config.py`)
   ```python
   import multiprocessing

   # Gunicorn config
   bind = "127.0.0.1:8000"
   workers = multiprocessing.cpu_count() * 2 + 1
   worker_class = "uvicorn.workers.UvicornWorker"
   max_requests = 1000
   max_requests_jitter = 50
   timeout = 60
   keepalive = 5
   errorlog = "/var/log/credit-cashflow-engine/gunicorn-error.log"
   accesslog = "/var/log/credit-cashflow-engine/gunicorn-access.log"
   loglevel = "info"
   ```

3. Configure Nginx as a reverse proxy
   ```nginx
   server {
       listen 443 ssl http2;
       server_name api.your-domain.com;

       ssl_certificate /etc/letsencrypt/live/api.your-domain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/api.your-domain.com/privkey.pem;

       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_prefer_server_ciphers on;
       ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;

       ssl_session_cache shared:SSL:10m;
       ssl_session_timeout 10m;
       ssl_session_tickets off;

       add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
       add_header X-Frame-Options DENY;
       add_header X-Content-Type-Options nosniff;
       add_header X-XSS-Protection "1; mode=block";
       add_header Content-Security-Policy "default-src 'self'; script-src 'self'; img-src 'self' data:; style-src 'self'; font-src 'self'; connect-src 'self'";

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_set_header X-Request-ID $request_id;
           proxy_redirect off;
           proxy_buffering off;
           proxy_read_timeout 120s;
       }

       location /health {
           proxy_pass http://127.0.0.1:8000/health;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           access_log off;
       }

       location /metrics {
           proxy_pass http://127.0.0.1:8000/metrics;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           deny all;
           allow 10.0.0.0/8;  # Allow internal monitoring systems
           allow 127.0.0.1;
       }
   }

   # Redirect HTTP to HTTPS
   server {
       listen 80;
       server_name api.your-domain.com;
       return 301 https://$host$request_uri;
   }
   ```

4. Start the application with proper process control
   ```bash
   gunicorn -c gunicorn_config.py app.main:app
   ```

### 4. Redis Configuration for Production

1. Install Redis with proper security settings
   ```bash
   apt-get install redis-server
   ```

2. Configure Redis for production in `/etc/redis/redis.conf`
   ```
   # Basic configuration
   bind 127.0.0.1
   port 6379
   daemonize yes
   pidfile /var/run/redis/redis-server.pid
   loglevel notice
   logfile /var/log/redis/redis-server.log

   # Security
   requirepass <strong-password>
   rename-command FLUSHALL ""
   rename-command FLUSHDB ""
   rename-command CONFIG ""
   rename-command SHUTDOWN ""

   # Memory management
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   maxmemory-samples 5

   # Persistence
   save 900 1
   save 300 10
   save 60 10000
   rdbcompression yes
   dbfilename dump.rdb
   dir /var/lib/redis

   # Performance
   tcp-backlog 511
   tcp-keepalive 300
   timeout 0
   
   # Replication (if using replicas)
   # slaveof <masterip> <masterport>
   # masterauth <master-password>
   ```

3. Enable and start Redis service
   ```bash
   systemctl enable redis-server
   systemctl start redis-server
   ```

### 5. Monitoring Setup

#### Prometheus Metrics

Ensure Prometheus is configured to scrape the `/metrics` endpoint for real-time monitoring of:

- API request rates and latencies
- Cache hit ratios
- Error rates
- Resource utilization

Sample Prometheus scrape configuration:
```yaml
scrape_configs:
  - job_name: 'credit-cashflow-engine'
    scrape_interval: 15s
    scheme: https
    tls_config:
      insecure_skip_verify: false
    basic_auth:
      username: 'prometheus'
      password: 'your-monitoring-password'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api.your-domain.com']
```

#### Health Checks

Set up periodic health checks to the `/health` endpoint to monitor system status.

Example health check script:
```bash
#!/bin/bash
HEALTH_URL="https://api.your-domain.com/health"
AUTH_TOKEN="your-auth-token"

response=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $AUTH_TOKEN" $HEALTH_URL)

if [ $response -eq 200 ]; then
  echo "Health check passed"
  exit 0
else
  echo "Health check failed with status $response"
  exit 1
fi
```

#### Alerting

Configure alerts for key metrics:
- Response time > 500ms for 95th percentile
- Error rate > 1%
- Cache hit ratio < 50%
- API availability < 99.9%

## Performance Tuning

### Redis Cache Optimization

1. Adjust TTL values based on data volatility:
   - Stable data (e.g., static asset pools): 24 hours
   - Dynamic data: 1-4 hours
   
2. Configure Redis memory limits and eviction policy:
   ```
   maxmemory 4gb
   maxmemory-policy volatile-lru
   ```

3. Implement proper connection pooling:
   ```python
   # Redis connection pool configuration
   pool = redis.ConnectionPool(
       host=settings.REDIS_HOST,
       port=settings.REDIS_PORT,
       password=settings.REDIS_PASSWORD,
       db=settings.REDIS_DATABASE,
       max_connections=settings.REDIS_MAX_CONNECTIONS,
       socket_timeout=settings.REDIS_TIMEOUT,
       socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
       retry_on_timeout=True,
       ssl=settings.REDIS_SSL,
       ssl_cert_reqs=None
   )
   ```

### API Optimization

1. Implement pagination for large result sets
2. Use compression for response payloads
3. Implement request timeouts for long-running operations
4. Use background processing for complex analyses

## Maintenance Procedures

### Backups

1. Configure daily Redis backup:
   ```bash
   # Redis backup script
   redis-cli -a <password> --rdb /backup/redis/redis-backup-$(date +%Y%m%d).rdb
   ```

2. Rotate backup files:
   ```bash
   find /backup/redis/ -name "redis-backup-*.rdb" -mtime +7 -delete
   ```

### Updates and Upgrades

1. Set up a proper CI/CD pipeline with:
   - Automated testing
   - Staging environment deployment
   - Blue/green production deployment
   - Rollback capability

2. Zero-downtime deployment procedure:
   ```bash
   # Deploy new version behind load balancer
   cd /opt/credit-cashflow-engine
   git pull
   source venv/bin/activate
   pip install -r requirements/production.txt
   
   # Start new instances
   systemctl start credit-cashflow-engine-new.service
   
   # Health check
   curl -f https://api-new.your-domain.com/health
   
   # Switch traffic
   # (Load balancer configuration update)
   
   # Stop old instances
   systemctl stop credit-cashflow-engine-old.service
   ```

### Monitoring and Troubleshooting

1. Set up log aggregation with ELK stack or similar
2. Create dashboards for key metrics:
   - Response times by endpoint
   - Error rates by type
   - Cache hit/miss ratios
   - Resource utilization

3. Implement distributed tracing for request flow visualization

## Security Considerations

1. Implement IP-based rate limiting
2. Configure proper JWT token expiration
3. Implement IP allow-listing for admin functions
4. Regular security audits and penetration testing
5. Secrets rotation policy

## Disaster Recovery

1. Document recovery procedures for:
   - Service interruption
   - Data corruption
   - Redis failure
   - Complete system failure

2. Implement regional failover for critical services

3. Test recovery procedures quarterly with simulated failures

## Compliance Requirements

Ensure the deployment meets:
- Data residency requirements
- Encryption standards (data at rest and in transit)
- Audit logging for compliance tracking
- Data retention policies

## Conclusion

This deployment guide ensures a production-ready implementation of the Asset Classes API with robust performance, security, and operational considerations. Follow all steps carefully to create a resilient and maintainable system that can handle production workloads reliably.

For any issues during deployment, consult the troubleshooting section in the main documentation or contact the platform team.
