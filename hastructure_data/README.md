# Hastructure Engine Data Directory

This directory contains data files used by the Hastructure calculation engine for AbsBox.

## Purpose

The Hastructure engine is a high-performance calculation engine for structured finance modeling using AbsBox. This directory is mounted as a volume in the Hastructure Docker container to persist data between container restarts.

## Contents

When the Hastructure engine is running, this directory may contain:

- Cache files for calculations
- Temporary results from complex modeling operations
- Log files (if enabled)
- Configuration files

## Docker Configuration

The Docker configuration for Hastructure is defined in `docker-compose.prod.yml`:

```yaml
hastructure:
  image: yellowbean/hastructure:latest
  restart: always
  ports:
    - "8081:8081"
  environment:
    - MODE=Production
    - MAX_POOL_SIZE=10
    - TIMEOUT_SECS=300
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s
  volumes:
    - ./hastructure_data:/app/data
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
```

## Health Check

The Hastructure engine exposes a health check endpoint at:

```
http://hastructure:8081/health
```

You can test this endpoint manually with:

```bash
curl http://localhost:8081/health
```

## Environment Variables

The Hastructure engine configuration can be controlled through environment variables:

- `MODE`: Operation mode (Production, Development)
- `MAX_POOL_SIZE`: Maximum thread pool size for parallel calculations
- `TIMEOUT_SECS`: Calculation timeout in seconds

## Troubleshooting

If you encounter issues with the Hastructure engine:

1. Check the container logs:
   ```bash
   docker logs <container_id>
   ```

2. Verify the health endpoint is responsive:
   ```bash
   curl http://localhost:8081/health
   ```

3. Ensure the container has adequate resources (CPU/memory)

4. Check if the volume is properly mounted:
   ```bash
   docker inspect <container_id> | grep hastructure_data
   ```

## Resources

- [AbsBox Documentation](https://absbox-doc.readthedocs.io/en/latest/)
- [Hastructure GitHub Repository](https://github.com/yellowbean/hastructure)
