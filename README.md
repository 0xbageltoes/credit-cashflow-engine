# Credit Cashflow Engine

A high-performance Python-based microservice for cash flow forecasting, built with FastAPI and integrated with Supabase.

## Features

- Cash flow forecasting with multiple scenarios
- Supabase JWT authentication
- Rate limiting and request tracking
- High-performance calculations using NumPy/Pandas
- Redis caching for frequent computations
- Docker deployment ready
- Prometheus metrics and Grafana dashboards
- Celery task queue for async processing
- WebSockets for real-time updates
- Production-grade security and error handling

## Setup

### Development Environment

1. Create a `.env` file with your Supabase credentials:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_JWT_SECRET=your_jwt_secret
UPSTASH_REDIS_HOST=your_redis_host
UPSTASH_REDIS_PORT=your_redis_port
UPSTASH_REDIS_PASSWORD=your_redis_password
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server (development mode):
```bash
uvicorn app.main:app --reload
```

### Using Docker Compose (Development)

```bash
docker-compose up
```

### Production Deployment

For detailed production deployment instructions, see the [Deployment Guide](docs/DEPLOYMENT.md).

1. Configure production environment variables (see [Configuration Guide](docs/CONFIGURATION.md))

2. Build and run using Docker Compose:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. For cloud deployment (AWS ECS):
```bash
# Deploy using GitHub Actions CI/CD pipeline
git push origin main

# Or manually deploy
aws ecs update-service --cluster your-cluster-name --service credit-cashflow-engine --force-new-deployment
```

## API Endpoints

### Main Endpoints

- POST `/api/v1/cashflow/forecast` - Run a cash flow forecast
- POST `/api/v1/cashflow/scenario/save` - Save a forecasting scenario
- GET `/api/v1/cashflow/scenario/{id}` - Load a specific scenario
- GET `/api/v1/cashflow/scenarios` - List all saved scenarios
- GET `/api/v1/cashflow/history` - View forecast history

### System Endpoints

- GET `/health` - Service health check
- GET `/metrics` - Prometheus metrics
- WS `/ws/updates` - WebSocket for real-time updates

## Architecture

The project structure follows domain-driven design principles:
```
app/
├── api/           # API routes and endpoints
│   └── v1/        # API version 1
├── core/          # Core business logic and config
│   ├── auth.py    # Authentication handling
│   ├── config.py  # Configuration settings
│   ├── security.py # Security utilities
│   └── monitoring.py # Metrics and monitoring
├── database/      # Database operations
├── models/        # Pydantic models and database schemas
├── services/      # Business logic services
├── tasks/         # Celery tasks
└── utils/         # Utility functions
```

## Monitoring and Observability

- Prometheus metrics available at `/metrics`
- Grafana dashboard for visualization
- Structured JSON logging
- Sentry integration for error tracking
- Health check endpoint for uptime monitoring

## Load Testing

A Locust file is included for load testing the API:

```bash
locust -f locustfile.py --host=http://localhost:8000
```

## Security Features

- JWT authentication with Supabase
- Rate limiting to prevent abuse
- Security headers (CSP, XSS Protection, etc.)
- Input validation and sanitization
- Error tracking and monitoring

## Documentation

- API documentation: Available at `/docs` (Swagger UI) or `/redoc` (ReDoc)
- Deployment guide: [DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Configuration guide: [CONFIGURATION.md](docs/CONFIGURATION.md)

## Testing

### Running Tests Locally

1. Set up test environment:
```bash
# Copy example environment to test environment
cp .env.example .env.test

# Add required test variables
echo "ENV=testing" >> .env.test
```

2. Run minimal tests:
```bash
# Run only the minimal tests to verify configuration
python -m pytest tests/test_minimal.py tests/test_config.py -v
```

3. Run all tests with coverage:
```bash
# Run all tests with coverage reporting
python -m pytest tests/ -v --cov=app --cov-report=term --cov-report=html
```

4. Using custom test runner:
```bash
# This script sets up environment and runs selected tests
python scripts/run_tests_with_path.py
```

### Test Dependencies

To run the tests, you'll need these additional packages:
```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock numpy-financial pandas python-jose
```

### CI/CD Pipeline

The CI/CD pipeline automatically runs tests on pull requests and pushes to main/development branches. It includes:

1. **Linting**: Static code analysis using flake8
2. **Security Scanning**: Dependency scanning with safety
3. **Testing**: Unit and integration tests with pytest
4. **Load Testing**: Performance testing with Locust
5. **Build & Push**: Docker image creation and publishing
6. **Deployment**: Automatic deployment to staging and production environments

The CI/CD workflow is defined in `.github/workflows/ci-cd.yml`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

[MIT License](LICENSE)
