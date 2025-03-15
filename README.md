# Credit Cashflow Engine

A high-performance Python-based microservice for cash flow forecasting, built with FastAPI and integrated with Supabase.

## Features

- Cash flow forecasting with multiple scenarios
- Advanced financial analytics using absbox (NPV, IRR, Duration, Convexity)
- Comprehensive asset class coverage (Residential Mortgages, Auto Loans, Consumer Credit, Commercial Loans, CLOs/CDOs)
- Scenario management and templating capabilities
- Risk metrics and sensitivity analysis
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
REDIS_CACHE_TTL=3600  # Default TTL in seconds
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
- GET `/api/v1/forecasts/` - List all forecast runs
- GET `/api/v1/forecasts/{forecast_id}` - Get a specific forecast run
- GET `/api/v1/forecasts/{forecast_id}/projections` - Get projections for a forecast

### Specialized Asset Class Endpoints

- POST `/api/v1/specialized-assets/consumer-credit/analyze` - Analyze consumer credit assets
- POST `/api/v1/specialized-assets/commercial-loans/analyze` - Analyze commercial loans
- POST `/api/v1/specialized-assets/clo-cdo/analyze` - Analyze CLO/CDO structured products
- GET `/api/v1/specialized-assets/consumer-credit/loan-types` - Get consumer credit loan types
- GET `/api/v1/specialized-assets/commercial-loans/property-types` - Get commercial property types
- POST `/api/v1/specialized-assets/clo-cdo/tranche-analysis` - Analyze CLO/CDO tranches

### Scenario Management

- POST `/api/v1/scenarios/` - Create a new scenario
- GET `/api/v1/scenarios/` - List all saved scenarios
- GET `/api/v1/scenarios/{scenario_id}` - Get a specific scenario
- PUT `/api/v1/scenarios/{scenario_id}` - Update a scenario
- DELETE `/api/v1/scenarios/{scenario_id}` - Delete a scenario
- POST `/api/v1/scenarios/{scenario_id}/run` - Run a saved scenario

### System Endpoints

- GET `/health` - Service health check
- GET `/metrics` - Prometheus metrics
- WS `/ws/updates` - WebSocket for real-time updates

## Advanced Financial Analytics

The engine provides advanced financial analytics using the powerful absbox library:

### Core Analytics Metrics
- **Net Present Value (NPV)**: Present value of all cash flows discounted at a specified rate
- **Internal Rate of Return (IRR)**: Discount rate that makes NPV equal to zero
- **Duration**: Weighted average time until cash flows are received
- **Convexity**: Measure of the curvature in the relationship between bond prices and bond yields
- **Weighted Average Life (WAL)**: Average time until principal is paid back
- **Yield Metrics**: Multiple yield measurements depending on security type

### Risk Analysis
- **Spread Calculations**: Various spread measurements (Z-spread, OAS, etc.)
- **Sensitivity Analysis**: Impact of rate/yield changes on portfolio value
- **Stress Testing**: Performance under various economic scenarios

### Monte Carlo Simulation
- Distribution analysis of potential outcomes
- Percentile measurements (5th, 50th, 95th)
- Volatility metrics and statistical analysis

### Specialized Asset Class Analytics

#### Consumer Credit
- Delinquency forecasting and roll rate analysis
- Vintage curve analytics and cohort performance
- Credit score migration modeling
- Loss forecasting with economic factor correlations

#### Commercial Loans
- Property type analysis and concentration risk
- Debt service coverage ratio (DSCR) stress testing
- Loan-to-value (LTV) sensitivity analysis
- Default correlation and recovery rate modeling

#### CLO/CDO Analysis
- Tranche waterfall modeling and cash flow allocation
- Overcollateralization and interest coverage tests
- Collateral quality test simulation
- Default and recovery simulations at instrument and portfolio level

## Architecture

The project structure follows domain-driven design principles:
```
app/
├── api/           # API routes and endpoints
│   ├── v1/        # API version 1
│   └── endpoints/ # Specialized asset class endpoints
├── core/          # Core business logic and config
│   ├── auth.py    # Authentication handling
│   ├── config.py  # Configuration settings
│   ├── security.py # Security utilities
│   └── monitoring.py # Metrics and monitoring
├── database/      # Database operations
├── models/        # Pydantic models and database schemas
│   └── specialized_assets.py # Specialized asset class models
├── services/      # Business logic services
│   ├── absbox_service.py  # Financial analytics using absbox
│   ├── redis_service.py   # Caching layer
│   └── asset_handlers/    # Specialized asset class handlers
│       ├── consumer_credit.py # Consumer credit analysis
│       ├── commercial_loan.py # Commercial loan analysis
│       └── clo_cdo.py     # CLO/CDO analysis
├── tasks/         # Celery tasks
└── utils/         # Utility functions
```

## Core Components

### Unified Redis Cache Implementation

The credit cashflow engine includes a production-ready unified Redis cache implementation with the following features:

#### Advanced Redis Configuration

- **Comprehensive configuration options**: Connection pooling, timeouts, retry policies, SSL support
- **Environment-based configuration**: Automatically loads configuration from environment variables
- **Compression support**: Configurable compression for large objects to reduce memory usage
- **Advanced key generation**: Consistent key generation based on function arguments

#### CacheService Features

- **Memory caching layer**: In-memory caching for frequently accessed items
- **Synchronous and asynchronous APIs**: Full support for both sync and async code
- **Circuit breaker pattern**: Prevents cascading failures when Redis is unavailable
- **Comprehensive error handling**: Graceful fallbacks and detailed error reporting
- **Task-specific methods**: Specialized methods for common caching tasks
- **Serialization flexibility**: Support for various serialization formats and custom serializers
- **Health checking**: Built-in health check functionality with detailed diagnostics
- **Cache statistics**: Monitoring of cache hit rates, sizes, and performance metrics

#### Production-Ready Features

- **Prometheus metrics integration**: Track cache performance, hit rates, and error rates
- **Sentry error reporting**: Detailed error tracking with context
- **Graceful degradation**: Fall back to direct computation when Redis is unavailable
- **Compatibility layer**: Support for gradual migration from legacy Redis implementations
- **Comprehensive logging**: Detailed logging for troubleshooting and monitoring

#### Usage Example

```python
from app.core.cache_service import CacheService, RedisConfig, cached

# Initialize cache service with custom configuration
cache = CacheService(RedisConfig(
    url="redis://localhost:6379/0",
    default_ttl=3600,
    default_compression=True,
    socket_timeout=5.0,
    socket_connect_timeout=3.0,
    socket_keepalive=True,
    retry_on_timeout=True,
    health_check_interval=30
))

# Using the @cached decorator with async functions
@cached(ttl=3600)
async def fetch_data(id: str, cache: CacheService):
    # Expensive operation
    return {"id": id, "data": "expensive result"}

# Using the @cached decorator with sync functions
@cached(ttl=1800)
def compute_result(x: int, y: int, cache_service: CacheService):
    # Expensive operation
    return x * y
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
