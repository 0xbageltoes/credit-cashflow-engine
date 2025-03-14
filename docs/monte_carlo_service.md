# Monte Carlo Simulation Service Documentation

## Overview

The Monte Carlo Simulation Service is a high-performance, production-ready component that provides comprehensive financial simulation capabilities for the credit cashflow engine. It enables users to model the uncertainty in financial projections by simulating multiple future scenarios based on probability distributions of key variables.

## Key Features

- **Comprehensive Statistical Outputs**: Calculate mean, median, standard deviation, percentiles, skewness, kurtosis, Value-at-Risk (VaR), and Conditional Value-at-Risk (CVaR).
- **Economic Factor Integration**: Incorporate macroeconomic factors such as inflation, unemployment, housing prices, and interest rate environments.
- **Robust Caching**: Uses Redis for efficient caching with proper TTL management, error handling, and fallbacks.
- **Multi-variable Simulation**: Model multiple uncertain variables with different probability distributions.
- **Correlation Support**: Specify correlations between variables to model real-world dependencies.
- **Production Monitoring**: Comprehensive metrics for performance tracking, error rates, and resource utilization.
- **Deterministic Testing**: Reproduce exact simulation results with seed-based random number generation.
- **Asynchronous Processing**: Non-blocking API for high concurrency environments.

## Usage

### Basic Simulation

```python
from app.services.monte_carlo_service import MonteCarloSimulationService
from app.models.monte_carlo import (
    MonteCarloSimulationRequest,
    Variable,
    DistributionType,
    DistributionParameters
)

# Initialize service
monte_carlo_service = MonteCarloSimulationService()

# Create simulation request
request = MonteCarloSimulationRequest(
    name="Loan Portfolio Simulation",
    description="Simulating default and recovery rates for loan portfolio",
    num_simulations=1000,
    variables=[
        Variable(
            name="default_rate",
            distribution=DistributionType.BETA,
            parameters=DistributionParameters(
                alpha=2.0,
                beta=8.0
            )
        ),
        Variable(
            name="recovery_rate",
            distribution=DistributionType.BETA,
            parameters=DistributionParameters(
                alpha=5.0,
                beta=2.0
            )
        )
    ],
    asset_class="consumer_loans",
    asset_parameters={
        "cashflows": cashflows,  # Your cashflow projections
        "discount_rate": 0.05
    },
    projection_months=60,
    include_detailed_paths=True,
    random_seed=42  # For reproducibility
)

# Run simulation
result = await monte_carlo_service.run_simulation(
    request=request,
    user_id="user_123"
)

# Access results
print(f"NPV Mean: {result.npv_stats.mean}")
print(f"NPV 95th Percentile: {result.npv_stats.percentiles['95']}")
```

### With Economic Factors

```python
from app.models.analytics import EconomicFactors

# Add economic factors to request
request.economic_factors = EconomicFactors(
    inflation_rate=0.03,
    unemployment_rate=0.05,
    housing_price_index=0.02,
    interest_rate_environment="neutral"
)

# Run enhanced simulation
result = await monte_carlo_service.run_enhanced_simulation(
    request=request,
    user_id="user_123"
)

# Access economic factor effects
print(result.economic_factor_effects)
```

## Configuration

The service can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_CACHE` | Enable/disable caching | `True` |
| `CACHE_TTL_SECONDS` | Cache time-to-live in seconds | `3600` (1 hour) |
| `REDIS_HOST` | Redis server hostname | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_PASSWORD` | Redis server password | `None` |
| `REDIS_DB` | Redis database number | `0` |
| `REDIS_SOCKET_TIMEOUT` | Socket timeout in seconds | `5.0` |
| `REDIS_SOCKET_CONNECT_TIMEOUT` | Connection timeout in seconds | `5.0` |
| `REDIS_RETRY_MAX_ATTEMPTS` | Maximum retry attempts for Redis operations | `3` |
| `REDIS_MAX_CONNECTIONS` | Maximum Redis connections | `10` |

## Error Handling

The service implements comprehensive error handling to ensure resilience in production environments:

- **Graceful Degradation**: Falls back to direct calculation when Redis is unavailable.
- **Detailed Error Messages**: Provides context-rich error messages for debugging.
- **Error Categorization**: Categorizes errors as user errors, system errors, or unknown errors.
- **No Exceptions**: All errors are captured and returned in the result object.

## Monitoring

The service integrates with Prometheus for comprehensive monitoring:

- **Performance Metrics**:
  - Simulation duration
  - Number of iterations
  - Cache hit/miss rates
  - Error rates
  
- **Resource Utilization**:
  - Memory usage
  - CPU usage
  - Redis connections

- **Business Metrics**:
  - Simulation complexity
  - Asset class distribution
  - User activity

## Testing

The service includes comprehensive test coverage:

- **Unit Tests**: Test individual components in isolation.
- **Integration Tests**: Validate integration with Redis.
- **Performance Tests**: Ensure service meets performance criteria.
- **Edge Cases**: Verify handling of boundary conditions.

To run the tests:

```bash
# Unit tests
pytest tests/test_monte_carlo_service.py

# Integration tests (requires Redis)
pytest tests/test_monte_carlo_service.py -m integration
```

## Scaling Considerations

When deploying in production, consider:

1. **Redis Scaling**: Use a clustered Redis configuration for high availability.
2. **Horizontal Scaling**: The service is stateless and can be scaled horizontally.
3. **Memory Consumption**: Large simulations consume significant memory; monitor and scale accordingly.
4. **Calculation Distribution**: Consider distributing large calculations across multiple workers.

## Typical Workflow

1. **Data Preparation**: Prepare cashflow projections and variable distributions.
2. **Request Creation**: Create a simulation request with appropriate parameters.
3. **Run Simulation**: Execute the simulation with `run_simulation` or `run_enhanced_simulation`.
4. **Results Analysis**: Analyze statistical outputs and percentiles.
5. **Decision Making**: Use results to inform investment or risk management decisions.

## Best Practices

1. **Use Appropriate Iterations**: More iterations provide more accurate results but consume more resources. Start with 1,000 for testing, 10,000+ for production.
2. **Set Seed for Reproducibility**: Always set a random seed for reproducible results.
3. **Calibrate Distributions**: Use historical data to calibrate distribution parameters.
4. **Monitor Memory**: Large simulations with many paths can consume significant memory.
5. **Enable Caching**: Use caching for repeated simulations with identical inputs.
6. **Correlate Variables**: Real-world variables are often correlated; model this with the correlation matrix.

## Technical Details

### Available Distributions

- **NORMAL**: Normal/Gaussian distribution (mean, std_dev)
- **BETA**: Beta distribution (alpha, beta)
- **UNIFORM**: Uniform distribution (min, max)
- **TRIANGULAR**: Triangular distribution (min, mode, max)
- **LOGNORMAL**: Log-normal distribution (mean, sigma)
- **EXPONENTIAL**: Exponential distribution (lambda)
- **GAMMA**: Gamma distribution (shape, scale)

### Statistical Outputs

- **Mean**: Arithmetic average of all simulations
- **Median**: Middle value of all simulations
- **Standard Deviation**: Measure of dispersion
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of "tailedness"
- **Min/Max**: Minimum and maximum values
- **Percentiles**: Values at specified percentiles
- **VaR**: Value-at-Risk at specified confidence level
- **CVaR**: Conditional Value-at-Risk (Expected Shortfall)

### Economic Factor Effects

Economic factors modify the underlying cashflow projections:

- **Inflation Rate**: Discount future cashflows by compounded inflation
- **Unemployment Rate**: Increase default rates when unemployment is high
- **Housing Price Index**: Improve recovery rates when housing prices rise
- **Interest Rate Environment**: Modify prepayment behavior based on rate environment

## Support

For questions or issues, contact the Credit Cashflow Engine support team or open an issue in the repository.
