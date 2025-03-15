# AbsBox Service Documentation

## Overview

The `AbsBoxService` provides a comprehensive interface for financial calculations using the AbsBox library. This service consolidates functionality from previous implementations with significant improvements in error handling, caching, metrics collection, and economic factor adjustments.

## Features

- **Unified Interface**: Consolidates functionality from previous implementations
- **Robust Error Handling**: Consistent error handling with detailed context
- **Efficient Caching**: Automatic caching of calculation results with TTL management
- **Performance Metrics**: Built-in metrics collection for monitoring performance
- **Economic Factor Integration**: Sophisticated adjustment of financial calculations based on economic indicators
- **Asynchronous API**: Fully asynchronous API for improved performance

## Installation

The service is part of the core financial calculation engine. No additional installation is required.

## Service Initialization

```python
from app.services.absbox_unified_service import AbsBoxService
from app.core.cache_service import CacheService
from app.core.monitoring import PrometheusMetrics

# Create dependencies
cache_service = CacheService()
metrics_service = PrometheusMetrics()

# Initialize the service
absbox_service = AbsBoxService(
    cache_service=cache_service,
    metrics_service=metrics_service,
    absbox_url="https://absbox-api.example.com"  # Optional remote URL
)
```

## Core Financial Calculations

### Calculate Net Present Value (NPV)

```python
async def calculate_npv_example():
    cashflows = [
        {"amount": -1000, "date": "2025-01-01"},
        {"amount": 200, "date": "2025-07-01"},
        {"amount": 200, "date": "2026-01-01"},
        {"amount": 200, "date": "2026-07-01"},
        {"amount": 200, "date": "2027-01-01"},
        {"amount": 200, "date": "2027-07-01"},
        {"amount": 200, "date": "2028-01-01"}
    ]
    
    discount_rate = 0.05  # 5%
    
    # Basic NPV calculation
    npv = await absbox_service.calculate_npv(
        cashflows=cashflows,
        discount_rate=discount_rate
    )
    
    print(f"NPV: {npv}")
    
    # NPV with economic factors
    economic_factors = {
        "inflation_rate": 0.03,
        "market_rate": 0.06,
        "unemployment_rate": 0.04,
        "gdp_growth": 0.02
    }
    
    npv_with_factors = await absbox_service.calculate_npv(
        cashflows=cashflows,
        discount_rate=discount_rate,
        economic_factors=economic_factors
    )
    
    print(f"NPV with economic factors: {npv_with_factors}")
```

### Calculate Internal Rate of Return (IRR)

```python
async def calculate_irr_example():
    cashflows = [
        {"amount": -1000, "date": "2025-01-01"},
        {"amount": 300, "date": "2026-01-01"},
        {"amount": 400, "date": "2027-01-01"},
        {"amount": 500, "date": "2028-01-01"}
    ]
    
    # Basic IRR calculation
    irr = await absbox_service.calculate_irr(
        cashflows=cashflows,
        initial_guess=0.1  # Initial guess of 10%
    )
    
    print(f"IRR: {irr * 100}%")
    
    # IRR with economic factors
    economic_factors = {
        "inflation_rate": 0.03,
        "market_rate": 0.06
    }
    
    irr_with_factors = await absbox_service.calculate_irr(
        cashflows=cashflows,
        initial_guess=0.1,
        economic_factors=economic_factors
    )
    
    print(f"IRR with economic factors: {irr_with_factors * 100}%")
```

### Calculate Duration

```python
async def calculate_duration_example():
    cashflows = [
        {"amount": -1000, "date": "2025-01-01"},
        {"amount": 100, "date": "2026-01-01"},
        {"amount": 100, "date": "2027-01-01"},
        {"amount": 100, "date": "2028-01-01"},
        {"amount": 100, "date": "2029-01-01"},
        {"amount": 1100, "date": "2030-01-01"}
    ]
    
    discount_rate = 0.05  # 5%
    
    # Calculate duration
    duration_results = await absbox_service.calculate_duration(
        cashflows=cashflows,
        discount_rate=discount_rate
    )
    
    print(f"Macaulay Duration: {duration_results['macaulay_duration']} years")
    print(f"Modified Duration: {duration_results['modified_duration']} years")
    
    # With economic factors
    economic_factors = {
        "inflation_rate": 0.04,
        "market_rate": 0.055
    }
    
    duration_with_factors = await absbox_service.calculate_duration(
        cashflows=cashflows,
        discount_rate=discount_rate,
        economic_factors=economic_factors
    )
    
    print(f"Macaulay Duration with factors: {duration_with_factors['macaulay_duration']} years")
    print(f"Modified Duration with factors: {duration_with_factors['modified_duration']} years")
```

## Stress Testing

The service provides comprehensive stress testing for cashflow projections:

```python
async def run_stress_test_example():
    cashflows = [
        {"amount": -1000, "date": "2025-01-01"},
        {"amount": 200, "date": "2026-01-01"},
        {"amount": 200, "date": "2027-01-01"},
        {"amount": 200, "date": "2028-01-01"},
        {"amount": 200, "date": "2029-01-01"},
        {"amount": 200, "date": "2030-01-01"}
    ]
    
    discount_rate = 0.05  # 5%
    
    # Define scenarios for stress testing
    scenarios = [
        {
            "name": "Base Case",
            "economic_factors": {
                "inflation_rate": 0.02,
                "market_rate": 0.05,
                "unemployment_rate": 0.04,
                "gdp_growth": 0.025
            }
        },
        {
            "name": "Rising Inflation",
            "economic_factors": {
                "inflation_rate": 0.04,
                "market_rate": 0.06,
                "unemployment_rate": 0.05,
                "gdp_growth": 0.02
            }
        },
        {
            "name": "Recession",
            "economic_factors": {
                "inflation_rate": 0.015,
                "market_rate": 0.035,
                "unemployment_rate": 0.08,
                "gdp_growth": -0.01
            }
        },
        {
            "name": "Economic Boom",
            "economic_factors": {
                "inflation_rate": 0.025,
                "market_rate": 0.065,
                "unemployment_rate": 0.03,
                "gdp_growth": 0.035
            }
        }
    ]
    
    # Run stress test
    stress_test_results = await absbox_service.run_stress_test(
        cashflows=cashflows,
        discount_rate=discount_rate,
        scenarios=scenarios
    )
    
    # Print results summary
    print("Stress Test Results:")
    print(f"Number of scenarios: {stress_test_results['summary']['scenario_count']}")
    print(f"Minimum NPV: {stress_test_results['summary']['min_npv']}")
    print(f"Maximum NPV: {stress_test_results['summary']['max_npv']}")
    print(f"Average NPV: {stress_test_results['summary']['average_npv']}")
    
    # Print detailed scenario results
    for name, result in stress_test_results['scenarios'].items():
        print(f"\nScenario: {name}")
        print(f"  NPV: {result['npv']}")
        print(f"  Discount Rate: {result['discount_rate']}")
        print(f"  Change from Base: {result['change_from_base']}")
```

## Economic Factors

The service supports incorporating economic factors into financial calculations. These factors allow for more realistic projections that account for macroeconomic conditions.

### Supported Economic Factors

| Factor | Description | Impact |
|--------|-------------|--------|
| `market_rate` | Current market interest rate | Blended with base rate (70% market, 30% base) |
| `inflation_rate` | Current inflation rate | Adds premium for inflation above 2% target |
| `unemployment_rate` | Current unemployment rate | Adds risk premium for high unemployment |
| `gdp_growth` | GDP growth rate | Adds risk premium for low/negative growth |

### How Factors Affect Calculations

The service uses economic factors to adjust discount rates as follows:

1. **Market Rate Adjustment**: Blends the base discount rate with the current market rate
2. **Inflation Premium**: Adds a premium when inflation exceeds the target rate
3. **Risk Premium**: Adds risk premium based on unemployment and GDP growth indicators

## Error Handling

The service provides consistent error handling with detailed context:

```python
try:
    npv = await absbox_service.calculate_npv(cashflows, discount_rate)
except CalculationError as e:
    print(f"Calculation failed: {e}")
    print(f"Context: {e.context}")
    print(f"Original cause: {e.cause}")
```

## Service Status

You can check the status of the AbsBox service:

```python
status = absbox_service.get_service_status()
print(f"Service status: {status['status']}")
print(f"Initialized: {status['initialized']}")
print(f"Error count: {status['error_count']}")
```

## Integration with Other Services

The AbsBox service is designed to integrate seamlessly with other services in the financial calculation engine:

```python
from app.services.financial_dashboard import FinancialDashboard

# Create dashboard with AbsBox service
dashboard = FinancialDashboard(absbox_service=absbox_service)

# Generate financial dashboard
report = await dashboard.generate_cashflow_report(investment_id="INV123")
```

## Performance Considerations

- The service uses asynchronous operations for improved performance
- Results are automatically cached to reduce calculation overhead
- Economic factor adjustments add minimal overhead to calculations
- Stress testing multiple scenarios is optimized for parallel processing

## Migrating from Previous Implementations

If you were using the previous `AbsBoxService` or `AbsBoxServiceEnhanced` implementations, you'll need to update your code to use the unified service:

### Previous Implementation:

```python
from app.services.absbox_service import AbsBoxService

absbox = AbsBoxService(hastructure_url="https://example.com")
npv = absbox.calculate_npv(cashflows, discount_rate)
```

### New Implementation:

```python
from app.services.absbox_unified_service import AbsBoxService
from app.core.cache_service import CacheService

cache_service = CacheService()
absbox = AbsBoxService(cache_service=cache_service, absbox_url="https://example.com")
npv = await absbox.calculate_npv(cashflows, discount_rate)
```

Key differences:
1. The service now requires a cache service
2. All calculation methods are now asynchronous
3. The unified service provides more detailed error information
4. Economic factors can be incorporated into calculations
