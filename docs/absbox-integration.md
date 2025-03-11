# AbsBox Integration Guide

This document provides details on the integration of the AbsBox library into the credit-cashflow-engine backend.

## Overview

AbsBox is a structured finance modeling library that enables the creation, analysis, and pricing of securitization and structured credit products. It provides tools for modeling loan portfolios, structuring deals, and performing cashflow projections and analysis.

The integration in this project provides:

1. A service layer for interacting with AbsBox functionality
2. REST API endpoints for structured finance calculations
3. Docker-based deployment of the Hastructure calculation engine
4. Integration with the existing caching and monitoring systems

## Components

### 1. AbsBox Service

The `AbsBoxService` class in `app/services/absbox_service.py` provides methods for:

- Creating loan pools from configuration
- Building waterfall structures
- Defining assumptions for analysis
- Running deal analysis and scenario testing
- Health checks for the service

### 2. API Endpoints

The API endpoints are defined in `app/api/v1/structured_products.py` and include:

- `/api/v1/structured-products/health` - Check health of the AbsBox and Hastructure engine
- `/api/v1/structured-products/deals/analyze` - Analyze a structured finance deal
- `/api/v1/structured-products/deals/scenarios` - Run scenario analysis on a deal structure
- `/api/v1/structured-products/metrics` - Get metrics about structured finance calculations

### 3. Data Models

The data models for structured finance products are defined in `app/models/structured_products.py` and include:

- `StructuredDealRequest` - Input model for deal analysis
- `StructuredDealResponse` - Response model with cashflows and metrics
- `LoanPoolConfig` - Configuration for loan pools
- `WaterfallConfig` - Configuration for deal waterfall
- `ScenarioConfig` - Configuration for scenario analysis

### 4. Hastructure Engine

The Hastructure calculation engine is a high-performance engine for AbsBox calculations. It is deployed as a Docker container as defined in `docker-compose.prod.yml`.

## Usage Examples

### Basic Deal Analysis

```python
from app.services.absbox_service import AbsBoxService
from app.models.structured_products import (
    StructuredDealRequest,
    LoanPoolConfig,
    LoanConfig,
    WaterfallConfig,
    AccountConfig,
    BondConfig,
    WaterfallAction,
    ScenarioConfig
)
from datetime import date

# Create loan pool configuration
pool_config = LoanPoolConfig(
    pool_name="Example Pool",
    loans=[
        LoanConfig(
            balance=100000.0,
            rate=0.05,
            term=360,
            start_date=date(2023, 1, 1),
            rate_type="fixed"
        ),
        LoanConfig(
            balance=150000.0,
            rate=0.06,
            term=240, 
            start_date=date(2023, 1, 1),
            rate_type="fixed"
        )
    ]
)

# Create waterfall configuration
waterfall_config = WaterfallConfig(
    start_date=date(2023, 2, 1),
    accounts=[
        AccountConfig(name="Collection", initial_balance=0.0),
        AccountConfig(name="Reserve", initial_balance=10000.0)
    ],
    bonds=[
        BondConfig(name="ClassA", balance=200000.0, rate=0.04),
        BondConfig(name="ClassB", balance=50000.0, rate=0.06)
    ],
    actions=[
        WaterfallAction(source="CollectedInterest", target="ClassA", amount="Interest"),
        WaterfallAction(source="CollectedInterest", target="ClassB", amount="Interest"),
        WaterfallAction(source="CollectedPrincipal", target="ClassA", amount="OutstandingPrincipal"),
        WaterfallAction(source="CollectedPrincipal", target="ClassB", amount="OutstandingPrincipal")
    ]
)

# Create scenario configuration
scenario_config = ScenarioConfig(
    name="Base Case",
    default_curve=DefaultCurveConfig(
        vector=[0.01, 0.02, 0.02, 0.015, 0.01]
    )
)

# Create deal request
deal_request = StructuredDealRequest(
    deal_name="Example Deal",
    pool=pool_config,
    waterfall=waterfall_config,
    scenario=scenario_config
)

# Analyze the deal
service = AbsBoxService()
result = service.analyze_deal(deal_request)

# Print results
print(f"Deal analysis completed in {result.execution_time:.2f} seconds")
print(f"Number of bond cashflows: {len(result.bond_cashflows)}")
print(f"Number of pool cashflows: {len(result.pool_cashflows)}")
```

### Scenario Analysis

```python
from app.services.absbox_service import AbsBoxService
from app.models.structured_products import ScenarioConfig, DefaultCurveConfig

# Create multiple scenarios
scenarios = [
    ScenarioConfig(
        name="Base Case",
        default_curve=DefaultCurveConfig(
            vector=[0.01, 0.02, 0.02, 0.015, 0.01]
        )
    ),
    ScenarioConfig(
        name="Stressed Case",
        default_curve=DefaultCurveConfig(
            vector=[0.03, 0.05, 0.06, 0.04, 0.03]
        )
    ),
    ScenarioConfig(
        name="Recovery Case",
        default_curve=DefaultCurveConfig(
            vector=[0.02, 0.01, 0.005, 0.0, 0.0]
        )
    )
]

# Run scenario analysis
service = AbsBoxService()
results = service.run_scenario_analysis(deal_request, scenarios)

# Print results
for result in results:
    print(f"Scenario: {result.scenario_name}")
    print(f"Status: {result.status}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    if result.bond_metrics.get("ClassA", {}).get("irr"):
        print(f"Class A IRR: {result.bond_metrics['ClassA']['irr']:.4f}")
    print("---")
```

## Docker Configuration

The Hastructure engine is configured in `docker-compose.prod.yml`:

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

## Environment Variables

The following environment variables are used for the AbsBox integration:

```
HASTRUCTURE_URL=http://hastructure:8081
HASTRUCTURE_TIMEOUT=300
HASTRUCTURE_MAX_POOL_SIZE=10
```

## Testing

Unit tests for the AbsBox integration are available in:
- `tests/unit/test_absbox_service.py` - Tests for the AbsBox service
- `tests/api/test_structured_products_api.py` - Tests for the API endpoints

To run the tests:

```bash
pytest tests/unit/test_absbox_service.py
pytest tests/api/test_structured_products_api.py
```

## References

- [AbsBox GitHub Repository](https://github.com/yellowbean/AbsBox)
- [AbsBox Documentation](https://absbox-doc.readthedocs.io/en/latest/)
- [Hastructure Engine](https://github.com/yellowbean/hastructure)
