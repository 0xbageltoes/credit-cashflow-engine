# Specialized Asset Class Handlers

This directory contains production-ready implementations of specialized asset class handlers for the Credit Cashflow Engine. Each handler provides comprehensive analytics for its respective asset class, leveraging the AbsBox engine with robust error handling and detailed analytics.

## Overview

The following asset class handlers are implemented:

1. **Consumer Credit Handler** - Analyzes consumer credit assets (credit cards, auto loans, personal loans, etc.)
2. **Commercial Loan Handler** - Analyzes commercial real estate and business loans
3. **CLO/CDO Handler** - Analyzes structured products like Collateralized Loan Obligations and Collateralized Debt Obligations

Each handler is designed for production use with:
- Comprehensive error handling
- Detailed logging for monitoring and troubleshooting
- Proper integration with Redis caching
- Performance monitoring
- Stress testing capabilities

## Usage

Each asset handler follows the same interface pattern and can be used in a similar manner:

```python
# Example for Commercial Loan Handler
from app.services.asset_handlers.commercial_loan import CommercialLoanHandler
from app.models.asset_classes import AssetPoolAnalysisRequest, AssetPool, CommercialLoan

# Create the handler
handler = CommercialLoanHandler()

# Create analysis request
request = AssetPoolAnalysisRequest(
    pool=AssetPool(
        pool_name="Commercial Loan Portfolio",
        assets=[...],  # List of commercial loan assets
        cut_off_date=date.today(),
    ),
    analysis_date=date.today(),
    discount_rate=0.06,
    include_cashflows=True,
    include_stress_tests=True
)

# Run analysis
result = handler.analyze_pool(request)
```

## API Endpoints

The handlers are exposed through dedicated API endpoints for each asset class:

- `/api/v1/specialized-assets/consumer-credit/...` - Consumer Credit endpoints
- `/api/v1/specialized-assets/commercial-loans/...` - Commercial Loan endpoints
- `/api/v1/specialized-assets/clo-cdo/...` - CLO/CDO endpoints

## Technical Design

### Common Handler Structure

All handlers follow a similar structure:

1. **Initialization**: Each handler is initialized with an AbsBox service instance
2. **analyze_pool()**: Main entry point that processes a pool of assets
3. **_create_absbox_*()**: Helper methods to convert models to AbsBox objects
4. **_calculate_cashflows()**: Calculates cashflow projections for the assets
5. **_calculate_metrics()**: Calculates key metrics like NPV, duration, etc.
6. **_run_*_analytics()**: Specialized analytics for the specific asset class
7. **_run_stress_tests()**: Implements stress scenarios for the asset class

### Error Handling

All handlers include robust error handling with:
- Detailed error messages and types
- Proper exception chaining to maintain context
- Graceful fallbacks where appropriate
- Comprehensive logging

### Dependency Injection

Handlers are available through dependency injection in the API layer:
- `get_consumer_credit_handler()`
- `get_commercial_loan_handler()`
- `get_clo_cdo_handler()`

## Redis Cache Integration

All handlers leverage the enhanced AbsBox service which includes Redis caching with:
- Proper TTL settings
- Graceful fallbacks when Redis is unavailable
- Configurable caching behavior

## Contributing

When extending or modifying these handlers:
1. Maintain the consistent error handling pattern
2. Follow the established pattern for handling assets
3. Ensure proper logging at appropriate levels
4. Maintain compatibility with the AbsBox service
5. Add proper tests for new functionality
