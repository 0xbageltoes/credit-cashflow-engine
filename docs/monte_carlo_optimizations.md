# Monte Carlo Simulation Optimization

This document describes the optimized Monte Carlo simulation implementation in the Credit Cashflow Engine. The optimizations focus on proper correlation modeling, memory efficiency, and parallel processing for improved performance in production environments.

## Key Optimizations

### 1. Correlation-Aware Scenario Generation

The optimized implementation properly models correlations between economic factors using Cholesky decomposition. This ensures that the generated scenarios reflect realistic relationships between factors, such as:

- Correlation between interest rates and inflation
- Correlation between GDP growth and unemployment rates
- Correlation between market rates and housing prices

Features:
- Validates and adjusts correlation matrices to ensure they are positive definite
- Supports customized volatility parameters for each economic factor
- Handles edge cases like ensuring non-negative values for appropriate factors

### 2. Memory-Efficient Processing

The implementation uses batched processing to limit memory usage, making it suitable for large simulations with thousands or millions of scenarios:

- Processes scenarios in configurable batch sizes
- Avoids storing all intermediate calculation results in memory
- Uses NumPy arrays for efficient storage and computation
- Maintains only necessary data in memory at any given time

### 3. Parallel Processing

The implementation leverages parallel processing through:

- Asynchronous I/O operations with asyncio
- Thread pooling for CPU-bound calculations
- Proper task distribution to maximize hardware utilization

### 4. Robust Error Handling

The optimized implementation includes comprehensive error handling:

- Graceful handling of invalid correlation matrices
- Detailed logging and error reporting
- Recovery mechanisms for failed simulations
- Appropriate exception propagation

## Usage

### API Endpoints

The optimized Monte Carlo simulation is available through the following API endpoints:

1. **Run Optimized Simulation**:
   - Endpoint: `/api/monte-carlo/optimized`
   - Method: POST
   - Description: Runs a Monte Carlo simulation with the optimized implementation

2. **Generate Correlated Scenarios**:
   - Endpoint: `/api/monte-carlo/optimized/scenarios`
   - Method: POST
   - Description: Generates correlated economic scenarios without running the full simulation

### Example Request

```json
{
  "name": "Mortgage Portfolio Simulation",
  "description": "Monte Carlo simulation for mortgage portfolio with correlation modeling",
  "loan_data": {
    "principal": 500000,
    "interest_rate": 0.045,
    "term_months": 360,
    "origination_date": "2024-01-01",
    "loan_type": "mortgage",
    "credit_score": 720,
    "ltv_ratio": 0.8,
    "dti_ratio": 0.36
  },
  "base_economic_factors": {
    "market_rate": 0.04,
    "inflation_rate": 0.02,
    "unemployment_rate": 0.042,
    "gdp_growth": 0.025,
    "house_price_index_growth": 0.03
  },
  "volatilities": {
    "market_rate": 0.1,
    "inflation_rate": 0.05,
    "unemployment_rate": 0.15,
    "gdp_growth": 0.2,
    "house_price_index_growth": 0.12
  },
  "correlation_matrix": [
    [1.0, 0.3, 0.2, -0.4, 0.5],
    [0.3, 1.0, 0.6, 0.1, 0.2],
    [0.2, 0.6, 1.0, 0.3, -0.1],
    [-0.4, 0.1, 0.3, 1.0, 0.0],
    [0.5, 0.2, -0.1, 0.0, 1.0]
  ],
  "num_scenarios": 10000,
  "batch_size": 500,
  "seed": 42
}
```

### Example Response

```json
{
  "status": "completed",
  "message": "Optimized simulation completed successfully",
  "simulation_id": "opt_user123_abc45678",
  "result": {
    "summary": {
      "npv_mean": 487650.23,
      "npv_std": 25432.12,
      "npv_min": 398765.45,
      "npv_max": 562341.89,
      "irr_mean": 0.0412,
      "duration_mean": 8.76,
      "var_95": 445234.56,
      "var_99": 423456.78,
      "expected_shortfall_95": 432123.45
    },
    "execution_info": {
      "num_scenarios": 10000,
      "batch_size": 500,
      "execution_time_seconds": 12.45
    }
  }
}
```

## Integration with Existing Systems

The optimized Monte Carlo implementation integrates with:

1. **Redis Caching**: Results can be cached for improved performance
2. **Celery Task Queue**: Supports asynchronous processing for long-running simulations
3. **Supabase Database**: Stores simulation results and metadata
4. **WebSocket Notifications**: Provides real-time updates on simulation progress

## Performance Considerations

When running large simulations, consider the following:

1. **Batch Size**: Adjust the batch size based on available memory and CPU resources
   - Smaller batches use less memory but may have higher overhead
   - Larger batches are more efficient but require more memory

2. **Number of Workers**: Set the appropriate number of worker processes based on hardware
   - The system automatically adapts to available CPU cores
   - Can be manually configured via the `max_workers` parameter

3. **Memory Usage**: Monitor memory usage for very large simulations
   - A simulation with 100,000 scenarios might require 1-2GB of memory
   - Use the batch processing feature to manage memory constraints

## Future Enhancements

Planned enhancements for the Monte Carlo optimization include:

1. **GPU Acceleration**: Support for GPU-based calculations for even faster processing
2. **Distributed Processing**: Support for distributed calculations across multiple nodes
3. **Advanced Correlation Models**: Additional correlation models beyond Cholesky decomposition
4. **Adaptive Batch Sizing**: Automatic adjustment of batch sizes based on system resources
