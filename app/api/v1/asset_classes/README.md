# Asset Classes API

## Overview

The Asset Classes API provides a comprehensive set of endpoints for analyzing different asset classes in structured finance, including:

- Residential mortgages
- Auto loans
- Consumer credit
- Commercial loans
- CLOs/CDOs

This module is designed for production use with extensive error handling, validation, caching via Redis, and performance optimization.

## API Endpoints

### `GET /api/v1/asset-classes/supported`

Returns a list of asset classes supported by the system.

**Response:**
```json
[
  "residential_mortgage",
  "auto_loan",
  "consumer_credit",
  "commercial_loan",
  "clo_cdo"
]
```

### `POST /api/v1/asset-classes/analyze`

Analyzes a pool of assets with specific parameters and returns detailed analytics.

**Request Parameters:**
- `pool`: Asset pool object containing assets to analyze
- `analysis_date`: Date for analysis (defaults to current date if not provided)
- `discount_rate`: Rate for NPV calculations
- `include_cashflows`: Whether to include cashflow projections (default: true)
- `include_metrics`: Whether to include performance metrics (default: true)
- `include_stress_tests`: Whether to run stress scenarios (default: false)
- `use_cache`: Query parameter to enable/disable caching (default: true)

**Example Request:**
```json
{
  "pool": {
    "pool_name": "Residential Mortgage Pool 2025-A",
    "pool_description": "Fixed-rate prime residential mortgages",
    "cut_off_date": "2025-03-01",
    "assets": [
      {
        "asset_class": "residential_mortgage",
        "balance": 250000,
        "rate": 0.0475,
        "term_months": 360,
        "origination_date": "2024-12-15",
        "rate_type": "fixed",
        "payment_frequency": "monthly",
        "status": "current",
        "property_type": "single_family",
        "ltv_ratio": 0.75,
        "lien_position": 1
      }
    ]
  },
  "analysis_date": "2025-03-15",
  "discount_rate": 0.05,
  "include_cashflows": true,
  "include_metrics": true
}
```

**Example Response:**
```json
{
  "request_id": "f8d7a982-f22d-4e34-9d7b-c04e435e6276",
  "pool_name": "Residential Mortgage Pool 2025-A",
  "analysis_date": "2025-03-15",
  "execution_time": 0.453,
  "status": "success",
  "metrics": {
    "total_principal": 250000,
    "total_interest": 213750.45,
    "total_cashflow": 463750.45,
    "npv": 242500.23,
    "irr": 0.0475,
    "duration": 8.3,
    "weighted_average_life": 9.1
  },
  "cashflows": [
    {
      "period": 1,
      "date": "2025-04-15",
      "scheduled_principal": 375.82,
      "scheduled_interest": 989.58,
      "prepayment": 0,
      "default": 0,
      "recovery": 0,
      "loss": 0,
      "balance": 249624.18
    }
  ],
  "analytics": {
    "credit_metrics": {
      "weighted_fico": 720,
      "delinquency_rate": 0.02
    },
    "prepayment_metrics": {
      "cpr": 0.05,
      "smm": 0.004
    }
  },
  "cache_hit": false
}
```

### `POST /api/v1/asset-classes/validate-pool`

Validates an asset pool for consistency and completeness before analysis.

**Request:**
Asset pool object to validate

**Response:**
```json
{
  "valid": true,
  "warnings": [
    "Mixed asset classes in pool: residential_mortgage, auto_loan"
  ],
  "errors": []
}
```

## Error Handling

The API provides detailed error responses with appropriate HTTP status codes:

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

Example error response:
```json
{
  "error": "Invalid asset data: negative balance detected",
  "detail": "Asset at index 2 has a negative balance of -50000"
}
```

## Caching

The API utilizes Redis for caching analysis results to improve performance. Cache keys are generated based on:

- Asset pool details
- Analysis parameters
- User ID

Cache can be bypassed by setting `use_cache=false` in the query string.

## Performance Considerations

- Analysis of large asset pools (>1000 assets) may take longer
- Use the `include_cashflows` parameter judiciously for large pools
- Consider disabling stress tests for initial quick analysis

## Authentication

All endpoints require authentication. Include a valid JWT token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Implementation Notes

This module uses specialized handlers for different asset classes via a factory pattern:

- `ResidentialMortgageHandler`: Specialized for residential mortgage analysis
- `AutoLoanHandler`: Optimized for auto loan analysis
- Generic handling for other asset classes

## Monitoring

All API calls are logged and monitored with:

- Request tracking
- Performance metrics
- Error tracking
- Redis cache hit rate

## Examples

### Analyzing a Mixed Asset Pool

```python
import requests
import json

url = "https://api.example.com/api/v1/asset-classes/analyze"
headers = {
    "Authorization": "Bearer your_token",
    "Content-Type": "application/json"
}

data = {
    "pool": {
        "pool_name": "Mixed Asset Pool",
        "cut_off_date": "2025-03-01",
        "assets": [
            # Residential mortgage
            {
                "asset_class": "residential_mortgage",
                "balance": 200000,
                "rate": 0.045,
                "term_months": 360,
                "origination_date": "2024-09-15"
            },
            # Auto loan
            {
                "asset_class": "auto_loan", 
                "balance": 25000,
                "rate": 0.065,
                "term_months": 60,
                "origination_date": "2025-01-01"
            }
        ]
    },
    "analysis_date": "2025-03-15",
    "discount_rate": 0.05
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()
```
