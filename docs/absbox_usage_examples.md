# AbsBox Usage Examples

This document provides detailed examples of how to use the AbsBox integration within the Credit Cashflow Engine.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Creating Loan Pools](#creating-loan-pools)
3. [Defining Waterfall Structures](#defining-waterfall-structures)
4. [Running Scenario Analysis](#running-scenario-analysis)
5. [Working with API Endpoints](#working-with-api-endpoints)
6. [Advanced Modeling Techniques](#advanced-modeling-techniques)
7. [Troubleshooting](#troubleshooting)

## Basic Usage

The AbsBox service can be used both programmatically within Python code and via the API endpoints.

### Programmatic Usage

```python
from app.services.absbox_service import AbsBoxService

# Initialize the service
service = AbsBoxService()

# Check if the service is healthy
health_status = service.health_check()
print(health_status)
```

### Via API Endpoints

```bash
# Check health status
curl -X GET "http://localhost:8000/api/v1/structured-products/health"

# Analyze a deal (POST with JSON body)
curl -X POST "http://localhost:8000/api/v1/structured-products/deals/analyze" \
     -H "Content-Type: application/json" \
     -d @deal_request.json
```

## Creating Loan Pools

AbsBox supports various types of loan pools. Here are examples of how to create different types of pools:

### Fixed Rate Mortgage Pool

```python
from datetime import date, timedelta
from app.models.structured_products import LoanPoolConfig, LoanConfig

# Create a pool of fixed-rate mortgages
pool_config = LoanPoolConfig(
    pool_name="Fixed Rate Mortgage Pool",
    loans=[
        LoanConfig(
            balance=200000.0,
            rate=0.05,  # 5% interest rate
            term=360,   # 30-year loan (360 months)
            start_date=date.today() - timedelta(days=30),
            rate_type="fixed",
            payment_frequency="Monthly"
        ),
        LoanConfig(
            balance=150000.0,
            rate=0.045,  # 4.5% interest rate
            term=360,    # 30-year loan
            start_date=date.today() - timedelta(days=60),
            rate_type="fixed",
            payment_frequency="Monthly"
        )
    ]
)
```

### Floating Rate Loan Pool

```python
from app.models.structured_products import LoanPoolConfig, LoanConfig, RateCurveConfig

# Create a pool of floating-rate loans
floating_pool_config = LoanPoolConfig(
    pool_name="Floating Rate Loan Pool",
    loans=[
        LoanConfig(
            balance=1000000.0,
            rate=0.04,            # Initial rate
            term=120,             # 10-year loan
            start_date=date.today() - timedelta(days=90),
            rate_type="floating",
            payment_frequency="Monthly",
            floating_rate_details={
                "index": "LIBOR",
                "spread": 0.02,   # 200 basis points over index
                "reset_frequency": "Quarterly",
                "cap": 0.08,      # 8% cap
                "floor": 0.03     # 3% floor
            }
        )
    ],
    rate_curves=[
        RateCurveConfig(
            name="LIBOR",
            rates=[0.02, 0.022, 0.025, 0.028, 0.03],  # Forward curve
            dates=[
                date.today() + timedelta(days=i*90) for i in range(5)
            ]
        )
    ]
)
```

### Commercial Real Estate Pool

```python
from app.models.structured_products import LoanPoolConfig, LoanConfig

# Create a commercial real estate loan pool
cre_pool_config = LoanPoolConfig(
    pool_name="CRE Loan Pool",
    loans=[
        LoanConfig(
            balance=5000000.0,
            rate=0.055,           # 5.5% rate
            term=120,             # 10-year loan
            start_date=date.today() - timedelta(days=180),
            rate_type="fixed",
            payment_frequency="Monthly",
            amortization_term=360,  # 30-year amortization with balloon payment
            property_type="Office",
            balloon_payment=True
        )
    ]
)
```

## Defining Waterfall Structures

AbsBox supports a variety of waterfall structures for securitizations. Here are some examples:

### Simple Sequential Pay Structure

```python
from app.models.structured_products import WaterfallConfig, AccountConfig, BondConfig, WaterfallAction

# Calculate the total pool balance first
total_pool_balance = sum(loan.balance for loan in pool_config.loans)

# Create a simple sequential pay waterfall
waterfall_config = WaterfallConfig(
    start_date=date.today(),
    accounts=[
        AccountConfig(name="Collection", initial_balance=0.0),
        AccountConfig(name="Reserve", initial_balance=total_pool_balance * 0.02)  # 2% reserve
    ],
    bonds=[
        # Senior tranche (80% of pool)
        BondConfig(name="ClassA", balance=total_pool_balance * 0.8, rate=0.04),
        # Mezzanine tranche (15% of pool)
        BondConfig(name="ClassB", balance=total_pool_balance * 0.15, rate=0.06),
        # Junior/Equity tranche (5% of pool)
        BondConfig(name="ClassC", balance=total_pool_balance * 0.05, rate=0.08)
    ],
    actions=[
        # Pay senior interest first
        WaterfallAction(source="CollectedInterest", target="ClassA", amount="Interest"),
        # Pay mezzanine interest
        WaterfallAction(source="CollectedInterest", target="ClassB", amount="Interest"),
        # Pay junior interest
        WaterfallAction(source="CollectedInterest", target="ClassC", amount="Interest"),
        # Senior principal (sequential)
        WaterfallAction(source="CollectedPrincipal", target="ClassA", amount="OutstandingPrincipal"),
        # Mezzanine principal (paid after Senior is fully paid)
        WaterfallAction(source="CollectedPrincipal", target="ClassB", amount="OutstandingPrincipal"),
        # Junior principal (paid last)
        WaterfallAction(source="CollectedPrincipal", target="ClassC", amount="OutstandingPrincipal")
    ]
)
```

### Pro-Rata Pay Structure with Triggers

```python
from app.models.structured_products import WaterfallConfig, AccountConfig, BondConfig, WaterfallAction, TriggerConfig

# Create a pro-rata pay waterfall with triggers
pro_rata_waterfall = WaterfallConfig(
    start_date=date.today(),
    accounts=[
        AccountConfig(name="Collection", initial_balance=0.0),
        AccountConfig(name="Reserve", initial_balance=total_pool_balance * 0.03)
    ],
    bonds=[
        BondConfig(name="ClassA", balance=total_pool_balance * 0.75, rate=0.04),
        BondConfig(name="ClassB", balance=total_pool_balance * 0.15, rate=0.05),
        BondConfig(name="ClassC", balance=total_pool_balance * 0.10, rate=0.07)
    ],
    actions=[
        # Interest waterfall (sequential)
        WaterfallAction(source="CollectedInterest", target="ClassA", amount="Interest"),
        WaterfallAction(source="CollectedInterest", target="ClassB", amount="Interest"),
        WaterfallAction(source="CollectedInterest", target="ClassC", amount="Interest"),
        
        # Principal waterfall (pro-rata with triggers)
        WaterfallAction(source="CollectedPrincipal", target="ClassA", 
                       amount="ProRataShare", condition="BeforeTrigger"),
        WaterfallAction(source="CollectedPrincipal", target="ClassB", 
                       amount="ProRataShare", condition="BeforeTrigger"),
        WaterfallAction(source="CollectedPrincipal", target="ClassC", 
                       amount="ProRataShare", condition="BeforeTrigger"),
        
        # After trigger, switch to sequential pay
        WaterfallAction(source="CollectedPrincipal", target="ClassA", 
                       amount="OutstandingPrincipal", condition="AfterTrigger"),
        WaterfallAction(source="CollectedPrincipal", target="ClassB", 
                       amount="OutstandingPrincipal", condition="AfterTrigger"),
        WaterfallAction(source="CollectedPrincipal", target="ClassC", 
                       amount="OutstandingPrincipal", condition="AfterTrigger")
    ],
    triggers=[
        TriggerConfig(
            name="DefermentTrigger",
            condition="CumulativeDefaultRate > 0.05",  # 5% cumulative default rate
            description="Switch to sequential pay if cumulative defaults exceed 5%"
        )
    ]
)
```

## Running Scenario Analysis

AbsBox allows for various types of scenario analysis to stress test your securitization structures:

### Creating Scenarios

```python
from app.models.structured_products import ScenarioConfig, DefaultCurveConfig, PrepaymentCurveConfig, RateCurveConfig

# Base case scenario
base_scenario = ScenarioConfig(
    name="Base Case",
    default_curve=DefaultCurveConfig(
        vector=[0.01, 0.015, 0.02, 0.015, 0.01]  # Annual default rates
    ),
    prepayment_curve=PrepaymentCurveConfig(
        vector=[0.05, 0.07, 0.1, 0.12, 0.14]     # Annual prepayment rates (CPR)
    )
)

# Stress scenario - high defaults
stress_default_scenario = ScenarioConfig(
    name="High Default Stress",
    default_curve=DefaultCurveConfig(
        vector=[0.03, 0.05, 0.07, 0.06, 0.04]    # Higher default rates
    ),
    prepayment_curve=PrepaymentCurveConfig(
        vector=[0.03, 0.04, 0.05, 0.06, 0.07]    # Lower prepayment rates in stress
    )
)

# Stress scenario - rate shock
rate_shock_scenario = ScenarioConfig(
    name="Interest Rate Shock",
    default_curve=DefaultCurveConfig(
        vector=[0.01, 0.02, 0.025, 0.02, 0.015]  # Slightly elevated defaults
    ),
    prepayment_curve=PrepaymentCurveConfig(
        vector=[0.02, 0.03, 0.03, 0.04, 0.04]    # Lower prepayments due to rate shock
    ),
    rate_curves=[
        RateCurveConfig(
            name="LIBOR",
            rates=[0.03, 0.04, 0.05, 0.055, 0.06],  # Rising rate environment
            dates=[date.today() + timedelta(days=i*90) for i in range(5)]
        )
    ]
)
```

### Running Multiple Scenarios

```python
from app.services.absbox_service import AbsBoxService
from app.models.structured_products import StructuredDealRequest

# Create the base deal request
deal_request = StructuredDealRequest(
    deal_name="Example Securitization",
    pool=pool_config,
    waterfall=waterfall_config,
    scenario=base_scenario  # Base scenario
)

# Initialize the service
service = AbsBoxService()

# Run multiple scenarios
scenarios = [base_scenario, stress_default_scenario, rate_shock_scenario]
scenario_results = service.run_scenario_analysis(deal_request, scenarios)

# Process and compare the results
for result in scenario_results:
    print(f"Scenario: {result.scenario_name}")
    print(f"Status: {result.status}")
    
    if result.status == "success":
        # Access bond metrics
        for bond_name, metrics in result.bond_metrics.items():
            print(f"  {bond_name}:")
            print(f"    IRR: {metrics.get('irr', 'N/A')}")
            print(f"    Loss: {metrics.get('loss', 'N/A')}")
            print(f"    Duration: {metrics.get('duration', 'N/A')}")
```

## Working with API Endpoints

The AbsBox integration exposes several API endpoints for structured finance analytics:

### Health Check

```bash
# Check service health
curl -X GET "http://localhost:8000/api/v1/structured-products/health"
```

Example response:
```json
{
  "status": "healthy",
  "version": "0.9.5",
  "engine": {
    "type": "hastructure",
    "url": "http://hastructure:8081",
    "status": "connected"
  },
  "cache": {
    "enabled": true,
    "provider": "redis",
    "status": "connected"
  }
}
```

### Analyze Deal

```bash
# Request body saved in deal_request.json
curl -X POST "http://localhost:8000/api/v1/structured-products/deals/analyze" \
     -H "Content-Type: application/json" \
     -d @deal_request.json
```

Example deal_request.json:
```json
{
  "deal_name": "API Example Deal",
  "pool": {
    "pool_name": "Mortgage Pool",
    "loans": [
      {
        "balance": 200000.0,
        "rate": 0.05,
        "term": 360,
        "start_date": "2023-01-15",
        "rate_type": "fixed",
        "payment_frequency": "Monthly"
      },
      {
        "balance": 150000.0,
        "rate": 0.045,
        "term": 360,
        "start_date": "2023-01-01",
        "rate_type": "fixed",
        "payment_frequency": "Monthly"
      }
    ]
  },
  "waterfall": {
    "start_date": "2023-02-01",
    "accounts": [
      {
        "name": "Collection",
        "initial_balance": 0.0
      },
      {
        "name": "Reserve",
        "initial_balance": 7000.0
      }
    ],
    "bonds": [
      {
        "name": "ClassA",
        "balance": 280000.0,
        "rate": 0.04
      },
      {
        "name": "ClassB",
        "balance": 70000.0,
        "rate": 0.06
      }
    ],
    "actions": [
      {
        "source": "CollectedInterest",
        "target": "ClassA",
        "amount": "Interest"
      },
      {
        "source": "CollectedInterest",
        "target": "ClassB",
        "amount": "Interest"
      },
      {
        "source": "CollectedPrincipal",
        "target": "ClassA",
        "amount": "OutstandingPrincipal"
      },
      {
        "source": "CollectedPrincipal",
        "target": "ClassB",
        "amount": "OutstandingPrincipal"
      }
    ]
  },
  "scenario": {
    "name": "Base Case",
    "default_curve": {
      "vector": [0.01, 0.015, 0.02, 0.015, 0.01]
    },
    "prepayment_curve": {
      "vector": [0.05, 0.07, 0.1, 0.12, 0.14]
    }
  }
}
```

### Run Scenario Analysis

```bash
# Request body saved in scenario_request.json
curl -X POST "http://localhost:8000/api/v1/structured-products/deals/scenarios" \
     -H "Content-Type: application/json" \
     -d @scenario_request.json
```

Example scenario_request.json:
```json
{
  "deal": {
    "deal_name": "Scenario Analysis Example",
    "pool": {
      "pool_name": "Mortgage Pool",
      "loans": [
        {
          "balance": 200000.0,
          "rate": 0.05,
          "term": 360,
          "start_date": "2023-01-15",
          "rate_type": "fixed"
        }
      ]
    },
    "waterfall": {
      "start_date": "2023-02-01",
      "accounts": [
        {
          "name": "Reserve",
          "initial_balance": 5000.0
        }
      ],
      "bonds": [
        {
          "name": "Senior",
          "balance": 160000.0,
          "rate": 0.04
        },
        {
          "name": "Junior",
          "balance": 40000.0,
          "rate": 0.07
        }
      ],
      "actions": [
        {
          "source": "CollectedInterest",
          "target": "Senior",
          "amount": "Interest"
        },
        {
          "source": "CollectedInterest",
          "target": "Junior",
          "amount": "Interest"
        },
        {
          "source": "CollectedPrincipal",
          "target": "Senior",
          "amount": "OutstandingPrincipal"
        },
        {
          "source": "CollectedPrincipal",
          "target": "Junior",
          "amount": "OutstandingPrincipal"
        }
      ]
    }
  },
  "scenarios": [
    {
      "name": "Base Case",
      "default_curve": {
        "vector": [0.01, 0.015, 0.02, 0.015, 0.01]
      }
    },
    {
      "name": "High Default",
      "default_curve": {
        "vector": [0.03, 0.05, 0.07, 0.06, 0.04]
      }
    },
    {
      "name": "Low Prepayment",
      "default_curve": {
        "vector": [0.01, 0.015, 0.02, 0.015, 0.01]
      },
      "prepayment_curve": {
        "vector": [0.02, 0.03, 0.03, 0.04, 0.04]
      }
    }
  ]
}
```

## Advanced Modeling Techniques

### Custom Assumptions

```python
from app.models.structured_products import ScenarioConfig, CustomAssumptionConfig

# Create a scenario with custom assumptions
custom_scenario = ScenarioConfig(
    name="Custom Assumptions",
    custom_assumptions=[
        CustomAssumptionConfig(
            name="RecoveryLag",
            value=6,        # 6 months recovery lag
            type="integer"
        ),
        CustomAssumptionConfig(
            name="RecoveryRate",
            value=0.65,     # 65% recovery rate
            type="float"
        ),
        CustomAssumptionConfig(
            name="DelinquencyRate",
            value=0.02,     # 2% delinquency rate
            type="float"
        )
    ]
)
```

### Custom Waterfalls

AbsBox allows for custom waterfall rules beyond standard sequential and pro-rata structures:

```python
from app.models.structured_products import WaterfallConfig, WaterfallAction

# Create a custom waterfall with excess spread capture
custom_waterfall = WaterfallConfig(
    # ... accounts and bonds defined as usual
    actions=[
        # Interest waterfall
        WaterfallAction(source="CollectedInterest", target="ServicerFee", amount="FixedAmount", 
                       parameters={"amount": 1000}),
        WaterfallAction(source="CollectedInterest", target="ClassA", amount="Interest"),
        WaterfallAction(source="CollectedInterest", target="ClassB", amount="Interest"),
        
        # Excess spread capture
        WaterfallAction(source="CollectedInterest", target="ReserveFund", amount="RemainingAmount"),
        
        # Release from reserve fund (when above target)
        WaterfallAction(source="ReserveFund", target="ExcessSpreadAccount", amount="ExcessOverTarget",
                       parameters={"target": 10000}),
        
        # Principal waterfall
        WaterfallAction(source="CollectedPrincipal", target="ClassA", amount="OutstandingPrincipal"),
        WaterfallAction(source="CollectedPrincipal", target="ClassB", amount="OutstandingPrincipal"),
        
        # Cover shortfalls from reserve
        WaterfallAction(source="ReserveFund", target="ClassA", amount="InterestShortfall"),
        WaterfallAction(source="ReserveFund", target="ClassB", amount="InterestShortfall")
    ]
)
```

## Troubleshooting

### Common Issues

1. **Connection Problems with Hastructure Engine**

   If you're having trouble connecting to the Hastructure engine, check:
   
   - The Hastructure container is running
   - The HASTRUCTURE_URL in your .env file is correct
   - Network connectivity between your application and the Hastructure service

   You can test the connection manually:
   
   ```bash
   curl http://localhost:8081/health
   ```

2. **Performance Issues**

   If you're experiencing slow performance:
   
   - Check that Redis caching is enabled (ABSBOX_CACHE_ENABLED=true)
   - Increase the thread pool size for parallel calculations (HASTRUCTURE_MAX_POOL_SIZE)
   - Consider simplifying complex deals or scenarios
   - Check the Hastructure container's resource allocation

3. **Error Handling**

   Common errors from the AbsBox service:
   
   ```python
   # Check error response details
   try:
       result = service.analyze_deal(deal_request)
   except Exception as e:
       print(f"Error: {str(e)}")
   ```

4. **Debug Mode**

   Enable debug logs to get more information:
   
   ```python
   from app.services.absbox_service import AbsBoxService
   
   # Enable debug logs
   service = AbsBoxService(debug=True)
   result = service.analyze_deal(deal_request)
   ```

### Getting Help

For additional help with AbsBox:

- Refer to the AbsBox documentation: [AbsBox Documentation](https://absbox-doc.readthedocs.io/en/latest/)
- Check the AbsBox GitHub repository: [yellowbean/AbsBox](https://github.com/yellowbean/AbsBox)
- For Hastructure engine issues: [yellowbean/Hastructure](https://github.com/yellowbean/hastructure)
