# AbsBox Visualization Guide

This guide explains how to use the AbsBox visualization tools to analyze structured finance results. The visualization tools provided in this package help you create interactive charts and dashboards to better understand the performance of structured deals.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Visualization Components](#visualization-components)
   - [Bond Cashflows](#bond-cashflows)
   - [Bond Balances](#bond-balances)
   - [Pool Performance](#pool-performance)
   - [Bond Metrics](#bond-metrics)
   - [Comprehensive Dashboard](#comprehensive-dashboard)
4. [Interpreting Results](#interpreting-results)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Installation

To use the AbsBox visualizer, you need to install the required dependencies:

```bash
pip install -r app/utils/absbox_visualization_requirements.txt
```

The visualization tools depend on:
- pandas and numpy for data manipulation
- matplotlib and plotly for chart generation
- dash (optional) for interactive web dashboards

## Quick Start

Here's a simple example to get started with visualizing AbsBox results:

```python
from app.utils.absbox_visualizer import AbsBoxVisualizer
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

# Create an AbsBox service
service = AbsBoxServiceEnhanced()

# Run an analysis
result = service.analyze_deal(deal_request)

# Create a visualizer
visualizer = AbsBoxVisualizer(output_dir="./viz_output")

# Generate a comprehensive dashboard
visualizer.create_dashboard(result, output_html="deal_dashboard.html", show=True)
```

Alternatively, if you have a saved result file:

```python
import json
from app.utils.absbox_visualizer import AbsBoxVisualizer

# Load results from a file
with open("result.json", "r") as f:
    result = json.load(f)

# Create a visualizer and generate a dashboard
visualizer = AbsBoxVisualizer()
visualizer.create_dashboard(result, output_html="deal_dashboard.html", show=True)
```

For a complete example, check out the demo script at `examples/absbox_visualization_demo.py`.

## Visualization Components

The AbsBox visualizer provides several visualization components for different aspects of structured finance analysis.

### Bond Cashflows

The bond cashflow visualization shows the projected principal and interest payments for each bond in the deal.

```python
visualizer.plot_bond_cashflows(
    result,
    plot_type="stacked_area",  # Options: "stacked_area", "line", "bar"
    save_file="bond_cashflows.html",
    show=True
)
```

**Interpretation:**
- **Principal payments**: These represent scheduled and unscheduled (prepayment) payments of principal.
- **Interest payments**: These represent scheduled interest payments based on the bond rate.
- **Stacked view**: Shows how cash is distributed across bonds in each period.

### Bond Balances

The bond balance visualization shows the projected outstanding balance for each bond over time.

```python
visualizer.plot_bond_balances(
    result,
    save_file="bond_balances.html",
    show=True
)
```

**Interpretation:**
- **Balance curves**: Show how quickly each bond is paid down over time.
- **Factor curves**: (If available) Show the percentage of the original balance remaining.
- **Stepdowns**: Visible as accelerated declines in the balance curves, indicating when subordinate bonds begin receiving principal.

### Pool Performance

The pool performance visualization shows key metrics about the underlying loan pool.

```python
visualizer.plot_pool_performance(
    result,
    save_file="pool_performance.html",
    show=True
)
```

**Interpretation:**
- **Remaining balance**: Shows the projected balance of the loan pool over time.
- **Scheduled principal**: Shows the amount of principal scheduled to be paid in each period.
- **Cumulative default rate**: Shows the cumulative percentage of the original loan pool that has defaulted.
- **Cumulative prepayment rate**: Shows the cumulative percentage of the original loan pool that has prepaid.

### Bond Metrics

The bond metrics visualization shows comparative performance metrics for each bond.

```python
visualizer.plot_bond_metrics(
    result,
    save_file="bond_metrics.html",
    show=True
)
```

**Interpretation:**
- **Yield**: The internal rate of return for each bond.
- **WAL (Weighted Average Life)**: The average time it takes for the bond to be repaid, weighted by the amount of principal repaid.
- **Duration**: A measure of the bond's sensitivity to interest rate changes.
- **Spread**: The yield spread over a reference rate (if calculated).

### Comprehensive Dashboard

The dashboard combines all visualizations into a single interactive HTML page.

```python
visualizer.create_dashboard(
    result,
    output_html="deal_dashboard.html",
    show=True
)
```

The dashboard includes:
- Deal summary information
- Bond cashflow charts
- Bond balance projections
- Pool performance metrics
- Bond performance metrics (yield, WAL, etc.)

## Interpreting Results

### Key Performance Indicators

When analyzing structured finance deals, pay attention to these key performance indicators:

1. **Bond Yields**:
   - Higher yields generally indicate higher risk.
   - The spread between different tranches reflects the credit enhancement.

2. **Weighted Average Life (WAL)**:
   - Shorter WALs indicate faster repayment of principal.
   - WAL is affected by prepayment and default assumptions.

3. **Duration**:
   - Measures the sensitivity of the bond price to interest rate changes.
   - Longer duration means higher interest rate risk.

4. **Credit Enhancement**:
   - Visible in the subordination structure (junior bonds absorb losses first).
   - Reflected in the yield differences between tranches.

5. **Default Sensitivity**:
   - How bond performance changes under different default scenarios.
   - Junior tranches are more sensitive to changes in default rates.

6. **Prepayment Sensitivity**:
   - How bond performance changes under different prepayment scenarios.
   - Important for premium and discount bonds.

### Common Patterns

1. **Sequential Pay Structure**:
   - Senior bonds receive all principal until they are paid off.
   - Visible as staggered paydowns in the balance charts.

2. **Pro-Rata Pay Structure**:
   - All bonds receive principal proportional to their outstanding balance.
   - Visible as parallel declines in the balance charts.

3. **Clean-up Calls**:
   - Often triggered when the pool balance falls below a certain threshold.
   - Visible as sudden paydowns of all remaining bonds.

## Advanced Usage

### Comparing Scenarios

To compare different scenarios (e.g., base case vs. stress case):

```python
# Run analyses for different scenarios
base_result = service.analyze_deal(base_case_deal)
stress_result = service.analyze_deal(stress_case_deal)

# Create visualizer
visualizer = AbsBoxVisualizer(output_dir="./scenarios")

# Create dashboards for each scenario
visualizer.create_dashboard(base_result, output_html="base_case_dashboard.html")
visualizer.create_dashboard(stress_result, output_html="stress_case_dashboard.html")

# Create a comparison chart (example: comparing ClassA bond balance under different scenarios)
import plotly.graph_objects as go

fig = go.Figure()

# Extract ClassA balance data from each scenario
base_df = base_result.get("bond_cashflows", {})
stress_df = stress_result.get("bond_cashflows", {})

fig.add_trace(go.Scatter(
    x=base_df.get("date", []),
    y=base_df.get("bond_ClassA_balance", []),
    name="Base Case - ClassA Balance"
))

fig.add_trace(go.Scatter(
    x=stress_df.get("date", []),
    y=stress_df.get("bond_ClassA_balance", []),
    name="Stress Case - ClassA Balance"
))

fig.update_layout(
    title="ClassA Balance Comparison",
    xaxis_title="Date",
    yaxis_title="Balance",
    legend_title="Scenario"
)

fig.write_html("scenario_comparison.html")
```

### Custom Visualizations

You can create custom visualizations by accessing the underlying data:

```python
# Get cashflow data from the result
dfs = visualizer._prepare_cashflow_data(result)

bond_df = dfs.get("bond_cashflows")
pool_df = dfs.get("pool_cashflows")

# Create a custom visualization using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(bond_df["date"], bond_df["bond_ClassA_interest"], label="ClassA Interest")
plt.plot(bond_df["date"], bond_df["bond_ClassB_interest"], label="ClassB Interest")
plt.title("Interest Payment Comparison")
plt.xlabel("Date")
plt.ylabel("Interest Amount")
plt.legend()
plt.savefig("custom_interest_chart.png")
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Error: "Visualization dependencies not available"
   - Solution: Install required packages with `pip install -r app/utils/absbox_visualization_requirements.txt`

2. **No Data to Visualize**:
   - Error: "No bond cashflow data available"
   - Solution: Check that your analysis result contains the expected data fields.

3. **Interactive Features Not Working**:
   - Issue: Plots are static in the browser
   - Solution: Make sure you're using the latest version of plotly and check browser compatibility.

4. **Memory Issues with Large Deals**:
   - Issue: Program crashes when visualizing very large deals
   - Solution: Reduce the number of periods or aggregate data (e.g., quarterly instead of monthly).

### Getting Help

If you encounter issues not covered in this guide:

1. Check the AbsBox service logs for errors in the analysis.
2. Verify that the result structure matches what the visualizer expects.
3. Try visualizing a simpler deal first to make sure the visualizer works correctly.
4. Contact the AbsBox support team for assistance with complex visualization needs.

---

## Appendix: Sample Data Structure

The visualizer expects results in the following format:

```json
{
  "deal_name": "Sample Deal",
  "status": "success",
  "execution_time": 1.25,
  "bond_cashflows": [
    {
      "date": "2023-01-01",
      "bond_ClassA_principal": 1000.0,
      "bond_ClassA_interest": 35.0,
      "bond_ClassA_balance": 19000.0,
      "bond_ClassB_principal": 0.0,
      "bond_ClassB_interest": 45.0,
      "bond_ClassB_balance": 5000.0
    },
    ...
  ],
  "pool_cashflows": [
    {
      "date": "2023-01-01",
      "scheduled_principal": 900.0,
      "prepaid_principal": 100.0,
      "defaulted_balance": 50.0,
      "remaining_balance": 19950.0
    },
    ...
  ],
  "metrics": {
    "bond_ClassA_yield": 0.035,
    "bond_ClassA_wal": 3.5,
    "bond_ClassB_yield": 0.055,
    "bond_ClassB_wal": 5.2
  }
}
```
