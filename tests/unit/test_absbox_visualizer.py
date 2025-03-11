"""
Unit tests for the AbsBox Visualizer component.

These tests verify that the AbsBox visualizer correctly processes
structured finance analysis results and generates appropriate visualizations.
"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import required packages
try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    VISUALIZATION_IMPORTS_OK = True
except ImportError:
    VISUALIZATION_IMPORTS_OK = False

# Skip all tests if visualization packages aren't available
pytestmark = pytest.mark.skipif(
    not VISUALIZATION_IMPORTS_OK,
    reason="Visualization dependencies not available"
)

# Import the AbsBox visualizer
try:
    from app.utils.absbox_visualizer import AbsBoxVisualizer
except ImportError as e:
    pytest.skip(f"Failed to import AbsBoxVisualizer: {e}")

@pytest.fixture
def sample_result():
    """Create a sample analysis result."""
    # Create dates for cashflows
    dates = pd.date_range(start="2023-01-01", periods=12, freq="MS").strftime("%Y-%m-%d").tolist()
    
    # Create bond cashflows
    bond_cashflows = []
    class_a_balance = 200000.0
    class_b_balance = 50000.0
    
    for i, date in enumerate(dates):
        # Calculate principal payments
        class_a_principal = 5000.0 if i > 0 else 0.0
        class_b_principal = 0.0 if i < 6 else 2500.0
        
        # Update balances
        class_a_balance -= class_a_principal
        class_b_balance -= class_b_principal
        
        # Calculate interest
        class_a_interest = class_a_balance * 0.04 / 12  # 4% annual rate
        class_b_interest = class_b_balance * 0.06 / 12  # 6% annual rate
        
        # Create cashflow entry
        bond_cashflows.append({
            "date": date,
            "bond_ClassA_principal": class_a_principal,
            "bond_ClassA_interest": class_a_interest,
            "bond_ClassA_balance": class_a_balance,
            "bond_ClassA_factor": class_a_balance / 200000.0,
            "bond_ClassB_principal": class_b_principal,
            "bond_ClassB_interest": class_b_interest,
            "bond_ClassB_balance": class_b_balance,
            "bond_ClassB_factor": class_b_balance / 50000.0
        })
    
    # Create pool cashflows
    pool_cashflows = []
    pool_balance = 250000.0
    initial_balance = pool_balance
    cumulative_defaults = 0.0
    cumulative_prepayments = 0.0
    
    for i, date in enumerate(dates):
        # Calculate cashflows
        scheduled_principal = 5000.0
        prepaid_principal = 1000.0 if i > 2 else 0.0
        defaulted_balance = 500.0 if i > 1 else 0.0
        
        # Update balances
        cumulative_defaults += defaulted_balance
        cumulative_prepayments += prepaid_principal
        pool_balance -= (scheduled_principal + prepaid_principal + defaulted_balance)
        
        # Create cashflow entry
        pool_cashflows.append({
            "date": date,
            "scheduled_principal": scheduled_principal,
            "prepaid_principal": prepaid_principal,
            "defaulted_balance": defaulted_balance,
            "remaining_balance": pool_balance,
            "initial_balance": initial_balance
        })
    
    # Create metrics
    metrics = {
        "bond_ClassA_yield": 0.042,
        "bond_ClassA_duration": 5.2,
        "bond_ClassA_wam": 6.3,
        "bond_ClassA_irr": 0.041,
        "bond_ClassB_yield": 0.065,
        "bond_ClassB_duration": 7.1,
        "bond_ClassB_wam": 8.5,
        "bond_ClassB_irr": 0.064,
        "pool_wam": 7.5,
        "pool_default_rate": 0.02,
        "pool_cpr": 0.05
    }
    
    # Create complete result
    return {
        "deal_name": "Test Deal",
        "status": "success",
        "execution_time": 1.5,
        "bond_cashflows": bond_cashflows,
        "pool_cashflows": pool_cashflows,
        "metrics": metrics
    }

@pytest.fixture
def temp_output_dir(tmpdir):
    """Create a temporary output directory."""
    return str(tmpdir.mkdir("viz_output"))

def test_visualizer_initialization():
    """Test that the visualizer initializes correctly."""
    # Create with default settings
    visualizer = AbsBoxVisualizer()
    assert visualizer is not None
    assert visualizer.output_dir is None
    
    # Create with output directory
    test_dir = "test_output"
    visualizer = AbsBoxVisualizer(output_dir=test_dir)
    assert visualizer.output_dir == test_dir
    assert os.path.exists(test_dir)
    
    # Clean up
    try:
        os.rmdir(test_dir)
    except:
        pass

def test_prepare_cashflow_data(sample_result):
    """Test that cashflow data is properly prepared."""
    visualizer = AbsBoxVisualizer()
    dfs = visualizer._prepare_cashflow_data(sample_result)
    
    # Check that we have both bond and pool dataframes
    assert "bond_cashflows" in dfs
    assert "pool_cashflows" in dfs
    
    # Check bond cashflow dataframe
    bond_df = dfs["bond_cashflows"]
    assert "date" in bond_df.columns
    assert "bond_ClassA_principal" in bond_df.columns
    assert "bond_ClassA_interest" in bond_df.columns
    assert "bond_ClassA_balance" in bond_df.columns
    assert "bond_ClassB_principal" in bond_df.columns
    assert "bond_ClassB_interest" in bond_df.columns
    assert "bond_ClassB_balance" in bond_df.columns
    
    # Check pool cashflow dataframe
    pool_df = dfs["pool_cashflows"]
    assert "date" in pool_df.columns
    assert "scheduled_principal" in pool_df.columns
    assert "prepaid_principal" in pool_df.columns
    assert "defaulted_balance" in pool_df.columns
    assert "remaining_balance" in pool_df.columns
    
    # Check date conversion
    assert pd.api.types.is_datetime64_dtype(bond_df["date"])
    assert pd.api.types.is_datetime64_dtype(pool_df["date"])

def test_prepare_metrics_data(sample_result):
    """Test that metrics data is properly prepared."""
    visualizer = AbsBoxVisualizer()
    metrics_df = visualizer._prepare_metrics_data(sample_result)
    
    # Check that metrics dataframe was created
    assert metrics_df is not None
    
    # Check metrics content
    assert "yield" in metrics_df.columns
    assert "duration" in metrics_df.columns
    assert "wam" in metrics_df.columns
    assert "irr" in metrics_df.columns
    
    # Check index (bond names)
    assert "ClassA" in metrics_df.index
    assert "ClassB" in metrics_df.index
    
    # Check values
    assert metrics_df.loc["ClassA", "yield"] == 0.042
    assert metrics_df.loc["ClassB", "yield"] == 0.065

def test_plot_bond_cashflows(sample_result, temp_output_dir):
    """Test bond cashflow plot generation."""
    visualizer = AbsBoxVisualizer(output_dir=temp_output_dir)
    
    # Test different plot types
    plot_types = ["stacked_area", "line", "bar"]
    
    for plot_type in plot_types:
        # Test visualization without saving
        with patch("plotly.graph_objects.Figure.show") as mock_show:
            visualizer.plot_bond_cashflows(sample_result, plot_type=plot_type, show=True)
            assert mock_show.called
        
        # Test visualization with saving
        output_path = visualizer.plot_bond_cashflows(
            sample_result, 
            plot_type=plot_type,
            save_file=f"bond_cashflows_{plot_type}",
            show=False
        )
        
        # Check that the file was created
        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith(f"bond_cashflows_{plot_type}.html")

def test_plot_bond_balances(sample_result, temp_output_dir):
    """Test bond balance plot generation."""
    visualizer = AbsBoxVisualizer(output_dir=temp_output_dir)
    
    # Test visualization without saving
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        visualizer.plot_bond_balances(sample_result, show=True)
        assert mock_show.called
    
    # Test visualization with saving
    output_path = visualizer.plot_bond_balances(
        sample_result,
        save_file="bond_balances",
        show=False
    )
    
    # Check that the file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("bond_balances.html")

def test_plot_pool_performance(sample_result, temp_output_dir):
    """Test pool performance plot generation."""
    visualizer = AbsBoxVisualizer(output_dir=temp_output_dir)
    
    # Test visualization without saving
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        visualizer.plot_pool_performance(sample_result, show=True)
        assert mock_show.called
    
    # Test visualization with saving
    output_path = visualizer.plot_pool_performance(
        sample_result,
        save_file="pool_performance",
        show=False
    )
    
    # Check that the file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("pool_performance.html")

def test_plot_bond_metrics(sample_result, temp_output_dir):
    """Test bond metrics plot generation."""
    visualizer = AbsBoxVisualizer(output_dir=temp_output_dir)
    
    # Test visualization without saving
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        visualizer.plot_bond_metrics(sample_result, show=True)
        assert mock_show.called
    
    # Test visualization with saving
    output_path = visualizer.plot_bond_metrics(
        sample_result,
        save_file="bond_metrics",
        show=False
    )
    
    # Check that the file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith("bond_metrics.html")

def test_create_dashboard(sample_result, temp_output_dir):
    """Test dashboard creation."""
    visualizer = AbsBoxVisualizer(output_dir=temp_output_dir)
    
    # Mock the webbrowser.open call to avoid opening a browser
    with patch("webbrowser.open") as mock_open:
        # Test dashboard creation with display
        output_path = visualizer.create_dashboard(
            sample_result,
            output_html=os.path.join(temp_output_dir, "test_dashboard.html"),
            show=True
        )
        
        # Check that the file was created
        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith("test_dashboard.html")
        
        # Check that the browser was opened
        assert mock_open.called
        mock_open.assert_called_with(f"file://{os.path.abspath(output_path)}")

def test_empty_result():
    """Test handling of empty result data."""
    visualizer = AbsBoxVisualizer()
    empty_result = {"status": "success", "deal_name": "Empty Deal"}
    
    # Test all visualization methods with empty data
    assert visualizer.plot_bond_cashflows(empty_result, show=False) is None
    assert visualizer.plot_bond_balances(empty_result, show=False) is None
    assert visualizer.plot_pool_performance(empty_result, show=False) is None
    assert visualizer.plot_bond_metrics(empty_result, show=False) is None

def test_error_result():
    """Test handling of error result data."""
    visualizer = AbsBoxVisualizer()
    error_result = {
        "status": "error",
        "deal_name": "Error Deal",
        "error": "Analysis failed due to invalid input",
        "error_type": "ValidationError"
    }
    
    # Test all visualization methods with error data
    assert visualizer.plot_bond_cashflows(error_result, show=False) is None
    assert visualizer.plot_bond_balances(error_result, show=False) is None
    assert visualizer.plot_pool_performance(error_result, show=False) is None
    assert visualizer.plot_bond_metrics(error_result, show=False) is None

def test_partial_result(sample_result):
    """Test handling of partial result data."""
    visualizer = AbsBoxVisualizer()
    
    # Create a result with only bond data
    bond_only_result = {
        "status": "success",
        "deal_name": "Bond Only Deal",
        "bond_cashflows": sample_result["bond_cashflows"]
    }
    
    # Create a result with only pool data
    pool_only_result = {
        "status": "success",
        "deal_name": "Pool Only Deal",
        "pool_cashflows": sample_result["pool_cashflows"]
    }
    
    # Create a result with only metrics
    metrics_only_result = {
        "status": "success",
        "deal_name": "Metrics Only Deal",
        "metrics": sample_result["metrics"]
    }
    
    # Test bond plots with different data availability
    assert visualizer.plot_bond_cashflows(bond_only_result, show=False) is not None
    assert visualizer.plot_bond_cashflows(pool_only_result, show=False) is None
    assert visualizer.plot_bond_cashflows(metrics_only_result, show=False) is None
    
    # Test pool plot with different data availability
    assert visualizer.plot_pool_performance(bond_only_result, show=False) is None
    assert visualizer.plot_pool_performance(pool_only_result, show=False) is not None
    assert visualizer.plot_pool_performance(metrics_only_result, show=False) is None
    
    # Test metrics plot with different data availability
    assert visualizer.plot_bond_metrics(bond_only_result, show=False) is None
    assert visualizer.plot_bond_metrics(pool_only_result, show=False) is None
    assert visualizer.plot_bond_metrics(metrics_only_result, show=False) is not None

def test_convenience_function(sample_result, temp_output_dir):
    """Test the convenience function for visualizing deal results."""
    # Write sample result to a temporary file
    result_file = os.path.join(temp_output_dir, "test_result.json")
    with open(result_file, "w") as f:
        json.dump(sample_result, f)
    
    # Mock the create_dashboard method
    with patch("app.utils.absbox_visualizer.AbsBoxVisualizer.create_dashboard") as mock_dashboard:
        # Mock return value
        mock_dashboard.return_value = os.path.join(temp_output_dir, "dashboard.html")
        
        # Import the convenience function
        from app.utils.absbox_visualizer import visualize_deal_results
        
        # Call the function
        visualize_deal_results(result_file, output_dir=temp_output_dir)
        
        # Check that dashboard was created with the right parameters
        assert mock_dashboard.called
        args, kwargs = mock_dashboard.call_args
        
        # Check that the result was loaded correctly
        assert args[0]["deal_name"] == sample_result["deal_name"]
        
        # Check that the output path and show flag were set correctly
        assert kwargs["output_html"].endswith("Test_Deal_dashboard.html")
        assert kwargs["show"] is True

if __name__ == "__main__":
    pytest.main(["-v", __file__])
