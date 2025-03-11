"""
AbsBox Results Visualizer

This module provides visualization tools for AbsBox analysis results,
generating interactive charts for cashflows, bond performance metrics,
and other structured finance analytics.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("absbox_visualizer")

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_IMPORTS_OK = True
except ImportError as e:
    logger.warning(f"Visualization dependencies not available: {e}")
    VISUALIZATION_IMPORTS_OK = False

class AbsBoxVisualizer:
    """Class for visualizing AbsBox results"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the visualizer
        
        Args:
            output_dir: Directory to save output files (optional)
        """
        if not VISUALIZATION_IMPORTS_OK:
            raise ImportError(
                "Visualization dependencies not available. "
                "Install with: pip install pandas matplotlib plotly"
            )
        
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _prepare_cashflow_data(self, result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Prepare cashflow data for visualization
        
        Args:
            result: Analysis result dictionary
        
        Returns:
            Dictionary with pandas DataFrames for bond and pool cashflows
        """
        # Extract cashflow data
        bond_cf = result.get("bond_cashflows", {})
        pool_cf = result.get("pool_cashflows", {})
        
        # Convert to DataFrames
        dfs = {}
        
        if bond_cf:
            # Convert bond cashflows to DataFrame
            bond_df = pd.DataFrame(bond_cf)
            # Convert date column if present
            if "date" in bond_df.columns:
                bond_df["date"] = pd.to_datetime(bond_df["date"])
            dfs["bond_cashflows"] = bond_df
        
        if pool_cf:
            # Convert pool cashflows to DataFrame
            pool_df = pd.DataFrame(pool_cf)
            # Convert date column if present
            if "date" in pool_df.columns:
                pool_df["date"] = pd.to_datetime(pool_df["date"])
            dfs["pool_cashflows"] = pool_df
        
        return dfs
    
    def _prepare_metrics_data(self, result: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare metrics data for visualization
        
        Args:
            result: Analysis result dictionary
        
        Returns:
            Pandas DataFrame with bond metrics
        """
        metrics = result.get("metrics", {})
        if not metrics:
            return None
        
        # Extract bond metrics
        bond_metrics = {}
        for key, value in metrics.items():
            if key.startswith("bond_"):
                parts = key.split("_", 1)
                if len(parts) > 1:
                    bond_name = parts[1]
                    metric_type = parts[0]
                    
                    if bond_name not in bond_metrics:
                        bond_metrics[bond_name] = {}
                    
                    bond_metrics[bond_name][metric_type] = value
        
        # Convert to DataFrame
        if bond_metrics:
            metrics_df = pd.DataFrame.from_dict(bond_metrics, orient="index")
            return metrics_df
        
        return None
    
    def plot_bond_cashflows(self, result: Dict[str, Any], plot_type: str = "stacked_area", 
                           save_file: Optional[str] = None, show: bool = True) -> Union[str, None]:
        """Plot bond cashflows from analysis results
        
        Args:
            result: Analysis result dictionary
            plot_type: Type of plot ("stacked_area", "line", "bar")
            save_file: Filename to save the plot (without extension)
            show: Whether to display the plot
        
        Returns:
            Path to the saved file if save_file is provided
        """
        # Prepare data
        dfs = self._prepare_cashflow_data(result)
        if "bond_cashflows" not in dfs:
            logger.warning("No bond cashflow data available")
            return None
        
        bond_df = dfs["bond_cashflows"]
        
        # Get bond names
        bond_columns = [col for col in bond_df.columns if col.startswith("bond_")]
        if not bond_columns:
            logger.warning("No bond columns found in cashflow data")
            return None
        
        # Extract dates and principal/interest columns
        date_col = "date" if "date" in bond_df.columns else None
        principal_cols = [col for col in bond_columns if col.endswith("_principal")]
        interest_cols = [col for col in bond_columns if col.endswith("_interest")]
        
        # Create figure
        if plot_type == "stacked_area":
            fig = self._create_stacked_area_plot(bond_df, date_col, principal_cols, interest_cols)
        elif plot_type == "line":
            fig = self._create_line_plot(bond_df, date_col, principal_cols, interest_cols)
        elif plot_type == "bar":
            fig = self._create_bar_plot(bond_df, date_col, principal_cols, interest_cols)
        else:
            logger.warning(f"Invalid plot type: {plot_type}")
            return None
        
        # Add title and labels
        fig.update_layout(
            title="Bond Cashflows Over Time",
            xaxis_title="Period",
            yaxis_title="Amount",
            legend_title="Cashflow Type",
            template="plotly_white"
        )
        
        # Save or show the plot
        if save_file:
            output_path = self._save_plot(fig, save_file)
            if show:
                fig.show()
            return output_path
        elif show:
            fig.show()
        
        return None
    
    def _create_stacked_area_plot(self, df, date_col, principal_cols, interest_cols):
        """Create a stacked area plot for bond cashflows"""
        fig = go.Figure()
        
        # Add principal cashflows
        for col in principal_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Scatter(
                x=df[date_col] if date_col else df.index,
                y=df[col],
                name=f"{bond_name} Principal",
                stackgroup="principal",
                line=dict(width=0.5),
                fill='tonexty'
            ))
        
        # Add interest cashflows
        for col in interest_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Scatter(
                x=df[date_col] if date_col else df.index,
                y=df[col],
                name=f"{bond_name} Interest",
                stackgroup="interest",
                line=dict(width=0.5),
                fill='tonexty'
            ))
        
        return fig
    
    def _create_line_plot(self, df, date_col, principal_cols, interest_cols):
        """Create a line plot for bond cashflows"""
        fig = go.Figure()
        
        # Add principal cashflows
        for col in principal_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Scatter(
                x=df[date_col] if date_col else df.index,
                y=df[col],
                name=f"{bond_name} Principal",
                mode='lines'
            ))
        
        # Add interest cashflows
        for col in interest_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Scatter(
                x=df[date_col] if date_col else df.index,
                y=df[col],
                name=f"{bond_name} Interest",
                mode='lines',
                line=dict(dash='dash')
            ))
        
        return fig
    
    def _create_bar_plot(self, df, date_col, principal_cols, interest_cols):
        """Create a bar plot for bond cashflows"""
        fig = go.Figure()
        
        # Add principal cashflows
        for col in principal_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Bar(
                x=df[date_col] if date_col else df.index,
                y=df[col],
                name=f"{bond_name} Principal"
            ))
        
        # Add interest cashflows
        for col in interest_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Bar(
                x=df[date_col] if date_col else df.index,
                y=df[col],
                name=f"{bond_name} Interest"
            ))
        
        return fig
    
    def plot_bond_balances(self, result: Dict[str, Any], save_file: Optional[str] = None, 
                          show: bool = True) -> Union[str, None]:
        """Plot bond balance projections
        
        Args:
            result: Analysis result dictionary
            save_file: Filename to save the plot (without extension)
            show: Whether to display the plot
        
        Returns:
            Path to the saved file if save_file is provided
        """
        # Prepare data
        dfs = self._prepare_cashflow_data(result)
        if "bond_cashflows" not in dfs:
            logger.warning("No bond cashflow data available")
            return None
        
        bond_df = dfs["bond_cashflows"]
        
        # Get bond balance columns
        balance_cols = [col for col in bond_df.columns if col.endswith("_balance")]
        if not balance_cols:
            logger.warning("No bond balance columns found in cashflow data")
            return None
        
        # Extract date column
        date_col = "date" if "date" in bond_df.columns else None
        
        # Create figure
        fig = go.Figure()
        
        # Add balance traces
        for col in balance_cols:
            bond_name = col.split("_")[1]
            fig.add_trace(go.Scatter(
                x=bond_df[date_col] if date_col else bond_df.index,
                y=bond_df[col],
                name=f"{bond_name} Balance",
                mode='lines'
            ))
        
        # Add factor traces if available
        factor_cols = [col for col in bond_df.columns if col.endswith("_factor")]
        if factor_cols:
            # Create a secondary y-axis for factors
            fig.update_layout(
                yaxis2=dict(
                    title="Factor",
                    overlaying="y",
                    side="right",
                    range=[0, 1.1]
                )
            )
            
            for col in factor_cols:
                bond_name = col.split("_")[1]
                fig.add_trace(go.Scatter(
                    x=bond_df[date_col] if date_col else bond_df.index,
                    y=bond_df[col],
                    name=f"{bond_name} Factor",
                    mode='lines',
                    line=dict(dash='dot'),
                    yaxis="y2"
                ))
        
        # Add title and labels
        fig.update_layout(
            title="Bond Balance Projections",
            xaxis_title="Period",
            yaxis_title="Balance",
            legend_title="Bond",
            template="plotly_white"
        )
        
        # Save or show the plot
        if save_file:
            output_path = self._save_plot(fig, save_file)
            if show:
                fig.show()
            return output_path
        elif show:
            fig.show()
        
        return None
    
    def plot_pool_performance(self, result: Dict[str, Any], save_file: Optional[str] = None,
                             show: bool = True) -> Union[str, None]:
        """Plot pool performance metrics
        
        Args:
            result: Analysis result dictionary
            save_file: Filename to save the plot (without extension)
            show: Whether to display the plot
        
        Returns:
            Path to the saved file if save_file is provided
        """
        # Prepare data
        dfs = self._prepare_cashflow_data(result)
        if "pool_cashflows" not in dfs:
            logger.warning("No pool cashflow data available")
            return None
        
        pool_df = dfs["pool_cashflows"]
        
        # Check for required columns
        required_metrics = ["defaulted_balance", "prepaid_balance", "scheduled_principal", "remaining_balance"]
        available_metrics = [col for col in required_metrics if col in pool_df.columns]
        
        if len(available_metrics) < 2:
            logger.warning("Not enough pool metrics available for visualization")
            return None
        
        # Extract date column
        date_col = "date" if "date" in pool_df.columns else None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Pool Balance and Cashflows", "Cumulative Default and Prepayment Rates"),
            vertical_spacing=0.15
        )
        
        # First subplot: Balance and cashflows
        if "remaining_balance" in pool_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pool_df[date_col] if date_col else pool_df.index,
                    y=pool_df["remaining_balance"],
                    name="Remaining Balance",
                    mode='lines'
                ),
                row=1, col=1
            )
        
        if "scheduled_principal" in pool_df.columns:
            fig.add_trace(
                go.Bar(
                    x=pool_df[date_col] if date_col else pool_df.index,
                    y=pool_df["scheduled_principal"],
                    name="Scheduled Principal"
                ),
                row=1, col=1
            )
        
        # Second subplot: Default and prepayment rates
        # Calculate cumulative default and prepayment rates if possible
        if all(col in pool_df.columns for col in ["defaulted_balance", "prepaid_balance", "initial_balance"]):
            # Get initial balance
            initial_balance = pool_df["initial_balance"].iloc[0] if "initial_balance" in pool_df.columns else pool_df["remaining_balance"].iloc[0]
            
            # Calculate cumulative rates
            pool_df["cum_default_rate"] = pool_df["defaulted_balance"].cumsum() / initial_balance
            pool_df["cum_prepay_rate"] = pool_df["prepaid_balance"].cumsum() / initial_balance
            
            # Plot cumulative rates
            fig.add_trace(
                go.Scatter(
                    x=pool_df[date_col] if date_col else pool_df.index,
                    y=pool_df["cum_default_rate"],
                    name="Cumulative Default Rate",
                    mode='lines'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pool_df[date_col] if date_col else pool_df.index,
                    y=pool_df["cum_prepay_rate"],
                    name="Cumulative Prepayment Rate",
                    mode='lines'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Pool Performance Metrics",
            xaxis2_title="Period",
            yaxis_title="Amount",
            yaxis2_title="Rate",
            legend_title="Metric",
            template="plotly_white",
            height=800
        )
        
        # Set y-axis range for rates
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        
        # Save or show the plot
        if save_file:
            output_path = self._save_plot(fig, save_file)
            if show:
                fig.show()
            return output_path
        elif show:
            fig.show()
        
        return None
    
    def plot_bond_metrics(self, result: Dict[str, Any], save_file: Optional[str] = None,
                          show: bool = True) -> Union[str, None]:
        """Plot bond performance metrics
        
        Args:
            result: Analysis result dictionary
            save_file: Filename to save the plot (without extension)
            show: Whether to display the plot
        
        Returns:
            Path to the saved file if save_file is provided
        """
        # Prepare metrics data
        metrics_df = self._prepare_metrics_data(result)
        if metrics_df is None:
            logger.warning("No bond metrics available")
            return None
        
        # Check for yield or IRR columns
        if not any(col in metrics_df.columns for col in ["yield", "irr", "duration", "wam"]):
            logger.warning("No relevant metrics found for visualization")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Plot metrics as a horizontal bar chart
        for metric in ["yield", "irr", "duration", "wam"]:
            if metric in metrics_df.columns:
                sorted_df = metrics_df.sort_values(by=metric)
                
                fig.add_trace(go.Bar(
                    y=sorted_df.index,
                    x=sorted_df[metric],
                    name=metric.upper(),
                    orientation='h'
                ))
        
        # Update layout
        fig.update_layout(
            title="Bond Performance Metrics",
            xaxis_title="Value",
            yaxis_title="Bond",
            legend_title="Metric",
            template="plotly_white",
            height=max(400, 100 * len(metrics_df)),
            barmode='group'
        )
        
        # Save or show the plot
        if save_file:
            output_path = self._save_plot(fig, save_file)
            if show:
                fig.show()
            return output_path
        elif show:
            fig.show()
        
        return None
    
    def create_dashboard(self, result: Dict[str, Any], output_html: Optional[str] = None,
                        show: bool = True) -> Union[str, None]:
        """Create a comprehensive dashboard for the analysis results
        
        Args:
            result: Analysis result dictionary
            output_html: HTML file to save the dashboard
            show: Whether to open the dashboard in a browser
        
        Returns:
            Path to the saved HTML file if output_html is provided
        """
        # Prepare data
        dfs = self._prepare_cashflow_data(result)
        metrics_df = self._prepare_metrics_data(result)
        
        if "bond_cashflows" not in dfs:
            logger.warning("No bond cashflow data available")
            return None
        
        # Create figures
        figures = {}
        
        # Bond cashflows
        try:
            bond_cf_fig = self.plot_bond_cashflows(result, show=False)
            if bond_cf_fig:
                figures["bond_cashflows"] = bond_cf_fig
        except Exception as e:
            logger.warning(f"Error creating bond cashflow plot: {e}")
        
        # Bond balances
        try:
            bond_bal_fig = self.plot_bond_balances(result, show=False)
            if bond_bal_fig:
                figures["bond_balances"] = bond_bal_fig
        except Exception as e:
            logger.warning(f"Error creating bond balance plot: {e}")
        
        # Pool performance
        try:
            pool_perf_fig = self.plot_pool_performance(result, show=False)
            if pool_perf_fig:
                figures["pool_performance"] = pool_perf_fig
        except Exception as e:
            logger.warning(f"Error creating pool performance plot: {e}")
        
        # Bond metrics
        try:
            if metrics_df is not None:
                bond_metrics_fig = self.plot_bond_metrics(result, show=False)
                if bond_metrics_fig:
                    figures["bond_metrics"] = bond_metrics_fig
        except Exception as e:
            logger.warning(f"Error creating bond metrics plot: {e}")
        
        # Create HTML dashboard
        html_content = self._create_html_dashboard(result, figures)
        
        if output_html:
            # Make sure the directory exists
            Path(output_html).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(output_html, "w") as f:
                f.write(html_content)
            
            if show:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_html)}")
            
            return output_html
        elif show:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                f.write(html_content.encode("utf-8"))
                temp_path = f.name
            
            # Open in browser
            import webbrowser
            webbrowser.open(f"file://{temp_path}")
            
            return temp_path
        
        return None
    
    def _create_html_dashboard(self, result: Dict[str, Any], figures: Dict[str, Any]) -> str:
        """Create HTML dashboard content
        
        Args:
            result: Analysis result
            figures: Dictionary of figures
        
        Returns:
            HTML content
        """
        deal_name = result.get("deal_name", "Structured Deal Analysis")
        
        # Create HTML with embedded plotly figures
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{deal_name} Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 20px; }}
                .dashboard-title {{ margin-bottom: 30px; }}
                .chart-container {{ margin-bottom: 30px; }}
                .metrics-table {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="dashboard-title">{deal_name} Dashboard</h1>
                
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h3>Deal Summary</h3>
                            </div>
                            <div class="card-body">
                                <table class="table table-striped">
                                    <tr>
                                        <th>Deal Name</th>
                                        <td>{deal_name}</td>
                                    </tr>
                                    <tr>
                                        <th>Status</th>
                                        <td>{result.get("status", "N/A")}</td>
                                    </tr>
                                    <tr>
                                        <th>Execution Time</th>
                                        <td>{result.get("execution_time", "N/A")} seconds</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
        """
        
        # Add bond cashflows chart
        if "bond_cashflows" in figures:
            html += """
                <div class="row chart-container">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h3>Bond Cashflows</h3>
                            </div>
                            <div class="card-body">
                                <div id="bond_cashflows_chart" style="height: 500px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Add bond balances chart
        if "bond_balances" in figures:
            html += """
                <div class="row chart-container">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h3>Bond Balances</h3>
                            </div>
                            <div class="card-body">
                                <div id="bond_balances_chart" style="height: 500px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Add pool performance chart
        if "pool_performance" in figures:
            html += """
                <div class="row chart-container">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h3>Pool Performance</h3>
                            </div>
                            <div class="card-body">
                                <div id="pool_performance_chart" style="height: 700px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Add bond metrics chart
        if "bond_metrics" in figures:
            html += """
                <div class="row chart-container">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h3>Bond Metrics</h3>
                            </div>
                            <div class="card-body">
                                <div id="bond_metrics_chart" style="height: 500px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        # Add footer
        html += """
            <footer class="text-center mt-4 mb-5">
                <p class="text-muted">Generated with AbsBox Visualizer</p>
            </footer>
            </div>
        """
        
        # Add JavaScript to create plots
        html += "<script>"
        
        for fig_name, fig in figures.items():
            html += f"""
                var {fig_name}_data = {json.dumps(fig.data)};
                var {fig_name}_layout = {json.dumps(fig.layout)};
                Plotly.newPlot('{fig_name}_chart', {fig_name}_data, {fig_name}_layout);
            """
        
        html += """
        </script>
        </body>
        </html>
        """
        
        return html
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save a plot to file
        
        Args:
            fig: Plotly figure
            filename: Base filename (without extension)
        
        Returns:
            Path to the saved file
        """
        # Add html extension if not present
        if not filename.endswith(".html"):
            filename += ".html"
        
        # Determine full path
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
        else:
            filepath = filename
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the figure
        fig.write_html(filepath)
        logger.info(f"Plot saved to {filepath}")
        
        return filepath

# Convenience functions
def visualize_deal_results(result_file: str, output_dir: Optional[str] = None) -> None:
    """Visualize deal results from a JSON file
    
    Args:
        result_file: Path to JSON file with analysis results
        output_dir: Directory to save visualization files
    """
    if not VISUALIZATION_IMPORTS_OK:
        logger.error("Visualization dependencies not available")
        return
    
    # Load results
    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results file: {e}")
        return
    
    # Create visualizer
    visualizer = AbsBoxVisualizer(output_dir=output_dir)
    
    # Extract deal name from result or filename
    deal_name = result.get("deal_name", os.path.splitext(os.path.basename(result_file))[0])
    
    # Create dashboard
    dashboard_path = os.path.join(output_dir, f"{deal_name}_dashboard.html") if output_dir else f"{deal_name}_dashboard.html"
    visualizer.create_dashboard(result, output_html=dashboard_path, show=True)
    
    logger.info(f"Dashboard saved to {dashboard_path}")
