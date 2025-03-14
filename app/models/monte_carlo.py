"""
Monte Carlo Simulation Models

This module contains Pydantic models for Monte Carlo simulations and scenario analysis.
It provides a comprehensive set of models for defining, executing, and retrieving
results from Monte Carlo simulations for financial analytics.
"""
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import date, datetime
from pydantic import BaseModel, Field, validator
import uuid

class SimulationStatus(str, Enum):
    """Status of a Monte Carlo simulation"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class DistributionType(str, Enum):
    """Types of probability distributions for variables in simulations"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BETA = "beta"
    CUSTOM = "custom"

class ScenarioType(str, Enum):
    """Types of scenarios for analysis"""
    BASE = "base"
    STRESS = "stress"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    REGULATORY = "regulatory"
    CUSTOM = "custom"

class SimulationVariable(BaseModel):
    """
    Model for a variable in a Monte Carlo simulation
    
    Represents a variable that will be randomized in the simulation
    with its distribution parameters.
    """
    name: str
    description: Optional[str] = None
    distribution: DistributionType
    parameters: Dict[str, float]
    
    # For normal/lognormal: mean, std_dev
    # For uniform: min, max
    # For triangular: min, mode, max
    # For beta: alpha, beta, min, max
    # For custom: values, probabilities
    
    @validator('parameters')
    def validate_parameters(cls, parameters, values):
        """Validate that the required parameters for the distribution are provided"""
        distribution = values.get('distribution')
        
        if distribution == DistributionType.NORMAL or distribution == DistributionType.LOGNORMAL:
            required = {'mean', 'std_dev'}
        elif distribution == DistributionType.UNIFORM:
            required = {'min', 'max'}
        elif distribution == DistributionType.TRIANGULAR:
            required = {'min', 'mode', 'max'}
        elif distribution == DistributionType.BETA:
            required = {'alpha', 'beta', 'min', 'max'}
        elif distribution == DistributionType.CUSTOM:
            required = {'values', 'probabilities'}
        else:
            return parameters
        
        if not required.issubset(set(parameters.keys())):
            missing = required - set(parameters.keys())
            raise ValueError(f"Missing required parameters for {distribution} distribution: {missing}")
        
        return parameters

class CorrelationMatrix(BaseModel):
    """
    Model for a correlation matrix between variables
    
    Represents the correlations between variables in a Monte Carlo simulation.
    The matrix is a dictionary with keys as variable name pairs ('var1:var2')
    and values as correlation coefficients.
    """
    correlations: Dict[str, float]
    
    @validator('correlations')
    def validate_correlations(cls, correlations):
        """Validate that correlation coefficients are between -1 and 1"""
        for key, value in correlations.items():
            if not -1 <= value <= 1:
                raise ValueError(f"Correlation coefficient must be between -1 and 1: {key} = {value}")
        return correlations

class MonteCarloSimulationRequest(BaseModel):
    """
    Request model for a Monte Carlo simulation
    
    Comprehensive model for defining a Monte Carlo simulation with
    variables, correlations, and execution parameters.
    """
    # Simulation identification
    name: str
    description: Optional[str] = None
    
    # Simulation parameters
    num_simulations: int = Field(..., gt=0, le=100000)
    variables: List[SimulationVariable] = Field(..., min_items=1)
    correlation_matrix: Optional[CorrelationMatrix] = None
    
    # Asset parameters
    asset_class: str
    asset_parameters: Dict[str, Any]
    
    # Analysis parameters
    analysis_date: date
    projection_months: int = Field(..., gt=0, le=600)  # Up to 50 years
    discount_rate: Optional[float] = None
    include_detailed_paths: bool = False  # If True, include all simulation paths in result
    
    # Calculation parameters
    percentiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]  # Default percentiles to calculate
    
    @validator('percentiles')
    def validate_percentiles(cls, percentiles):
        """Validate that percentiles are between 0 and 1"""
        for p in percentiles:
            if not 0 <= p <= 1:
                raise ValueError(f"Percentile must be between 0 and 1: {p}")
        return percentiles

class SimulationResult(BaseModel):
    """
    Model for a single simulation result
    
    Represents the result of a single path in a Monte Carlo simulation.
    """
    simulation_id: int
    variables: Dict[str, float]  # Input variables for this simulation
    metrics: Dict[str, float]    # Output metrics for this simulation
    cashflows: Optional[List[Dict[str, float]]] = None

class MonteCarloSimulationResult(BaseModel):
    """
    Result model for a Monte Carlo simulation
    
    Comprehensive model for the results of a Monte Carlo simulation
    with statistics, percentiles, and optional detailed paths.
    """
    # Simulation identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    status: SimulationStatus
    
    # Execution information
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    num_simulations: int
    num_completed: int
    error: Optional[str] = None
    
    # Summary statistics
    summary_statistics: Dict[str, Dict[str, float]]  # Metric -> {mean, std_dev, min, max}
    percentiles: Dict[str, Dict[float, float]]       # Metric -> {percentile -> value}
    
    # Detailed paths (optional)
    detailed_paths: Optional[List[SimulationResult]] = None

class ScenarioDefinition(BaseModel):
    """
    Model for a scenario definition
    
    Represents a scenario for analysis with specific parameter values.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: ScenarioType
    parameters: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Stress Test Scenario 1",
                "description": "Economic downturn stress test",
                "type": "stress",
                "parameters": {
                    "default_rate_multiplier": 2.0,
                    "recovery_rate_multiplier": 0.8,
                    "interest_rate_shock": 0.02,
                    "property_value_decline": 0.15
                }
            }
        }

class SavedSimulation(BaseModel):
    """
    Model for a saved simulation
    
    Represents a saved Monte Carlo simulation with its definition,
    results, and metadata for retrieval from the database.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    description: Optional[str] = None
    request: MonteCarloSimulationRequest
    result: Optional[MonteCarloSimulationResult] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "sim_12345",
                "user_id": "user_54321",
                "name": "Prepayment Risk Analysis",
                "description": "Analysis of prepayment risk under varying interest rate environments",
                "created_at": "2023-07-15T12:30:45",
                "updated_at": "2023-07-15T12:45:12"
            }
        }

class StatisticalOutputs(BaseModel):
    """
    Model for statistical outputs of a Monte Carlo simulation metric
    
    Comprehensive statistical summary for a metric from a Monte Carlo simulation.
    """
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentiles: Dict[str, float]  # Percentile (as string '10', '50', etc.) -> value

class RiskMetrics(BaseModel):
    """
    Model for comprehensive risk metrics from a Monte Carlo simulation
    
    Provides a complete set of risk measurements including Value at Risk (VaR),
    Conditional Value at Risk (CVaR), volatility measures, and performance ratios.
    """
    # Value at Risk metrics at different confidence levels (e.g., "0.95": 100000.0)
    var: Dict[str, float]
    
    # Conditional Value at Risk (Expected Shortfall) at different confidence levels
    cvar: Dict[str, float]
    
    # Volatility measures
    volatility: float
    downside_deviation: float
    
    # Performance ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Drawdown measures
    max_drawdown: float = 0.0
    
    # Optional advanced metrics
    tail_risk: Optional[float] = None
    stress_loss: Optional[float] = None
    
    class Config:
        """Configuration for the RiskMetrics model"""
        schema_extra = {
            "example": {
                "var": {"0.95": 100000.0, "0.99": 150000.0},
                "cvar": {"0.95": 120000.0, "0.99": 175000.0},
                "volatility": 50000.0,
                "downside_deviation": 45000.0,
                "sharpe_ratio": 0.8,
                "sortino_ratio": 0.9,
                "max_drawdown": 200000.0
            }
        }

class MonteCarloResult(BaseModel):
    """
    Enhanced result model for a Monte Carlo simulation
    
    Comprehensive model for the results of a Monte Carlo simulation
    with detailed statistics, risk metrics, and visualization data.
    """
    # Simulation identification
    simulation_id: str
    
    # Execution information
    num_iterations: int
    time_horizon: int
    calculation_time: float
    completed_iterations: Optional[int] = None
    partial_completion: Optional[bool] = False
    error: Optional[str] = None
    
    # NPV statistics
    npv_stats: StatisticalOutputs
    
    # Risk metrics
    risk_metrics: Optional[RiskMetrics] = None
    
    # Additional statistics (optional)
    irr_stats: Optional[StatisticalOutputs] = None
    duration_stats: Optional[StatisticalOutputs] = None
    default_stats: Optional[StatisticalOutputs] = None
    prepayment_stats: Optional[StatisticalOutputs] = None
    loss_stats: Optional[StatisticalOutputs] = None
    
    # Additional risk metrics (optional)
    irr_risk_metrics: Optional[RiskMetrics] = None
    loss_risk_metrics: Optional[RiskMetrics] = None
    
    # Loss distribution and capital metrics
    loss_distribution: Optional[Dict[str, float]] = None
    expected_loss: Optional[float] = None
    unexpected_loss: Optional[float] = None
    economic_capital: Optional[float] = None
    
    # Portfolio metrics
    diversification_benefit: Optional[float] = None
    correlation_effect: Optional[float] = None
    
    # Cashflow projections (optional)
    best_case_cashflows: Optional[Any] = None
    worst_case_cashflows: Optional[Any] = None
    median_case_cashflows: Optional[Any] = None
    representative_paths: Optional[List[Any]] = None
    
    class Config:
        """Configuration for the MonteCarloResult model"""
        schema_extra = {
            "example": {
                "simulation_id": "abc123-xyz789",
                "num_iterations": 1000,
                "time_horizon": 120,
                "calculation_time": 10.5,
                "completed_iterations": 1000,
                "npv_stats": {
                    "mean": 500000.0,
                    "median": 520000.0,
                    "std_dev": 75000.0,
                    "min_value": 250000.0,
                    "max_value": 750000.0,
                    "percentiles": {"10": 400000.0, "50": 520000.0, "90": 600000.0}
                },
                "risk_metrics": {
                    "var": {"0.95": 100000.0},
                    "cvar": {"0.95": 120000.0},
                    "volatility": 75000.0,
                    "downside_deviation": 65000.0,
                    "sharpe_ratio": 0.85,
                    "sortino_ratio": 0.95,
                    "max_drawdown": 150000.0
                }
            }
        }
