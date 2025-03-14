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
