"""
HAStructure Service for AbsBox Integration

This service implements the HAStructure functionality as described in the AbsBox documentation:
https://absbox-doc.readthedocs.io/en/latest/installation.html

It provides:
1. Connection to HAStructure for structured finance calculations
2. High-performance modeling and simulation capabilities
3. Integration with the AbsBox ecosystem
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
import os
import asyncio
import time
from datetime import datetime, date
import httpx
import pandas as pd
import numpy as np

# Core imports
from app.core.config import settings
from app.core.cache_service import CacheService
from app.core.monitoring import CalculationTracker

# AbsBox imports
try:
    import absbox as ab
    from absbox.apis import HastructureEngine
    from absbox.local.engine import LiqEngine
    ABSBOX_AVAILABLE = True
except ImportError as e:
    ABSBOX_AVAILABLE = False
    logging.warning(f"AbsBox not fully available: {e}")

# Setup logging
logger = logging.getLogger(__name__)

class HAStructureService:
    """
    HAStructure Service for advanced structured finance analytics
    
    This service manages the HAStructure integration, providing:
    - Connection to HAStructure engine
    - Advanced calculation capabilities
    - Performance optimization for complex models
    """
    
    def __init__(self, 
                cache_service: Optional[CacheService] = None,
                hastructure_url: Optional[str] = None,
                hastructure_timeout: Optional[int] = None):
        """
        Initialize the HAStructure service
        
        Args:
            cache_service: Optional cache service for improved performance
            hastructure_url: Optional URL for HAStructure engine
            hastructure_timeout: Optional timeout for HAStructure operations in seconds
        """
        self.cache = cache_service or CacheService()
        self.hastructure_url = hastructure_url or settings.HASTRUCTURE_URL
        self.hastructure_timeout = hastructure_timeout or int(os.getenv("HASTRUCTURE_TIMEOUT", "300"))
        
        # Track available engines
        self.engines = {
            "local": None,  # Local engine instance
            "hastructure": None,  # HAStructure engine instance
        }
        
        # Performance metrics
        self.performance = {
            "calculations": 0,
            "cache_hits": 0,
            "errors": 0,
            "avg_calculation_time": 0,
        }
        
        # Check HAStructure availability
        self.hastructure_available = ABSBOX_AVAILABLE
        
        if self.hastructure_available:
            try:
                # Initialize both engines for flexibility
                self._initialize_engines()
                logger.info(f"HAStructure service initialized with URL: {self.hastructure_url}")
            except Exception as e:
                logger.error(f"Error initializing HAStructure engines: {e}")
                self.hastructure_available = False
        else:
            logger.warning("HAStructure integration not available due to missing dependencies")
    
    def _initialize_engines(self):
        """Initialize both local and HAStructure engines"""
        # Initialize local engine first as fallback
        try:
            self.engines["local"] = LiqEngine()
            logger.info("Local AbsBox engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize local AbsBox engine: {e}")
        
        # Then try to initialize HAStructure engine
        try:
            self.engines["hastructure"] = HastructureEngine(self.hastructure_url)
            logger.info(f"HAStructure engine connected successfully at {self.hastructure_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize HAStructure engine: {e}")
    
    def _get_engine(self, engine_type: str = "auto"):
        """
        Get the appropriate calculation engine based on availability and type
        
        Args:
            engine_type: The type of engine to use ('auto', 'local', or 'hastructure')
            
        Returns:
            The selected engine instance
            
        Raises:
            RuntimeError: If no engine is available
        """
        if engine_type == "auto":
            # Prefer HAStructure for production, fallback to local
            if self.engines["hastructure"] is not None:
                return self.engines["hastructure"]
            elif self.engines["local"] is not None:
                return self.engines["local"]
        elif engine_type == "hastructure" and self.engines["hastructure"] is not None:
            return self.engines["hastructure"]
        elif engine_type == "local" and self.engines["local"] is not None:
            return self.engines["local"]
        
        # If we get here, the requested engine is not available
        if self.engines["local"] is not None:
            logger.warning(f"Requested engine '{engine_type}' not available, using local engine")
            return self.engines["local"]
        
        # No engines available
        raise RuntimeError("No calculation engines are available")
    
    async def calculate_cashflows(self, 
                            deal_structure: Dict[str, Any],
                            engine_type: str = "auto",
                            use_cache: bool = True) -> Dict[str, Any]:
        """
        Calculate cashflows for a structured finance deal
        
        Args:
            deal_structure: The deal structure in AbsBox format
            engine_type: The type of engine to use ('auto', 'local', or 'hastructure')
            use_cache: Whether to use cache for the calculation
            
        Returns:
            Calculated cashflows and analytics
            
        Raises:
            ValueError: If the calculation fails
        """
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            # Create a deterministic cache key from the deal structure
            cache_key = f"hastructure:cashflows:{hash(json.dumps(deal_structure, sort_keys=True))}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                self.performance["cache_hits"] += 1
                return json.loads(cached_result)
        
        # Track calculation time
        start_time = time.time()
        
        try:
            # Get the appropriate engine
            engine = self._get_engine(engine_type)
            
            # Execute the calculation
            with CalculationTracker("cashflow_calculation"):
                result = engine.calculate(deal_structure)
            
            # Process and standardize the result
            standardized_result = self._standardize_result(result)
            
            # Update performance metrics
            calc_time = time.time() - start_time
            self._update_performance_metrics(calc_time)
            
            # Cache the result if caching is enabled
            if use_cache and cache_key:
                await self.cache.set(
                    cache_key, 
                    json.dumps(standardized_result),
                    expire=3600  # Cache for 1 hour
                )
            
            return standardized_result
        
        except Exception as e:
            self.performance["errors"] += 1
            logger.error(f"Error calculating cashflows: {e}")
            raise ValueError(f"Failed to calculate cashflows: {e}")
    
    async def run_scenario_analysis(self,
                               deal_structure: Dict[str, Any],
                               scenarios: List[Dict[str, Any]],
                               engine_type: str = "auto",
                               use_cache: bool = True) -> Dict[str, Any]:
        """
        Run multiple scenarios on a structured finance deal
        
        Args:
            deal_structure: The base deal structure in AbsBox format
            scenarios: List of scenario configurations to apply
            engine_type: The type of engine to use ('auto', 'local', or 'hastructure')
            use_cache: Whether to use cache for the calculation
            
        Returns:
            Results of all scenarios
            
        Raises:
            ValueError: If the scenario analysis fails
        """
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            # Create a deterministic cache key from the deal structure and scenarios
            cache_key = (
                f"hastructure:scenarios:{hash(json.dumps(deal_structure, sort_keys=True))}:"
                f"{hash(json.dumps(scenarios, sort_keys=True))}"
            )
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                self.performance["cache_hits"] += 1
                return json.loads(cached_result)
        
        # Track calculation time
        start_time = time.time()
        
        try:
            # Get the appropriate engine
            engine = self._get_engine(engine_type)
            
            # Execute the calculation for each scenario
            results = {}
            
            with CalculationTracker("scenario_analysis"):
                for i, scenario in enumerate(scenarios):
                    # Apply scenario configuration to the base deal
                    scenario_deal = self._apply_scenario(deal_structure, scenario)
                    
                    # Calculate this scenario
                    scenario_result = engine.calculate(scenario_deal)
                    
                    # Store the result
                    scenario_name = scenario.get("name", f"Scenario {i+1}")
                    results[scenario_name] = self._standardize_result(scenario_result)
            
            # Create a summary of the scenarios
            summary = self._create_scenario_summary(results)
            
            # Combine all results
            combined_result = {
                "scenarios": results,
                "summary": summary
            }
            
            # Update performance metrics
            calc_time = time.time() - start_time
            self._update_performance_metrics(calc_time)
            
            # Cache the result if caching is enabled
            if use_cache and cache_key:
                await self.cache.set(
                    cache_key, 
                    json.dumps(combined_result),
                    expire=3600  # Cache for 1 hour
                )
            
            return combined_result
        
        except Exception as e:
            self.performance["errors"] += 1
            logger.error(f"Error running scenario analysis: {e}")
            raise ValueError(f"Failed to run scenario analysis: {e}")
    
    async def validate_deal_structure(self,
                                 deal_structure: Dict[str, Any],
                                 engine_type: str = "auto") -> Dict[str, Any]:
        """
        Validate a deal structure against AbsBox and HAStructure requirements
        
        Args:
            deal_structure: The deal structure to validate
            engine_type: The type of engine to use for validation
            
        Returns:
            Validation results with any issues found
            
        Raises:
            ValueError: If the validation cannot be performed
        """
        try:
            # Get the appropriate engine
            engine = self._get_engine(engine_type)
            
            # Basic structure validation
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check for required sections
            required_sections = ["pool", "liabilities", "waterfall"]
            for section in required_sections:
                if section not in deal_structure:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Missing required section: {section}")
            
            # Check for pool data
            if "pool" in deal_structure:
                pool = deal_structure["pool"]
                if not isinstance(pool, list) or len(pool) == 0:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Pool must be a non-empty list of assets")
            
            # Perform engine-specific validation if possible
            try:
                # This will raise an exception if the structure is invalid
                engine.validate(deal_structure)
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Engine validation error: {str(e)}")
            
            return validation_results
        
        except Exception as e:
            logger.error(f"Error validating deal structure: {e}")
            raise ValueError(f"Failed to validate deal structure: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the HAStructure service
        
        Returns:
            Health status with details about engines and performance
        """
        status = {
            "hastructure_available": self.hastructure_available,
            "hastructure_url": self.hastructure_url,
            "local_engine_available": self.engines["local"] is not None,
            "hastructure_engine_available": self.engines["hastructure"] is not None,
            "performance": self.performance,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try a simple calculation as health check
        try:
            # Simple test deal
            test_deal = {
                "pool": [
                    {
                        "balance": 1000,
                        "rate": 0.05,
                        "term": 12,
                        "type": "fixed"
                    }
                ],
                "liabilities": [
                    {
                        "balance": 1000,
                        "rate": 0.04,
                        "name": "A"
                    }
                ],
                "waterfall": {
                    "normal": [
                        {"type": "interest", "source": "pool", "target": "A"},
                        {"type": "principal", "source": "pool", "target": "A"}
                    ]
                }
            }
            
            # Try with local engine first
            if self.engines["local"]:
                try:
                    start_time = time.time()
                    self.engines["local"].calculate(test_deal)
                    local_time = time.time() - start_time
                    status["local_engine_test"] = {
                        "success": True,
                        "calculation_time": local_time
                    }
                except Exception as e:
                    status["local_engine_test"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Then try HAStructure engine
            if self.engines["hastructure"]:
                try:
                    start_time = time.time()
                    self.engines["hastructure"].calculate(test_deal)
                    hastructure_time = time.time() - start_time
                    status["hastructure_engine_test"] = {
                        "success": True,
                        "calculation_time": hastructure_time
                    }
                except Exception as e:
                    status["hastructure_engine_test"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            status["health"] = "ok"
        except Exception as e:
            status["health"] = "error"
            status["error"] = str(e)
        
        return status
    
    # Helper methods
    
    def _standardize_result(self, result: Any) -> Dict[str, Any]:
        """
        Standardize engine result format for consistent API responses
        
        Args:
            result: Raw calculation result from engine
            
        Returns:
            Standardized result dictionary
        """
        # Convert engine-specific result to standard format
        try:
            if isinstance(result, dict):
                # Already a dictionary, ensure it has the expected structure
                standardized = {}
                
                # Extract cashflows
                if "cashflows" in result:
                    standardized["cashflows"] = result["cashflows"]
                elif "liabilities" in result:
                    # Extract cashflows from liabilities
                    standardized["cashflows"] = {
                        name: cf for name, cf in result["liabilities"].items()
                    }
                
                # Extract metrics
                if "metrics" in result:
                    standardized["metrics"] = result["metrics"]
                else:
                    # Calculate basic metrics if not provided
                    standardized["metrics"] = self._calculate_basic_metrics(result)
                
                # Include original structure
                if "original_structure" in result:
                    standardized["original_structure"] = result["original_structure"]
                
                return standardized
            else:
                # Attempt to convert to dictionary
                return {
                    "result": result,
                    "metrics": self._calculate_basic_metrics(result)
                }
        except Exception as e:
            logger.error(f"Error standardizing result: {e}")
            # Return a minimal structure with the original result
            return {"raw_result": str(result)}
    
    def _calculate_basic_metrics(self, result: Any) -> Dict[str, Any]:
        """
        Calculate basic metrics from calculation result
        
        Args:
            result: Raw calculation result
            
        Returns:
            Dictionary of basic metrics
        """
        # Calculate simple metrics based on the result structure
        metrics = {}
        
        try:
            # Extract cashflows if available
            cashflows = None
            if isinstance(result, dict):
                if "cashflows" in result:
                    cashflows = result["cashflows"]
                elif "liabilities" in result:
                    cashflows = result["liabilities"]
            
            if cashflows:
                # Calculate simple metrics for each cashflow series
                for name, cf in cashflows.items():
                    if isinstance(cf, list) and len(cf) > 0:
                        # Total cash
                        total = sum(item.get("amount", 0) for item in cf)
                        # Present value (simple calculation)
                        pv = sum(
                            item.get("amount", 0) / (1 + 0.05) ** (i / 12)
                            for i, item in enumerate(cf)
                        )
                        
                        metrics[name] = {
                            "total": total,
                            "present_value": pv,
                            "count": len(cf)
                        }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
        
        return metrics
    
    def _apply_scenario(self, 
                       base_deal: Dict[str, Any], 
                       scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply scenario configuration to base deal structure
        
        Args:
            base_deal: Base deal structure
            scenario: Scenario configuration to apply
            
        Returns:
            Modified deal structure for this scenario
        """
        # Create a deep copy of the base deal
        import copy
        scenario_deal = copy.deepcopy(base_deal)
        
        # Apply scenario parameters
        if "pool" in scenario:
            # Apply pool-level changes
            if "pool" in scenario_deal:
                for pool_change in scenario["pool"]:
                    pool_index = pool_change.get("index")
                    if pool_index is not None and 0 <= pool_index < len(scenario_deal["pool"]):
                        for key, value in pool_change.items():
                            if key != "index":
                                scenario_deal["pool"][pool_index][key] = value
        
        if "liabilities" in scenario:
            # Apply liability-level changes
            if "liabilities" in scenario_deal:
                for liability_change in scenario["liabilities"]:
                    liability_index = liability_change.get("index")
                    if liability_index is not None and 0 <= liability_index < len(scenario_deal["liabilities"]):
                        for key, value in liability_change.items():
                            if key != "index":
                                scenario_deal["liabilities"][liability_index][key] = value
        
        if "assumptions" in scenario:
            # Apply assumption changes
            if "assumptions" not in scenario_deal:
                scenario_deal["assumptions"] = {}
            
            for key, value in scenario["assumptions"].items():
                scenario_deal["assumptions"][key] = value
        
        return scenario_deal
    
    def _create_scenario_summary(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of scenario analysis results
        
        Args:
            scenario_results: Dictionary of results for each scenario
            
        Returns:
            Summary metrics across scenarios
        """
        summary = {
            "scenario_count": len(scenario_results),
            "metrics": {},
            "comparison": {}
        }
        
        try:
            # Extract common metrics for comparison
            all_metrics = {}
            
            for scenario_name, result in scenario_results.items():
                if "metrics" in result:
                    for metric_name, metric_value in result["metrics"].items():
                        if isinstance(metric_value, dict):
                            for sub_metric, value in metric_value.items():
                                full_metric_name = f"{metric_name}.{sub_metric}"
                                if full_metric_name not in all_metrics:
                                    all_metrics[full_metric_name] = {}
                                all_metrics[full_metric_name][scenario_name] = value
                        else:
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = {}
                            all_metrics[metric_name][scenario_name] = metric_value
            
            # Calculate summary statistics for each metric
            for metric_name, scenario_values in all_metrics.items():
                # Filter numeric values
                numeric_values = [v for v in scenario_values.values() 
                                if isinstance(v, (int, float)) and not pd.isna(v)]
                
                if numeric_values:
                    summary["metrics"][metric_name] = {
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "mean": sum(numeric_values) / len(numeric_values),
                        "range": max(numeric_values) - min(numeric_values),
                        "scenarios": scenario_values
                    }
            
            # Create comparison table
            # Select up to 5 key metrics for comparison
            key_metrics = list(summary["metrics"].keys())[:5]
            
            for metric in key_metrics:
                summary["comparison"][metric] = {
                    scenario_name: scenario_values.get(metric, "N/A")
                    for scenario_name, scenario_values in scenario_results.items()
                }
        
        except Exception as e:
            logger.error(f"Error creating scenario summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def _update_performance_metrics(self, calculation_time: float):
        """
        Update service performance metrics
        
        Args:
            calculation_time: Time taken for the calculation in seconds
        """
        self.performance["calculations"] += 1
        
        # Update average calculation time
        current_avg = self.performance["avg_calculation_time"]
        current_count = self.performance["calculations"]
        
        if current_count > 1:
            self.performance["avg_calculation_time"] = (
                (current_avg * (current_count - 1) + calculation_time) / current_count
            )
        else:
            self.performance["avg_calculation_time"] = calculation_time
