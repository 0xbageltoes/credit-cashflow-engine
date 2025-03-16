"""
Asset Class Service

Provides functionality for working with different asset classes in structured finance,
implementing specialized behavior for each supported asset class:
- Residential Mortgages
- Auto Loans
- Consumer Credit
- Commercial Loans
- CLOs/CDOs

Production-ready implementation with comprehensive error handling and Redis caching.
"""
import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date
from functools import lru_cache

import numpy as np
import pandas as pd

# Import AbsBox libraries
import absbox as ab

# Setup logging
logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.cache_service import CacheService
from app.core.monitoring import CALCULATION_TIME, CalculationTracker
from app.models.asset_classes import (
    AssetClass, BaseAsset, AssetPool, 
    ResidentialMortgage, AutoLoan, ConsumerCredit, 
    CommercialLoan, CLOCDO,
    AssetPoolAnalysisRequest, AssetPoolAnalysisResponse,
    AssetPoolCashflow, AssetPoolMetrics, AssetPoolStressTest
)
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced
from app.services.asset_handlers.factory import AssetHandlerFactory

class AssetClassService:
    """Service for handling different asset classes in structured finance"""
    
    def __init__(self, cache: Optional[CacheService] = None):
        """
        Initialize the asset class service with production-ready components
        
        Args:
            cache: Optional CacheService instance (created if not provided)
        """
        self.cache = cache or CacheService()
        # Initialize AbsBox service for core functionality
        self.absbox_service = AbsBoxServiceEnhanced()
        
        # Initialize handler factory
        self.handler_factory = AssetHandlerFactory(absbox_service=self.absbox_service)
        
        logger.info("AssetClassService initialized")
        
    def generate_cache_key(self, request: AssetPoolAnalysisRequest, user_id: str) -> str:
        """
        Generate a deterministic cache key for asset pool analysis
        
        Args:
            request: AssetPoolAnalysisRequest object
            user_id: User ID for isolation of cached results
            
        Returns:
            str: Unique cache key
        """
        # Convert the request model to a string and hash it for a consistent key
        request_dict = request.model_dump(exclude={'analysis_date', 'pricing_date'})
        request_hash = hashlib.sha256(json.dumps(request_dict, sort_keys=True, default=str).encode()).hexdigest()
        pool_name = request.pool.pool_name.replace(" ", "_").lower()
        
        return f"asset_pool_analysis:{user_id}:{pool_name}:{request_hash}"
    
    async def analyze_asset_pool(
        self, 
        request: AssetPoolAnalysisRequest, 
        user_id: str, 
        use_cache: bool = True
    ) -> AssetPoolAnalysisResponse:
        """
        Analyze an asset pool with production-quality implementation
        
        Args:
            request: The analysis request with pool details
            user_id: User ID for cache isolation and results tracking
            use_cache: Whether to use caching (default: True)
            
        Returns:
            AssetPoolAnalysisResponse with analysis results
        """
        start_time = time.time()
        cache_key = None
        
        try:
            # Generate cache key if caching is enabled
            if use_cache:
                cache_key = self.generate_cache_key(request, user_id)
                # Check cache with proper error handling
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for asset pool analysis: {request.pool.pool_name}")
                    return cached_result
            
            # Cache miss or caching disabled, perform analysis
            logger.info(f"Analyzing asset pool: {request.pool.pool_name}")
            
            # Process the pool based on predominant asset class
            asset_class = self._determine_predominant_asset_class(request.pool)
            logger.info(f"Determined predominant asset class: {asset_class}")
            
            # Check if we have a handler for this asset class
            if not self.handler_factory.is_supported(asset_class):
                logger.warning(f"No specialized handler for asset class: {asset_class}, using generic analysis")
                result = self._analyze_generic_pool(request)
            else:
                # Get the appropriate handler and analyze
                handler = self.handler_factory.get_handler(asset_class)
                result = handler.analyze_pool(request)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Cache the result if caching is enabled
            if use_cache and cache_key:
                await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error analyzing asset pool: {str(e)}")
            execution_time = time.time() - start_time
            
            # Return a proper error response with production-quality details
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date or datetime.now().date(),
                execution_time=execution_time,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def _get_cached_result(self, cache_key: str) -> Optional[AssetPoolAnalysisResponse]:
        """
        Get cached result with proper error handling
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached AssetPoolAnalysisResponse or None if not found/error
        """
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                return AssetPoolAnalysisResponse(**json.loads(cached_data))
        except Exception as e:
            # Log error but allow the process to continue by returning None
            logger.warning(f"Error retrieving from cache: {str(e)}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: AssetPoolAnalysisResponse, ttl: int = 3600) -> bool:
        """
        Cache analysis result with proper error handling
        
        Args:
            cache_key: Cache key to store under
            result: Analysis result to cache
            ttl: Time-to-live in seconds (default: 1 hour)
            
        Returns:
            bool: True if caching succeeded, False otherwise
        """
        try:
            serialized_result = json.dumps(result.model_dump())
            await self.cache.set(cache_key, serialized_result, ttl=ttl)
            logger.debug(f"Cached result under key: {cache_key} with TTL: {ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")
            return False
    
    def _determine_predominant_asset_class(self, pool: AssetPool) -> AssetClass:
        """
        Determine the predominant asset class in a pool
        
        Args:
            pool: Asset pool to analyze
            
        Returns:
            AssetClass: The predominant asset class
        """
        if not pool.assets:
            raise ValueError("Asset pool is empty")
            
        # Count assets by class
        class_counts = {}
        class_balances = {}
        
        for asset in pool.assets:
            asset_class = asset.asset_class
            class_counts[asset_class] = class_counts.get(asset_class, 0) + 1
            class_balances[asset_class] = class_balances.get(asset_class, 0) + asset.balance
        
        # Find predominant class by balance (preferred) or count
        if class_balances:
            predominant_class = max(class_balances.items(), key=lambda x: x[1])[0]
        else:
            predominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
        return predominant_class
    
    def _analyze_generic_pool(self, request: AssetPoolAnalysisRequest) -> AssetPoolAnalysisResponse:
        """
        Generic analysis for asset pools without specialized handlers
        
        Args:
            request: Analysis request
            
        Returns:
            AssetPoolAnalysisResponse with analysis results
        """
        try:
            logger.info(f"Performing generic analysis for pool: {request.pool.pool_name}")
            
            # Calculate simple metrics based on pool characteristics
            total_balance = sum(asset.balance for asset in request.pool.assets)
            weighted_rate = sum(asset.balance * asset.rate for asset in request.pool.assets) / total_balance if total_balance > 0 else 0
            
            # Create simple cashflows (principal + interest only)
            cashflows = []
            analysis_date = request.analysis_date or datetime.now().date()
            
            # Determine max term
            max_term = max(asset.term_months for asset in request.pool.assets)
            
            # Simple amortization calculation
            remaining_balance = total_balance
            monthly_rate = weighted_rate / 12
            
            for i in range(max_term):
                # Simple payment calculation (principal + interest)
                if monthly_rate > 0:
                    payment = remaining_balance * monthly_rate * (1 + monthly_rate) ** (max_term - i) / ((1 + monthly_rate) ** (max_term - i) - 1)
                else:
                    payment = remaining_balance / (max_term - i) if (max_term - i) > 0 else remaining_balance
                
                interest_payment = remaining_balance * monthly_rate
                principal_payment = payment - interest_payment
                
                # Cap principal at remaining balance
                principal_payment = min(principal_payment, remaining_balance)
                
                # Update remaining balance
                new_balance = remaining_balance - principal_payment
                
                # Create cashflow
                cf = AssetPoolCashflow(
                    period=i,
                    date=analysis_date.replace(month=analysis_date.month + i) if i == 0 else \
                        analysis_date.replace(month=analysis_date.month + i % 12, 
                                            year=analysis_date.year + i // 12),
                    scheduled_principal=principal_payment,
                    scheduled_interest=interest_payment,
                    prepayment=0.0,
                    default=0.0,
                    recovery=0.0,
                    loss=0.0,
                    balance=new_balance
                )
                
                cashflows.append(cf)
                remaining_balance = new_balance
                
                # Break if balance is paid off
                if remaining_balance <= 0.01:
                    break
            
            # Calculate simple metrics
            total_principal = sum(cf.scheduled_principal for cf in cashflows)
            total_interest = sum(cf.scheduled_interest for cf in cashflows)
            
            # Calculate NPV
            npv = 0.0
            discount_rate = request.discount_rate or 0.05
            for i, cf in enumerate(cashflows):
                monthly_rate = (1 + discount_rate) ** (1/12) - 1
                period_cf = cf.scheduled_principal + cf.scheduled_interest
                npv += period_cf / ((1 + monthly_rate) ** (i+1))
            
            # Calculate simple WAL
            wal = 0.0
            if total_principal > 0:
                for i, cf in enumerate(cashflows):
                    t = (i+1) / 12
                    wal += t * cf.scheduled_principal / total_principal
            
            # Create metrics
            metrics = AssetPoolMetrics(
                total_principal=total_principal,
                total_interest=total_interest,
                total_cashflow=total_principal + total_interest,
                npv=npv,
                weighted_average_life=wal
            )
            
            # Create response
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=analysis_date,
                execution_time=0.0,  # Will be updated by caller
                status="success",
                metrics=metrics,
                cashflows=cashflows if request.include_cashflows else None,
                stress_tests=None  # No stress testing in generic analysis
            )
            
        except Exception as e:
            logger.exception(f"Error in generic pool analysis: {str(e)}")
            
            # Return error response
            return AssetPoolAnalysisResponse(
                pool_name=request.pool.pool_name,
                analysis_date=request.analysis_date or datetime.now().date(),
                execution_time=0.0,  # Will be updated by caller
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
    
    def get_supported_asset_classes(self) -> list[AssetClass]:
        """
        Get a list of supported asset classes
        
        Returns:
            list: List of supported asset classes
        """
        return self.handler_factory.supported_asset_classes()
