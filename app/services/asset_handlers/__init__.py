"""
Asset Handlers Package

This package contains handlers for different asset classes in structured finance.
Each handler provides specialized analysis for a specific asset class.

The implementation follows production best practices:
- Comprehensive error handling
- Detailed logging for monitoring
- Integration with Redis caching
- Performance optimization
"""
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

# Export handlers
from app.services.asset_handlers.consumer_credit import ConsumerCreditHandler
from app.services.asset_handlers.commercial_loan import CommercialLoanHandler  
from app.services.asset_handlers.clo_cdo import CLOCDOHandler

__all__ = [
    'ConsumerCreditHandler',
    'CommercialLoanHandler',
    'CLOCDOHandler'
]

# Log initialization
logger.info("Asset handlers module initialized with handlers for: Consumer Credit, Commercial Loans, CLO/CDO")
