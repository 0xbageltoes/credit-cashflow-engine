"""
Asset Handler Factory

Production-ready factory pattern implementation for creating appropriate asset handlers
based on asset class type with comprehensive error handling and logging.
"""
import logging
from typing import Dict, Optional, Type, Any

from app.core.monitoring import CalculationTracker
from app.models.asset_classes import AssetClass
from app.services.absbox_service_enhanced import AbsBoxServiceEnhanced

# Import handlers
from app.services.asset_handlers.residential_mortgage import ResidentialMortgageHandler
from app.services.asset_handlers.auto_loan import AutoLoanHandler

# Setup logging
logger = logging.getLogger(__name__)

class AssetHandlerFactory:
    """
    Factory for creating appropriate asset handlers based on asset class
    
    Production-ready implementation with proper registration and resolution
    of handlers for different asset classes.
    """
    
    def __init__(self, absbox_service: Optional[AbsBoxServiceEnhanced] = None):
        """
        Initialize the asset handler factory
        
        Args:
            absbox_service: Optional AbsBox service to pass to handlers
        """
        self.absbox_service = absbox_service or AbsBoxServiceEnhanced()
        self._handlers: Dict[AssetClass, Any] = {}
        
        # Register handlers
        self._register_handlers()
        
        logger.info("AssetHandlerFactory initialized with handlers")
    
    def _register_handlers(self) -> None:
        """Register all available asset handlers"""
        try:
            # Register the residential mortgage handler
            self._handlers[AssetClass.RESIDENTIAL_MORTGAGE] = ResidentialMortgageHandler(
                absbox_service=self.absbox_service
            )
            
            # Register the auto loan handler
            self._handlers[AssetClass.AUTO_LOAN] = AutoLoanHandler(
                absbox_service=self.absbox_service
            )
            
            # TODO: Register handlers for other asset classes as they're implemented
            # self._handlers[AssetClass.CONSUMER_CREDIT] = ConsumerCreditHandler(...)
            # self._handlers[AssetClass.COMMERCIAL_LOAN] = CommercialLoanHandler(...)
            # self._handlers[AssetClass.CLO_CDO] = CLOCDOHandler(...)
            
            logger.info(f"Registered handlers for {len(self._handlers)} asset classes")
        except Exception as e:
            logger.exception(f"Error registering asset handlers: {str(e)}")
            # Don't raise - still allow factory to operate with partial handler set
    
    def get_handler(self, asset_class: AssetClass) -> Any:
        """
        Get the appropriate handler for an asset class
        
        Args:
            asset_class: The asset class to get a handler for
            
        Returns:
            The appropriate handler
            
        Raises:
            ValueError: If no handler is registered for the asset class
        """
        with CalculationTracker(f"get_handler_{asset_class}"):
            if asset_class not in self._handlers:
                logger.error(f"No handler registered for asset class: {asset_class}")
                raise ValueError(f"No handler registered for asset class: {asset_class}")
                
            logger.debug(f"Retrieved handler for asset class: {asset_class}")
            return self._handlers[asset_class]
    
    def is_supported(self, asset_class: AssetClass) -> bool:
        """
        Check if an asset class is supported
        
        Args:
            asset_class: The asset class to check
            
        Returns:
            bool: True if supported, False otherwise
        """
        return asset_class in self._handlers
        
    def register_handler(self, asset_class: AssetClass, handler: Any) -> None:
        """
        Register a new handler (useful for testing with mocks)
        
        Args:
            asset_class: The asset class to register
            handler: The handler to register
        """
        self._handlers[asset_class] = handler
        logger.info(f"Registered custom handler for asset class: {asset_class}")
        
    def supported_asset_classes(self) -> list[AssetClass]:
        """
        Get the list of supported asset classes
        
        Returns:
            list: List of supported asset classes
        """
        return list(self._handlers.keys())
