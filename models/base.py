from abc import ABC, abstractmethod
from core.instruments import Option
from core.market_data import MarketData

class PricingEngine(ABC):
    """
    Abstract base class for all pricing engines. 
    Defines capabilities and standardizes the interface.
    """
    
    # Metadata for UI Dynamic Filtering
    SUPPORTED_STYLES = ['European']
    SUPPORTED_EXOTICS = ['None']

    def __init__(self, market_data: MarketData):
        self.market_data = market_data

    @classmethod
    def check_compatibility(cls, option: Option) -> bool:
        """
        Checks if this engine is mathematically capable of pricing the given option.
        """
        if option.style not in cls.SUPPORTED_STYLES:
            return False
        if option.exotic_type not in cls.SUPPORTED_EXOTICS:
            return False
        return True

    @abstractmethod
    def calculate_price(self, option: Option) -> float:
        """
        Core pricing logic. Must be implemented by specific engines.
        """
        pass

    @abstractmethod
    def calculate_greeks(self, option: Option) -> dict:
        """
        Core Greek calculation logic. 
        Should return a dictionary: {'delta': ..., 'gamma': ..., etc.}
        """
        pass

    def price(self, option: Option):
        """
        Standardized public method to safely price an instrument.
        """
        if not self.check_compatibility(option):
            raise TypeError(
                f"{self.__class__.__name__} cannot price {option.style} "
                f"options with exotic feature '{option.exotic_type}'."
            )
            
        return self.calculate_price(option)