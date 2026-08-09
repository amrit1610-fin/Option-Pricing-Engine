from abc import ABC, abstractmethod
import numpy as np

class Option(ABC):
    """Abstract base class for all option contracts."""
    
    def __init__(self, strike: float, option_type: str):
        self.strike = strike
        self.option_type = option_type.lower()
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")

    @property
    def style(self) -> str:
        """Returns the exercise style of the option (e.g., 'European', 'American')."""
        return "Unknown"

    @property
    def exotic_type(self) -> str:
        """Returns the exotic feature of the option (e.g., 'None', 'Asian', 'Barrier')."""
        return "None"

    @abstractmethod
    def get_payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff of the option given a matrix of price paths.
        Expected shape of paths: (num_paths, num_time_steps)
        Returns a 1D array of payoffs of shape (num_paths,)
        """
        pass


class EuropeanOption(Option):
    
    @property
    def style(self) -> str:
        return "European"

    def get_payoff(self, paths: np.ndarray) -> np.ndarray:
        # European options only care about the terminal (final) price
        terminal_prices = paths[:, -1] if paths.ndim > 1 else paths
        
        if self.option_type == 'call':
            return np.maximum(terminal_prices - self.strike, 0)
        else:
            return np.maximum(self.strike - terminal_prices, 0)


class AmericanOption(Option):
    
    @property
    def style(self) -> str:
        return "American"

    def get_payoff(self, paths: np.ndarray) -> np.ndarray:
        # For American options, the standard payoff function only represents the intrinsic 
        # value at a specific point in time. The Engine handles the early exercise logic.
        prices = paths[:, -1] if paths.ndim > 1 else paths
        
        if self.option_type == 'call':
            return np.maximum(prices - self.strike, 0)
        else:
            return np.maximum(self.strike - prices, 0)


class AsianOption(Option):
    
    def __init__(self, strike: float, option_type: str, averaging_type: str = 'arithmetic'):
        super().__init__(strike, option_type)
        self.averaging_type = averaging_type.lower()

    @property
    def style(self) -> str:
        return "European" # Usually European exercise, but path-dependent payoff

    @property
    def exotic_type(self) -> str:
        return "Asian"

    def get_payoff(self, paths: np.ndarray) -> np.ndarray:
        # Asian options depend on the average price over the path
        if paths.ndim == 1:
            raise ValueError("Asian options require a full price path, not just a terminal price.")
            
        if self.averaging_type == 'arithmetic':
            average_prices = np.mean(paths, axis=1)
        elif self.averaging_type == 'geometric':
            # Compute geometric mean safely
            average_prices = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError("Averaging type must be 'arithmetic' or 'geometric'.")

        if self.option_type == 'call':
            return np.maximum(average_prices - self.strike, 0)
        else:
            return np.maximum(self.strike - average_prices, 0)


class BarrierOption(Option):
    
    def __init__(self, strike: float, option_type: str, barrier_level: float, barrier_type: str = 'up-and-out'):
        super().__init__(strike, option_type)
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type.lower()
        
        valid_types = ['up-and-out', 'up-and-in', 'down-and-out', 'down-and-in']
        if self.barrier_type not in valid_types:
            raise ValueError(f"Barrier type must be one of {valid_types}")

    @property
    def style(self) -> str:
        return "European"

    @property
    def exotic_type(self) -> str:
        return "Barrier"

    def get_payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 1:
            raise ValueError("Barrier options require a full price path.")
            
        terminal_prices = paths[:, -1]
        
        # Calculate standard European payoff first
        if self.option_type == 'call':
            standard_payoff = np.maximum(terminal_prices - self.strike, 0)
        else:
            standard_payoff = np.maximum(self.strike - terminal_prices, 0)
            
        # Apply barrier conditions
        if self.barrier_type == 'up-and-out':
            knocked_out = np.max(paths, axis=1) >= self.barrier_level
            return np.where(knocked_out, 0.0, standard_payoff)
            
        elif self.barrier_type == 'up-and-in':
            knocked_in = np.max(paths, axis=1) >= self.barrier_level
            return np.where(knocked_in, standard_payoff, 0.0)
            
        elif self.barrier_type == 'down-and-out':
            knocked_out = np.min(paths, axis=1) <= self.barrier_level
            return np.where(knocked_out, 0.0, standard_payoff)
            
        elif self.barrier_type == 'down-and-in':
            knocked_in = np.min(paths, axis=1) <= self.barrier_level
            return np.where(knocked_in, standard_payoff, 0.0)