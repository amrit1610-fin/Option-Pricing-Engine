import numpy as np
import copy

from models.base import PricingEngine
from core.instruments import Option, EuropeanOption
from core.market_data import MarketData

class MonteCarloEngine(PricingEngine):
    """
    Monte Carlo Simulation Engine using Geometric Brownian Motion (GBM).
    Handles path-dependent Exotic options seamlessly.
    """
    
    # Engine Capabilities (UI Dynamic Metadata)
    # Note: American options via LSMC will be added in Phase 4
    SUPPORTED_STYLES = ['European'] 
    SUPPORTED_EXOTICS = ['None', 'Asian', 'Barrier']

    def __init__(self, market_data: MarketData, num_paths: int = 50000, time_steps: int = 252):
        super().__init__(market_data)
        self.num_paths = num_paths
        self.time_steps = time_steps

    def _generate_paths(self, seed: int = None) -> np.ndarray:
        """Generates the matrix of simulated stock paths."""
        if seed is not None:
            np.random.seed(seed)
            
        S0 = self.market_data.spot_price
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        T = self.market_data.time_to_expiry
        sigma = self.market_data.volatility
        
        dt = T / self.time_steps
        
        # Antithetic Variates for variance reduction
        Z = np.random.standard_normal((int(self.num_paths / 2), self.time_steps))
        Z = np.concatenate((Z, -Z), axis=0)
        
        paths = np.zeros((self.num_paths, self.time_steps + 1))
        paths[:, 0] = S0
        
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        for t in range(1, self.time_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
            
        return paths

    def calculate_price(self, option: Option, seed: int = None, return_se: bool = False):
        """
        Executes the pricing. 
        Passes the entire path matrix to the Instrument's payoff function.
        """
        paths = self._generate_paths(seed)
        
        # 1. The magic of OOP: The Option calculates its own exotic payoff
        simulated_payoffs = option.get_payoff(paths)
        
        r = self.market_data.risk_free_rate
        T = self.market_data.time_to_expiry
        discount_factor = np.exp(-r * T)
        
        # 2. Apply Control Variate variance reduction ONLY for standard European options
        if isinstance(option, EuropeanOption):
            terminal_prices = paths[:, -1]
            expected_terminal_price = self.market_data.spot_price * np.exp((r - self.market_data.dividend_yield) * T)
            
            covariance = np.cov(simulated_payoffs, terminal_prices)[0, 1]
            variance = np.var(terminal_prices)
            c = covariance / variance if variance > 0 else 0
            
            adjusted_payoffs = simulated_payoffs - c * (terminal_prices - expected_terminal_price)
            price = discount_factor * np.mean(adjusted_payoffs)
            se = np.std(adjusted_payoffs * discount_factor) / np.sqrt(self.num_paths)
        else:
            # Standard Monte Carlo mean for Exotics (Asian/Barrier)
            price = discount_factor * np.mean(simulated_payoffs)
            se = np.std(simulated_payoffs * discount_factor) / np.sqrt(self.num_paths)
            
        return (price, se) if return_se else price

    def calculate_greeks(self, option: Option) -> dict:
        """
        Calculates Greeks using Finite Difference Method.
        CRITICAL: We pass a fixed random seed so that simulation noise 
        doesn't corrupt the tiny bumps used to calculate the derivatives.
        """
        FIXED_SEED = 42 
        
        dS = self.market_data.spot_price * 0.01
        dVol = 0.01
        dT = 1 / 365
        dR = 0.0001

        # Helper to quickly re-price with bumped market data
        def get_price(bumped_data: MarketData) -> float:
            temp_engine = MonteCarloEngine(bumped_data, self.num_paths, self.time_steps)
            return temp_engine.calculate_price(option, seed=FIXED_SEED)

        base_price = self.calculate_price(option, seed=FIXED_SEED)
        md = self.market_data
        
        # Spot bumps
        md_up_s = copy.copy(md)
        md_up_s.spot_price += dS
        md_dn_s = copy.copy(md)
        md_dn_s.spot_price -= dS
        
        # Vol bumps
        md_up_vol = copy.copy(md)
        md_up_vol.volatility += dVol
        md_dn_vol = copy.copy(md)
        md_dn_vol.volatility -= dVol
        
        # Time and Rate bumps
        md_time_pass = copy.copy(md)
        md_time_pass.time_to_expiry -= dT
        md_up_r = copy.copy(md)
        md_up_r.risk_free_rate += dR
        md_dn_r = copy.copy(md)
        md_dn_r.risk_free_rate -= dR

        # Derivative calculations
        delta = (get_price(md_up_s) - get_price(md_dn_s)) / (2 * dS)
        gamma = (get_price(md_up_s) - 2 * base_price + get_price(md_dn_s)) / (dS ** 2)
        vega = (get_price(md_up_vol) - get_price(md_dn_vol)) / (2 * dVol)
        theta = (get_price(md_time_pass) - base_price) / dT
        rho = (get_price(md_up_r) - get_price(md_dn_r)) / (2 * dR)

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,      
            'theta': theta / 365,    
            'rho': rho / 100         
        }