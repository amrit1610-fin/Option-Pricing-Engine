import numpy as np
from scipy.stats import norm

from models.base import PricingEngine
from core.instruments import Option
from core.market_data import MarketData

class BlackScholesEngine(PricingEngine):
    """
    Analytical Black-Scholes-Merton Pricing Engine.
    Mathematically limited to standard European Options.
    """
    
    # Engine Capabilities (UI Dynamic Metadata)
    SUPPORTED_STYLES = ['European']
    SUPPORTED_EXOTICS = ['None']

    def __init__(self, market_data: MarketData):
        super().__init__(market_data)

    def calculate_price(self, option: Option) -> float:
        S = self.market_data.spot_price
        K = option.strike
        T = self.market_data.time_to_expiry
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        sigma = self.market_data.volatility

        if T <= 0:
            # If expired, return immediate intrinsic value
            return max(S - K, 0) if option.option_type == 'call' else max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option.option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def calculate_greeks(self, option: Option) -> dict:
        """Calculates closed-form analytical Greeks."""
        S = self.market_data.spot_price
        K = option.strike
        T = self.market_data.time_to_expiry
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        sigma = self.market_data.volatility

        if T <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option.option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)

        # Gamma (Same for Call and Put)
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

        # Vega (Same for Call and Put, scaled to 1% change)
        vega = (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)) / 100

        # Theta (Scaled to 1 day decay)
        term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option.option_type == 'call':
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
            theta = (term1 - term2 + term3) / 365
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta = (term1 + term2 - term3) / 365

        # Rho (Scaled to 1 bp change)
        if option.option_type == 'call':
            rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        else:
            rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }