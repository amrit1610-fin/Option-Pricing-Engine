import numpy as np
import copy

from models.base import PricingEngine
from core.instruments import Option
from core.market_data import MarketData

class HestonFourierEngine(PricingEngine):
    """
    Fourier transform pricing for Heston Model using the Carr-Madan (1999) FFT method.
    Fast and accurate, but restricted to European options.
    """
    
    SUPPORTED_STYLES = ['European']
    SUPPORTED_EXOTICS = ['None']

    def __init__(self, market_data: MarketData, v0: float, rho: float, kappa: float, theta: float, sigma_v: float):
        super().__init__(market_data)
        self.v0 = v0
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        
        self.x0 = np.log(self.market_data.spot_price)
        self.i = 1j

    def _cf(self, u):
        """Heston Characteristic Function (Albrecher formulation)"""
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        T = self.market_data.time_to_expiry
        
        a = self.kappa * self.theta
        b = self.kappa - (self.rho * self.sigma_v * self.i * u)
        d = np.sqrt(b**2 + self.sigma_v**2 * (self.i * u + u**2))
        g = (b - d) / (b + d)

        eDT = np.exp(-d * T)
        one_minus_g_eDT = 1 - g * eDT
        one_minus_g     = 1 - g
        
        one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
        one_minus_g     = np.where(np.abs(one_minus_g)     < 1e-15, 1e-15, one_minus_g)

        C = self.i * u * (r - q) * T + (a / (self.sigma_v**2)) * ((b - d) * T - 2.0 * np.log(one_minus_g_eDT / one_minus_g))
        D = ((b - d) / (self.sigma_v**2)) * ((1 - eDT) / one_minus_g_eDT)
        
        return np.exp(C + D * self.v0 + self.i * u * self.x0)

    def fft_calls(self, N: int = 4096, eta: float = 0.25, alpha: float = 1.5):
        """Computes call prices over a grid of strikes using FFT."""
        r = self.market_data.risk_free_rate
        T = self.market_data.time_to_expiry
        
        n = np.arange(N)
        v = eta * n
        u = v - (alpha + 1) * self.i
        ert = np.exp(-r * T)
        
        psi = (ert * self._cf(u)) / (alpha**2 + alpha - v**2 + self.i * (2 * alpha + 1) * v)
        
        w = np.ones(N)
        w[1:N-1:2] = 4
        w[2:N-2:2] = 2
        w = w * (eta / 3.0)

        lam = 2.0 * np.pi / (N * eta)   
        b   = 0.5 * N * lam             
        x   = psi * np.exp(self.i * b * v) * w

        F = np.fft.fft(x)
        F = np.real(F) 

        j = np.arange(N)
        k = -b + j * lam                
        K = np.exp(k)

        calls = np.exp(-alpha * k) / np.pi * F
        order = np.argsort(K)
        return K[order], np.maximum(calls[order], 0.0)

    def calculate_price(self, option: Option) -> float:
        N, eta, alpha = 4096, 0.25, 1.5
        K_grid, C_grid = self.fft_calls(N=N, eta=eta, alpha=alpha)
        
        # Linear interpolation
        if option.strike <= K_grid[0]:
            call_price = C_grid[0]
        elif option.strike >= K_grid[-1]:
            call_price = C_grid[-1]
        else:
            idx = np.searchsorted(K_grid, option.strike)
            x0, x1 = K_grid[idx-1], K_grid[idx]
            y0, y1 = C_grid[idx-1], C_grid[idx]
            call_price = y0 + (y1 - y0) * (option.strike - x0) / (x1 - x0)
            
        if option.option_type == 'call':
            return call_price
        else:
            # Put-Call Parity
            S = self.market_data.spot_price
            q = self.market_data.dividend_yield
            r = self.market_data.risk_free_rate
            T = self.market_data.time_to_expiry
            return call_price - S * np.exp(-q * T) + option.strike * np.exp(-r * T)

    def calculate_greeks(self, option: Option) -> dict:
        """Finite Differences for Greeks using the fast Fourier Engine."""
        dS = self.market_data.spot_price * 0.01
        dVol = 0.01          # Bump initial variance
        dT = 1 / 365
        dR = 0.0001

        base_price = self.calculate_price(option)

        def get_price(bumped_md: MarketData, v0_bump: float = 0.0):
            temp_engine = HestonFourierEngine(bumped_md, self.v0 + v0_bump, self.rho, self.kappa, self.theta, self.sigma_v)
            return temp_engine.calculate_price(option)

        md = self.market_data
        
        md_up_s = copy.copy(md); md_up_s.spot_price += dS
        md_dn_s = copy.copy(md); md_dn_s.spot_price -= dS
        
        md_time_pass = copy.copy(md); md_time_pass.time_to_expiry -= dT
        
        md_up_r = copy.copy(md); md_up_r.risk_free_rate += dR
        md_dn_r = copy.copy(md); md_dn_r.risk_free_rate -= dR

        delta = (get_price(md_up_s) - get_price(md_dn_s)) / (2 * dS)
        gamma = (get_price(md_up_s) - 2 * base_price + get_price(md_dn_s)) / (dS ** 2)
        vega = (get_price(md, dVol) - get_price(md, -dVol)) / (2 * dVol)
        theta = (get_price(md_time_pass) - base_price) / dT
        rho = (get_price(md_up_r) - get_price(md_dn_r)) / (2 * dR)

        return {'delta': delta, 'gamma': gamma, 'vega': vega / 100, 'theta': theta / 365, 'rho': rho / 100}


class HestonMonteCarloEngine(PricingEngine):
    """
    Monte Carlo simulation of the Heston Model using the Full Truncation scheme.
    Because it simulates paths, it naturally supports Path-Dependent Exotics.
    """
    
    SUPPORTED_STYLES = ['European']
    SUPPORTED_EXOTICS = ['None', 'Asian', 'Barrier']

    def __init__(self, market_data: MarketData, v0: float, rho: float, kappa: float, theta: float, sigma_v: float, steps: int = 100, paths: int = 10000):
        super().__init__(market_data)
        self.v0 = v0
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.steps = steps
        self.paths = paths

    def _generate_paths(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        S0 = self.market_data.spot_price
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        T = self.market_data.time_to_expiry
        
        dt = T / self.steps       

        Z1 = np.random.standard_normal(size=(self.paths, self.steps))
        Z2 = np.random.standard_normal(size=(self.paths, self.steps))

        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        S = np.zeros((self.paths, self.steps + 1))
        v = np.zeros((self.paths, self.steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        for t in range(self.steps):
            v_pos = np.maximum(v[:, t], 0)
            S[:, t+1] = S[:, t] * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * W1[:, t])
            v[:, t+1] = v[:, t] + (self.kappa * (self.theta - v_pos) * dt) + (self.sigma_v * np.sqrt(v_pos * dt) * W2[:, t])

        return S

    def calculate_price(self, option: Option, seed: int = None, return_se: bool = False):
        paths = self._generate_paths(seed)
        
        # Object-Oriented magic: The Option handles its own payoff (Asian, Barrier, etc.)
        simulated_payoffs = option.get_payoff(paths)
        
        r = self.market_data.risk_free_rate
        T = self.market_data.time_to_expiry
        discount_factor = np.exp(-r * T)
        
        discounted_payoffs = discount_factor * simulated_payoffs
        price = np.mean(discounted_payoffs)
        se = np.std(discounted_payoffs) / np.sqrt(self.paths)
        
        return (price, se) if return_se else price

    def calculate_greeks(self, option: Option) -> dict:
        """Finite difference Greeks using fixed seeds to eliminate Monte Carlo noise."""
        FIXED_SEED = 42 
        dS = self.market_data.spot_price * 0.01
        dVol = 0.01
        dT = 1 / 365
        dR = 0.0001

        base_price = self.calculate_price(option, seed=FIXED_SEED)

        def get_price(bumped_md: MarketData, v0_bump: float = 0.0):
            temp_engine = HestonMonteCarloEngine(bumped_md, self.v0 + v0_bump, self.rho, self.kappa, self.theta, self.sigma_v, self.steps, self.paths)
            return temp_engine.calculate_price(option, seed=FIXED_SEED)

        md = self.market_data
        md_up_s = copy.copy(md); md_up_s.spot_price += dS
        md_dn_s = copy.copy(md); md_dn_s.spot_price -= dS
        md_time_pass = copy.copy(md); md_time_pass.time_to_expiry -= dT
        md_up_r = copy.copy(md); md_up_r.risk_free_rate += dR
        md_dn_r = copy.copy(md); md_dn_r.risk_free_rate -= dR

        delta = (get_price(md_up_s) - get_price(md_dn_s)) / (2 * dS)
        gamma = (get_price(md_up_s) - 2 * base_price + get_price(md_dn_s)) / (dS ** 2)
        vega = (get_price(md, dVol) - get_price(md, -dVol)) / (2 * dVol)
        theta = (get_price(md_time_pass) - base_price) / dT
        rho = (get_price(md_up_r) - get_price(md_dn_r)) / (2 * dR)

        return {'delta': delta, 'gamma': gamma, 'vega': vega / 100, 'theta': theta / 365, 'rho': rho / 100}