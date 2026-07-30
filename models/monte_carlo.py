import numpy as np
from black_scholes import BlackScholesMerton 

class MonteCarloSimulations:

    def __init__(self, num_paths: int = 100000, time_steps: int = 100):
        self.num_paths = num_paths
        self.time_steps = time_steps

    def _generate_paths(self, spot: float, rate: float, 
                        vol: float, maturity: float) -> np.ndarray:
        """
        Generates Geometric Brownian Motion (GBM) paths.
        Uses Antithetic Variates for basic variance reduction.
        """
        dt = maturity / self.time_steps
        
        # We generate half the random normals, and use their negatives for the other half (Antithetic Variates)
        Z = np.random.standard_normal((int(self.num_paths / 2), self.time_steps))
        Z = np.concatenate((Z, -Z), axis=0)
        
        # Euler-Maruyama discretization for GBM
        paths = np.zeros((self.num_paths, self.time_steps + 1))
        paths[:, 0] = spot
        
        drift = (rate - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt)
        
        for t in range(1, self.time_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
            
        return paths

    def price_european(self, spot: float, strike: float, rate: float, vol: float, maturity: float, option_type: str = 'call') -> float:
        """
        Standard Monte Carlo pricing for a European Option.
        """
        paths = self._generate_paths(spot, rate, vol, maturity)
        terminal_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - terminal_prices, 0)
            
        discounted_price = np.exp(-rate * maturity) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(self.num_paths)
        
        return discounted_price, standard_error

    def price_with_control_variate(self, spot: float, strike: float, rate: float, vol: float, maturity: float, option_type: str = 'call'):
        """
        Advanced: Uses Black-Scholes as a control variate to reduce variance.
        This is why you import your Black-Scholes file!
        """
        # 1. Get the analytical price from your imported library
        # bs_model = BlackScholes(spot, strike, rate, vol, maturity, option_type)
        # analytical_bs_price = bs_model.price()
        
        # Placeholder for the analytical price
        analytical_bs_price = 10.50 # Replace with actual call to your BS code
        
        paths = self._generate_paths(spot, rate, vol, maturity)
        terminal_prices = paths[:, -1]
        
        # Simulate both the exotic payoff (if you had one) and the standard European payoff
        # Here we just use standard European for both to demonstrate the concept
        if option_type.lower() == 'call':
            simulated_payoffs = np.maximum(terminal_prices - strike, 0)
        else:
            simulated_payoffs = np.maximum(strike - terminal_prices, 0)
            
        simulated_bs_payoffs = simulated_payoffs # In a real scenario, this is the simple European payoff
        
        # Control Variate Formula:
        # Adjusted Price = Simulated Price + c * (Analytical BS - Simulated BS)
        # Optimal 'c' is the covariance(simulated, simulated_bs) / variance(simulated_bs)
        
        covariance = np.cov(simulated_payoffs, simulated_bs_payoffs)[0, 1]
        variance = np.var(simulated_bs_payoffs)
        c = covariance / variance if variance > 0 else 0
        
        adjusted_payoffs = simulated_payoffs + c * (analytical_bs_price - np.mean(simulated_bs_payoffs))
        
        discounted_price = np.exp(-rate * maturity) * np.mean(adjusted_payoffs)
        standard_error = np.std(adjusted_payoffs) / np.sqrt(self.num_paths)
        
        return discounted_price, standard_error


#mc = MonteCarloSimulations(num_paths=50000, time_steps=252) # 252 trading days
#price, se = mc.price_european(spot=100, strike=95, rate=0.07, vol=0.2, maturity=2.0, option_type='put')
#sprint(f"Standard MC Price: {price:.4f} (Standard Error: {se:.4f})")