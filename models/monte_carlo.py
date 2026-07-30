import numpy as np
# from black_scholes import BlackScholesMerton 

class MonteCarloSimulations:

    def __init__(self, 
                s0: float, K: float,
                r: float, T: float, sigma: float, 
                num_paths: int, time_steps: int
        ):
        self.s0 = s0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.num_paths = num_paths
        self.time_steps = time_steps

    def _generate_paths(self) -> np.ndarray:

        dt = self.T / self.time_steps
        # We generate half the random normals, and use their negatives for the other half (Antithetic Variates)
        Z = np.random.standard_normal((int(self.num_paths / 2), self.time_steps))
        Z = np.concatenate((Z, -Z), axis=0)
        
        # Euler-Maruyama discretization for GBM
        paths = np.zeros((self.num_paths, self.time_steps + 1))
        paths[:, 0] = self.s0
        
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        for t in range(1, self.time_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
            
        return paths

    def option_price(self, option_type: str = 'call'):
        """
        Standard Monte Carlo pricing for a European Option.
        """
        paths = self._generate_paths()
        terminal_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - self.K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(self.K - terminal_prices, 0)
        else:
            raise ValueError("!Option type can only be call or put!")
            
        discounted_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(self.num_paths)
        
        return discounted_price, standard_error

    def price_with_control_variate(self, option_type: str = 'call'):
        """
        Since here we're pricing vanilla European options, we are using the simple terminal stock price (S_t).
        Else if we need to price an exotic option, we need to import the Black-Scholes model from black_scholes.py, and then use it in control variate pricing. 
        """        
        paths = self._generate_paths()
        terminal_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            simulated_payoffs = np.maximum(terminal_prices - self.K, 0)
        else:
            simulated_payoffs = np.maximum(self.K - terminal_prices, 0)
    
        # Expected terminal price (S_t)
        expected_terminal_price = self.s0 * np.exp(self.r * self.T)

        # Control Variate process        
        covariance = np.cov(simulated_payoffs, terminal_prices)[0, 1]
        variance = np.var(terminal_prices)
        c = covariance / variance if variance > 0 else 0
        
        adjusted_payoffs = simulated_payoffs - c * (terminal_prices - expected_terminal_price)
        # Discount and calculate error
        discounted_price = np.exp(-self.r * self.T) * np.mean(adjusted_payoffs)
        standard_error = np.std(adjusted_payoffs * np.exp(-self.r * self.T)) / np.sqrt(self.num_paths)

        return discounted_price, standard_error


mc = MonteCarloSimulations(s0 = 100, K = 106, r=0.07, T=2.0, sigma=0.2, num_paths=50000, time_steps=252) # 252 trading days
price, se = mc.option_price(option_type='call')
print(f"Standard MC Price: {price:.4f} (Standard Error: {se:.4f})")

cv_price, cv_se = mc.price_with_control_variate(option_type='call')
print(f"Control variate MC Price: {cv_price:.4f} (Standard Error: {cv_se:.4f})")




