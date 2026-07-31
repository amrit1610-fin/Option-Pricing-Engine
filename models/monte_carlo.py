import numpy as np

class MonteCarloSimulations:

    def __init__(self, 
                 s0: float, K: float,
                 r: float, T: float, sigma: float, 
                 num_paths: int, time_steps: int,
                 q: float = 0.0
        ):
        self.s0 = s0
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.num_paths = num_paths
        self.time_steps = time_steps

    def _generate_paths(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        dt = self.T / self.time_steps
        # Antithetic Variates
        Z = np.random.standard_normal((int(self.num_paths / 2), self.time_steps))
        Z = np.concatenate((Z, -Z), axis=0)
        
        paths = np.zeros((self.num_paths, self.time_steps + 1))
        paths[:, 0] = self.s0
        
        # Include dividend yield 'q' in drift
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        for t in range(1, self.time_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
            
        return paths

    def option_price(self, option_type: str = 'call', seed: int = None):
        """Standard Monte Carlo pricing for a European Option."""
        paths = self._generate_paths(seed=seed)
        terminal_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - self.K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(self.K - terminal_prices, 0)
        else:
            raise ValueError("Option type can only be call or put!")
            
        discounted_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        standard_error = np.std(payoffs * np.exp(-self.r * self.T)) / np.sqrt(self.num_paths)
        
        return discounted_price, standard_error

    def price_with_control_variate(self, option_type: str = 'call', seed: int = None):
        """Uses the terminal stock price S_T as a control variate to reduce variance."""
        paths = self._generate_paths(seed=seed)
        terminal_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            simulated_payoffs = np.maximum(terminal_prices - self.K, 0)
        else:
            simulated_payoffs = np.maximum(self.K - terminal_prices, 0)
    
        # Expected terminal price (S_t) accounting for dividend yield
        expected_terminal_price = self.s0 * np.exp((self.r - self.q) * self.T)

        # Control Variate process        
        covariance = np.cov(simulated_payoffs, terminal_prices)[0, 1]
        variance = np.var(terminal_prices)
        c = covariance / variance if variance > 0 else 0
        
        adjusted_payoffs = simulated_payoffs - c * (terminal_prices - expected_terminal_price)
        
        discounted_price = np.exp(-self.r * self.T) * np.mean(adjusted_payoffs)
        standard_error = np.std(adjusted_payoffs * np.exp(-self.r * self.T)) / np.sqrt(self.num_paths)

        return discounted_price, standard_error

    def calculate_greeks(self, option_type: str = 'call'):
        """Calculates Greeks using Finite Differences with a fixed random seed."""
        fixed_seed = 42
        
        dS = self.s0 * 0.01
        dVol = 0.01
        dT = 1 / 365
        dR = 0.0001

        def get_price(s, v, t, r_rate):
            model = MonteCarloSimulations(s0=s, K=self.K, r=r_rate, q=self.q, T=t, sigma=v, 
                                          num_paths=self.num_paths, time_steps=self.time_steps)
            price, _ = model.price_with_control_variate(option_type=option_type, seed=fixed_seed)
            return price

        base_price = get_price(self.s0, self.sigma, self.T, self.r)
        
        # Delta & Gamma
        price_up = get_price(self.s0 + dS, self.sigma, self.T, self.r)
        price_dn = get_price(self.s0 - dS, self.sigma, self.T, self.r)
        delta = (price_up - price_dn) / (2 * dS)
        gamma = (price_up - 2 * base_price + price_dn) / (dS ** 2)

        # Vega
        price_vol_up = get_price(self.s0, self.sigma + dVol, self.T, self.r)
        price_vol_dn = get_price(self.s0, self.sigma - dVol, self.T, self.r)
        vega = ((price_vol_up - price_vol_dn) / (2 * dVol)) / 100

        # Theta 
        price_time_pass = get_price(self.s0, self.sigma, self.T - dT, self.r)
        theta = ((price_time_pass - base_price) / dT) / 365.0

        # Rho
        price_r_up = get_price(self.s0, self.sigma, self.T, self.r + dR)
        price_r_dn = get_price(self.s0, self.sigma, self.T, self.r - dR)
        rho = ((price_r_up - price_r_dn) / (2 * dR)) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

#mc = MonteCarloSimulations(s0 = 100, K = 106, r=0.07, q=0.04 T=2.0, sigma=0.2, num_paths=50000, time_steps=252) # 252 trading days
#price, se = mc.option_price(option_type='call')
#print(f"Standard MC Price: {price:.4f} (Standard Error: {se:.4f})")

#cv_price, cv_se = mc.price_with_control_variate(option_type='call')
#print(f"Control variate MC Price: {cv_price:.4f} (Standard Error: {cv_se:.4f})")