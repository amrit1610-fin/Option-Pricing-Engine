import numpy as np

class HestonMonteCarloModel:
    """
    Monte Carlo simulation of the Heston Stochastic Volatility Model.
    Uses the Full Truncation scheme (Lord et al., 2008) to handle negative variances.
    """
    def __init__(self, s0: float, v0: float, r: float, q: float, T: float, 
                 sigma: float, rho: float, kappa: float, theta: float):
        self.s0 = s0
        self.v0 = v0
        self.r = r
        self.q = q  
        self.T = T
        self.sigma = sigma
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        
        # Check Feller Condition
        feller_val = 2 * self.kappa * self.theta
        if feller_val <= self.sigma**2:
            print(f"Warning: Feller condition violated ({feller_val:.4f} <= {self.sigma**2:.4f}). Variance may hit zero.")

    def option_price(self, K: float, option_type: str = 'call', steps: int = 100, paths: int = 10000):
        dt = self.T / steps       

        # 1. Generating standard normal RVs
        Z1 = np.random.standard_normal(size=(paths, steps))
        Z2 = np.random.standard_normal(size=(paths, steps))

        # 2. Correlating the RVs (Cholesky Decomposition)
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        # 3. Monte-Carlo path simulation 
        S = np.zeros((paths, steps + 1))
        v = np.zeros((paths, steps + 1))

        S[:, 0] = self.s0
        v[:, 0] = self.v0

        for t in range(steps):
            # Full Truncation: Keep positive values for drift/diffusion terms
            v_pos = np.maximum(v[:, t], 0)

            # Stock SDE with dividend yield (q)
            S[:, t+1] = S[:, t] * np.exp((self.r - self.q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * W1[:, t])

            # Variance SDE (CIR process)
            v[:, t+1] = v[:, t] + (self.kappa * (self.theta - v_pos) * dt) + (self.sigma * np.sqrt(v_pos * dt) * W2[:, t])

        # 4. Payoff calculation
        terminal_prices = S[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - terminal_prices, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        # 5. Discounting and Error Calculation
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(paths)
        
        return price, standard_error


class HestonFourierModel:
    """
    Fourier transform pricing for Heston Model using the Carr-Madan (1999) FFT method.
    Uses the "Little Trap" characteristic function formulation to avoid branch cut issues.
    """
    def __init__(self, s0: float, v0: float, r: float, q: float, T: float, 
                 sigma: float, rho: float, kappa: float, theta: float):
        self.s0 = s0
        self.v0 = v0
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        
        self.x0 = np.log(self.s0)
        self.i = 1j

    def _cf(self, u):
        """Heston Characteristic Function (Albrecher formulation)"""
        a = self.kappa * self.theta
        b = self.kappa - (self.rho * self.sigma * self.i * u)
        d = np.sqrt(b**2 + self.sigma**2 * (self.i * u + u**2))
        g = (b - d) / (b + d)

        eDT = np.exp(-d * self.T)
        one_minus_g_eDT = 1 - g * eDT
        one_minus_g     = 1 - g
        
        # Small guards to prevent division by zero
        one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
        one_minus_g     = np.where(np.abs(one_minus_g)     < 1e-15, 1e-15, one_minus_g)

        C = self.i * u * (self.r - self.q) * self.T + (a / (self.sigma**2)) * ((b - d) * self.T - 2.0 * np.log(one_minus_g_eDT / one_minus_g))
        D = ((b - d) / (self.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)
        
        return np.exp(C + D * self.v0 + self.i * u * self.x0)

    def _simpson_weights(self, N: int):
        """Simpson weights on an N-point uniform grid."""
        w = np.ones(N)
        w[1:N-1:2] = 4
        w[2:N-2:2] = 2
        return w

    def fft_calls(self, N: int = 4096, eta: float = 0.25, alpha: float = 1.5):
        """Computes call prices over a grid of strikes using FFT."""
        n = np.arange(N)
        v = eta * n

        u = v - (alpha + 1) * self.i
        ert = np.exp(-self.r * self.T)
        
        # Carr-Madan damped characteristic function
        psi = (ert * self._cf(u)) / (alpha**2 + alpha - v**2 + self.i * (2 * alpha + 1) * v)

        w = self._simpson_weights(N) * (eta / 3.0)

        # FFT coupling
        lam = 2.0 * np.pi / (N * eta)   # Log-strike step
        b   = 0.5 * N * lam             # Half-width in k
        x   = psi * np.exp(self.i * b * v) * w

        F = np.fft.fft(x)
        F = np.real(F) 

        j = np.arange(N)
        k = -b + j * lam                # k = ln(K)
        K = np.exp(k)

        calls = np.exp(-alpha * k) / np.pi * F
        order = np.argsort(K)
        return K[order], np.maximum(calls[order], 0.0)

    def option_price(self, K: float, option_type: str = 'call', N: int = 4096, eta: float = 0.25, alpha: float = 1.5):
        """Prices a specific option by interpolating the FFT grid."""
        K_grid, C_grid = self.fft_calls(N=N, eta=eta, alpha=alpha)
        
        # Linear interpolation for Call
        if K <= K_grid[0]:
            call_price = C_grid[0]
        elif K >= K_grid[-1]:
            call_price = C_grid[-1]
        else:
            idx = np.searchsorted(K_grid, K)
            x0, x1 = K_grid[idx-1], K_grid[idx]
            y0, y1 = C_grid[idx-1], C_grid[idx]
            call_price = y0 + (y1 - y0) * (K - x0) / (x1 - x0)
            
        if option_type.lower() == 'call':
            return call_price
        elif option_type.lower() == 'put':
            # Put-Call Parity
            return call_price - self.s0 * np.exp(-self.q * self.T) + K * np.exp(-self.r * self.T)
        else:
            raise ValueError("Option type must be 'call' or 'put'")


#s0, v0, r, q, T = 100, 0.04, 0.07, 0.0, 1.0
#sigma, rho, kappa, theta = 0.3, -0.7, 2.0, 0.04
#K = 90
    
#print("========== Heston Fourier (Analytical) ==========")
#ft_model = HestonFourierModel(s0, v0, r, q, T, sigma, rho, kappa, theta)
#ft_call = ft_model.option_price(K, 'call')
#ft_put = ft_model.option_price(K, 'put')
#print(f"Call: {ft_call:.4f} | Put: {ft_put:.4f}")
#print("-" * 50)
    
#print("========== Heston Monte Carlo (Simulated) ==========")
#mc_model = HestonMonteCarloModel(s0, v0, r, q, T, sigma, rho, kappa, theta)
#mc_call, mc_call_se = mc_model.option_price(K, 'call', steps=100, paths=20000)
#mc_put, mc_put_se = mc_model.option_price(K, 'put', steps=100, paths=20000)
#print(f"Call: {mc_call:.4f} (SE: {mc_call_se:.4f})")
#print(f"Put:  {mc_put:.4f} (SE: {mc_put_se:.4f})")