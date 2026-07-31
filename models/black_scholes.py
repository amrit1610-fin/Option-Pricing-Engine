import numpy as np
from scipy.stats import norm

class BlackScholesMerton:

    def __init__(self, 
                S:float = None, K:float = None, 
                r:float = None, q:float = None, 
                T:float = None, sigma:float = None
        ):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma

        # defining d1 and d2 for Black-scholes-merton model
        if self.T > 0:   # T approaches 0
            self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
            self.d2 = self.d1 - (self.sigma * np.sqrt(self.T))
        else:            # when the option expires
            self.d1 = float('inf') if self.S > self.K else float('-inf')
            self.d2 = float('inf') if self.S > self.K else float('-inf')

    # defining the closed-form option price
    def option_price(self, option_type):
        if option_type == 'call':
            if self.T <= 0:
                return max(self.S - self.K, 0)
            return (self.S * np.exp(-self.q*self.T) * norm.cdf(self.d1)) - (self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2))
        elif option_type == 'put':
            if self.T <= 0:
                return max (self.K - self.S, 0)
            return (self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2)) - (self.S * np.exp(-self.q*self.T) * norm.cdf(-self.d1))
        else:
            raise ValueError("!Input correct opton type!")

    # developing greeks by partially differentiating the Black-scholes PDE
    def calculate_greeks(self, option_type):

        # global term to be used by both put and call for theta calculation
        theta_term1 = -(self.S * norm.pdf(self.d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))

        # calculating delta, theta, rho for call option
        if option_type == 'call':
            # 1. Delta
            delta = np.exp(-self.q*self.T) * norm.cdf(self.d1)

            # 2. Theta
            theta_term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
            theta_term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            annual_theta = theta_term1 + theta_term2 - theta_term3
            theta = annual_theta / 365.0     # get per day theta

            # 3. Rho
            annual_rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            rho = annual_rho / 100.0         # get percentage change

        # calculating delta, theta, rho for put option
        elif option_type == 'put':
            # 1. Delta
            delta = np.exp(-self.q*self.T) * (norm.cdf(self.d1) - 1)

            # 2. Theta
            theta_term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
            theta_term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            annual_theta = theta_term1 - theta_term2 + theta_term3
            theta = annual_theta / 365.0      # get per day theta

            # 3. Rho
            annual_rho = -(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
            rho = annual_rho / 100.0    # get percentage change

        # Gamma and Vega are independent of option type (call/ put)
        # 4. Gamma
        g_numerator = np.exp(-self.q * self.T) * norm.pdf(self.d1)
        g_denominator = self.S * self.sigma * np.sqrt(self.T)
        gamma = g_numerator / g_denominator

        # 5. Vega
        annual_vega = self.S * np.sqrt(self.T) * norm.pdf(self.d1) * np.exp(-self.q * self.T)
        vega = annual_vega / 100.0     # get percentage change

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

#model = BlackScholesMerton(S=100.0, K=105.0, r=0.07, q=0.04, T=2.0, sigma=0.2)
#price = model.option_price(option_type = 'call')
#delta, theta, rho, gamma, vega = model.calculate_greeks(option_type='call')

#print(price)
#print(delta, theta, rho, gamma, vega)