class BlackScholesMerton:

    def __init__(self, S, K, r, q, T, sigma):
        import numpy as np
        self.np = np

        from scipy.stats import norm
        self.norm = norm

        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma

        # defining d1 and d2 for Black-scholes-merton model
        if self.T > 0:   # T approaches 0
            self.d1 = (self.np.log(self.S / self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
            self.d2 = self.d1 - (self.sigma * self.np.sqrt(self.T))
        else:        # when the option expires
            self.d1 = float('inf') if self.S > self.K else float('-inf')
            self.d2 = float('inf') if self.S > self.K else float('-inf')

    def option_price(self, option_type):
        if option_type == 'call':
            if self.T <= 0:
                return max(self.S - self.K, 0)
            return (self.S * self.np.exp(-self.q*self.T) * self.norm.cdf(self.d1)) - (self.K * self.np.exp(-self.r*self.T) * self.norm.cdf(self.d2))
        elif option_type == 'put':
            if self.T <= 0:
                return max (self.K - self.S, 0)
            return (self.K * self.np.exp(-self.r*self.T) * self.norm.cdf(-self.d2)) - (self.S * self.np.exp(-self.q*self.T) * self.norm.cdf(-self.d1))
        else:
            raise ValueError("!Input correct opton type!")

    def greeks(self, option_type):

        theta_term1 = -(self.S * self.norm.pdf(self.d1) * self.sigma * self.np.exp(-self.q * self.T)) / (2 * self.np.sqrt(self.T))

        if option_type == 'call':
            # 1. Delta
            delta = self.np.exp(-self.q*self.T) * self.norm.cdf(self.d1)

            # 2. Theta
            theta_term2 = self.q * self.S * self.np.exp(-self.q * self.T) * self.norm.cdf(self.d1)
            theta_term3 = self.r * self.K * self.np.exp(-self.r * self.T) * self.norm.cdf(self.d2)
            annual_theta = theta_term1 + theta_term2 - theta_term3
            theta = annual_theta / 365.0 

            # 3. Rho
            annual_rho = self.K * self.T * self.np.exp(-self.r * self.T) * self.norm.cdf(self.d2)
            rho = annual_rho / 100.0

        elif option_type == 'put':
            # 1. Delta
            delta = self.np.exp(-self.q*self.T) * (self.norm.cdf(self.d1) - 1)

            # 2. Theta
            theta_term2 = self.q * self.S * self.np.exp(-self.q * self.T) * self.norm.cdf(-self.d1)
            theta_term3 = self.r * self.K * self.np.exp(-self.r * self.T) * self.norm.cdf(-self.d2)
            annual_theta = theta_term1 - theta_term2 + theta_term3
            theta = annual_theta / 365.0 

            # 3. Rho
            annual_rho = -(self.K * self.T * self.np.exp(-self.r * self.T) * self.norm.cdf(-self.d2))
            rho = annual_rho / 100.0

        # 4. Gamma
        g_numerator = self.np.exp(-self.q * self.T) * self.norm.pdf(self.d1)
        g_denominator = self.S * self.sigma * self.np.sqrt(self.T)
        gamma = g_numerator / g_denominator

        # 5. Vega
        annual_vega = self.S * self.np.sqrt(self.T) * self.norm.pdf(self.d1) * self.np.exp(-self.q * self.T)
        vega = annual_vega / 100.0 

        return delta, theta, rho, gamma, vega

# model = BlackScholesMerton(100.0, 105.0, 0.07, 0.04, 2.0, 0.2)
# price = model.option_price(option_type = 'call')
# delta, theta, rho, gamma, vega = model.greeks(option_type='call')

# print(price)
# print(delta, theta, rho, gamma, vega)