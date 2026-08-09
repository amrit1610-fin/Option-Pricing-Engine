import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

from models.base import PricingEngine
from core.instruments import Option
from core.market_data import MarketData

class BinomialTreeEngine(PricingEngine):
    """
    Cox-Ross-Rubinstein Binomial Tree Model.
    Mathematically capable of pricing both European and American options.
    """
    
    # Engine Capabilities (UI Dynamic Metadata)
    SUPPORTED_STYLES = ['European', 'American']
    SUPPORTED_EXOTICS = ['None']

    def __init__(self, market_data: MarketData, N: int = 500):
        super().__init__(market_data)
        self.N = N

    def _build_trees(self, option: Option):
        """Core logic to build stock and option price trees."""
        S = self.market_data.spot_price
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        T = self.market_data.time_to_expiry
        sigma = self.market_data.volatility

        # Tree factors
        dt = T / self.N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        # Initialize empty tree grids
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        option_tree = np.zeros((self.N + 1, self.N + 1))

        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

        # We pass the entire terminal column to the Option object's payoff function
        terminal_prices = stock_tree[:, self.N]
        option_tree[:, self.N] = option.get_payoff(terminal_prices)

        # Check if option is American for early exercise logic
        is_american = option.style == 'American'

        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                # 1. Calculate the expected discounted value of holding the option
                expected_value = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                
                if is_american:
                    # 2. Calculate the immediate intrinsic value if exercised right now
                    current_spot = stock_tree[j, i]
                    # The get_payoff expects arrays, so we wrap the scalar in an array
                    intrinsic_value = option.get_payoff(np.array([current_spot]))[0]
                    
                    # 3. The true option value is the maximum of holding vs exercising early
                    option_tree[j, i] = max(expected_value, intrinsic_value)
                else:
                    # European options cannot be exercised early
                    option_tree[j, i] = expected_value

        return stock_tree, option_tree

    def calculate_price(self, option: Option) -> float:
        """Required by PricingEngine base class. Executes the pricing."""
        _, option_tree = self._build_trees(option)
        return option_tree[0, 0]

    def calculate_greeks(self, option: Option) -> dict:
        """Calculates Greeks using Finite Difference Method (Bump and Revalue)"""
        dS = self.market_data.spot_price * 0.01  # 1% bump in spot
        dVol = 0.01                              # 1% bump in vol
        dT = 1 / 365                             # 1 day bump in time
        dR = 0.0001                              # 1 basis point bump in rate

        # Helper to quickly re-price with bumped market data
        def get_price(bumped_data: MarketData) -> float:
            temp_engine = BinomialTreeEngine(bumped_data, self.N)
            return temp_engine.calculate_price(option)

        base_price = self.calculate_price(option)
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
            'vega': vega / 100,      # Scaled for a 1% change
            'theta': theta / 365,    # Scaled for 1-day time decay
            'rho': rho / 100         # Scaled for a 1% change
        }

    def plot_binomial_tree(self, option: Option):
        """Visualizes the calculated trees using NetworkX and Matplotlib."""
        stock_tree, option_tree = self._build_trees(option)
        
        G = nx.DiGraph()
        pos = {}
        labels = {}

        for i in range(self.N + 1):
            for j in range(i + 1):
                node_id = f"{i}_{j}"
                G.add_node(node_id)
                pos[node_id] = (i, i - 2 * j)
                labels[node_id] = f"S: {stock_tree[j, i]:.2f}\nV: {option_tree[j, i]:.2f}"

                if i < self.N:
                    G.add_edge(node_id, f"{i+1}_{j}")     
                    G.add_edge(node_id, f"{i+1}_{j+1}")   

        plt.figure(figsize=(12, 7))
        nx.draw(G, pos, labels=labels, with_labels=True, 
                node_size=2000, node_color="lightsteelblue",
                font_size=9, font_weight="bold", arrows=True,
                edge_color="gray")
        
        plt.title(f"{self.N}-Step Binomial Tree ({option.style} {option.option_type.capitalize()})\nS = Stock Price, V = Option Value", fontsize=14)
        plt.margins(0.1)
        plt.show()