import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BinomialOptionPricing:

    def __init__(self, 
                 s0:float = None, K:float = None, 
                 r:float = None, T:float = None, 
                 sigma:float = None, N:int = None,
                 q:float = 0.0
        ):
        self.s0 = s0
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.N = N

        # derived formulas to be needed later
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)

    def option_prices(self, option_type='call'):

        # 1. Initialize trees with zeros
        stock_tree = np.zeros((self.N+1 , self.N+1))
        option_tree = np.zeros((self.N+1 , self.N+1))

        # 2. Building stock price tree (Forward pass)
        for i in range(self.N+1):
            for j in range(i+1):
                stock_tree[j, i] = self.s0 * (self.u ** (i - j)) * (self.d ** j)

        # 3. Calculating terminal values at expiry
        for j in range(self.N + 1):
            if option_type.lower() == 'call':
                option_tree[j, self.N] = max(0, stock_tree[j, self.N] - self.K)
            elif option_type.lower() == 'put':
                option_tree[j, self.N] = max(0, self.K - stock_tree[j, self.N])

        # 4. Building option tree by calculating Option prices (Backward Induction)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                expected_value = self.p * option_tree[j, i+1] + (1 - self.p) * option_tree[j+1, i+1]
                option_tree[j, i] = self.discount * expected_value

        return stock_tree, option_tree


    def plot_binomial_tree(self,stock_tree, option_tree):
        """
        Visualizes the calculated trees using NetworkX and Matplotlib.
        """
        G = nx.DiGraph()
        pos = {}
        labels = {}

        # Map the nodes, positions, and labels
        for i in range(self.N + 1):
            for j in range(i + 1):
                node_id = f"{i}_{j}"
                G.add_node(node_id)
                
                # X-axis is time step, Y-axis spreads out nodes based on up/down moves
                pos[node_id] = (i, i - 2 * j)
                
                # Label contains both the Stock Price (S) and Option Value (V)
                labels[node_id] = f"S: {stock_tree[j, i]:.2f}\nV: {option_tree[j, i]:.2f}"

                # Add directional edges connecting to the next time step
                if i < self.N:
                    G.add_edge(node_id, f"{i+1}_{j}")     # Up move path
                    G.add_edge(node_id, f"{i+1}_{j+1}")   # Down move path

        # Render the plot
        plt.figure(figsize=(12, 7))
        nx.draw(G, pos, labels=labels, with_labels=True, 
                node_size=2000, node_color="lightsteelblue",
                font_size=9, font_weight="bold", arrows=True,
                edge_color="gray")
        
        plt.title(f"{self.N}-Step Binomial Tree\nS = Stock Price, V = Option Value", fontsize=14)
        plt.margins(0.1)
        plt.show()

#r = 0.05      # risk-free rate
#T = 1         # Time to expiry
#N = 10        # time steps
#sigma = 0.3   # constant volatility
#s0 = 50       # initial stock price
#K = 52        # Strike price
#option_type = "Call"

#engine = BinomialOptionPricing(s0=s0, K=K, r=r, T=T, sigma=sigma, N=N)
#stock_tree, option_tree = engine.option_prices(option_type = option_type)
#print(f"Calculated European {option_type.capitalize()} Price: ${option_tree[0, 0]:.4f}")

#engine.plot_binomial_tree(stock_tree, option_tree)