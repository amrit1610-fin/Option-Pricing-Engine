import time
import numpy as np

from core.market_data import MarketData
from models.black_scholes import BlackScholesMerton
from models.monte_carlo import MonteCarloSimulations
from models.binomial_tree import BinomialOptionPricing
from models.heston_stoch_vol import HestonMonteCarloModel, HestonFourierModel

# Handy function to print header for out table
def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def main():
    print_header("QUANTITATIVE OPTION PRICING ENGINE")
    
    # Initialization of Market Data
    market_data = MarketData(
        spot_price=105.0,
        risk_free_rate=0.07,
        time_to_expiry=2.0,
        dividend_yield=0.05,
        volatility=0.2,
        strike_price=110.0
    )
    
    print("\n--- Market Parameters ---")
    print(f"Spot (S0):      ${market_data.spot_price:.2f}")
    print(f"Strike (K):     ${market_data.strike_price:.2f}")
    print(f"Time (T):       {market_data.time_to_expiry:.2f} Years")
    print(f"Risk-Free (r):  {market_data.risk_free_rate * 100:.1f}%")
    print(f"Dividend (q):   {market_data.dividend_yield * 100:.1f}%")
    print(f"Volatility (σ): {market_data.volatility * 100:.1f}%")
    
    # Extracting params for easier passing
    S = market_data.spot_price
    K = market_data.strike_price
    r = market_data.risk_free_rate
    T = market_data.time_to_expiry
    q = market_data.dividend_yield
    sigma = market_data.volatility
    
    # Heston Specific Parameters
    v0 = sigma ** 2             # Initial variance matches BS variance
    theta = 0.04                # Long-run average variance
    kappa = 2.0                 # Mean reversion rate
    rho = -0.7                  # Correlation (Spot/Vol)
    sigma_v = 0.3               # Volatility of volatility
    
    print("\n--- Heston Parameters ---")
    print(f"Initial Var (v0): {v0:.4f}")
    print(f"Long-Run Var (θ): {theta:.4f}")
    print(f"Mean Rev (κ):     {kappa:.1f}")
    print(f"Correlation (ρ):  {rho:.1f}")
    print(f"Vol of Vol (σ_v): {sigma_v:.2f}")

    print_header("PRICING RESULTS")
    print(f"{'Model':<25} | {'Call Price':<15} | {'Put Price':<15} | {'Compute Time':<15}")
    print("-" * 80)

    results = []

    # 1. Black-Scholes Model
    # ---------------------------------------------------------
    start_time = time.perf_counter()
    bs_model = BlackScholesMerton(S=S, K=K, r=r, q=q, T=T, sigma=sigma)
    bs_call = bs_model.option_price(option_type='call')
    bs_put = bs_model.option_price(option_type='put')
    bs_time = time.perf_counter() - start_time
    results.append(("Black-Scholes (Analytical)", bs_call, bs_put, bs_time))

    # 2. Binomial Tree Model
    # ---------------------------------------------------------
    tree_steps = 500
    start_time = time.perf_counter()
    # Note: Using s0=S to match your binomial_tree.py init arguments
    tree_model = BinomialOptionPricing(s0=S, K=K, r=r, q=q, T=T, sigma=sigma, N=tree_steps)
    
    # Unpack the tuple and extract the price at node [0, 0]
    _, call_tree = tree_model.option_prices(option_type='call')
    tree_call = call_tree[0, 0]
    
    _, put_tree = tree_model.option_prices(option_type='put')
    tree_put = put_tree[0, 0]
    
    tree_time = time.perf_counter() - start_time
    results.append((f"Binomial Tree (N={tree_steps})", tree_call, tree_put, tree_time))

    # 3. Monte Carlo Simulation (Standard)
    # ---------------------------------------------------------
    mc_paths = 50000
    mc_steps = 252 * int(T) # Daily steps
    start_time = time.perf_counter()
    mc_model = MonteCarloSimulations(s0=S, K=K, r=r, q=q, T=T, sigma=sigma, num_paths=mc_paths, time_steps=mc_steps)
    mc_call, mc_call_se = mc_model.option_price(option_type='call')
    mc_put, mc_put_se = mc_model.option_price(option_type='put')
    mc_time = time.perf_counter() - start_time
    results.append((f"Monte Carlo ({mc_paths} paths)", mc_call, mc_put, mc_time))

    # 4. Heston Model (Fourier Transform)
    # ---------------------------------------------------------
    start_time = time.perf_counter()
    heston_ft = HestonFourierModel(s0=S, v0=v0, r=r, q=q, T=T, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta)
    h_ft_call = heston_ft.option_price(K=K, option_type='call')
    h_ft_put = heston_ft.option_price(K=K, option_type='put')
    h_ft_time = time.perf_counter() - start_time
    results.append(("Heston (Fourier FFT)", h_ft_call, h_ft_put, h_ft_time))

    # 5. Heston Model (Monte Carlo)
    # ---------------------------------------------------------
    start_time = time.perf_counter()
    heston_mc = HestonMonteCarloModel(s0=S, v0=v0, r=r, q=q, T=T, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta)
    h_mc_call, h_mc_call_se = heston_mc.option_price(K=K, option_type='call', steps=100, paths=20000)
    h_mc_put, h_mc_put_se = heston_mc.option_price(K=K, option_type='put', steps=100, paths=20000)
    h_mc_time = time.perf_counter() - start_time
    results.append(("Heston (Monte Carlo)", h_mc_call, h_mc_put, h_mc_time))

    # Print Results Table
    for name, call, put, t in results:
        print(f"{name:<25} | ${call:<14.4f} | ${put:<14.4f} | {t*1000:<8.2f} ms")
        
    print("-" * 80)
    
    # Print Standard Errors for Monte Carlo Models
    print("\nMonte Carlo Standard Errors (95% Confidence):")
    print(f"  Standard MC Call: ±{1.96 * mc_call_se:.4f}")
    print(f"  Standard MC Put:  ±{1.96 * mc_put_se:.4f}")
    print(f"  Heston MC Call:   ±{1.96 * h_mc_call_se:.4f}")
    print(f"  Heston MC Put:    ±{1.96 * h_mc_put_se:.4f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()