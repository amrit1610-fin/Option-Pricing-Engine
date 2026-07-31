import numpy as np
import pandas as pd
import time

from core.market_data import MarketData
from models.binomial_tree import BinomialOptionPricing
from models.black_scholes import BlackScholesMerton
from models.monte_carlo import MonteCarloSimulations
from models.heston_stoch_vol import HestonMonteCarloModel, HestonFourierModel

def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def main():
    print_header("QUANTITATIVE OPTION PRICING ENGINE")
    
    # 1. Initializing Market Data
    market_data = MarketData(
        spot_price=105.0,
        risk_free_rate=0.07,
        time_to_expiry=2.0,
        dividend_yield=0.05,
        volatility=0.2,
        strike_price=110.0
    )
    
    S = market_data.spot_price
    K = market_data.strike_price
    r = market_data.risk_free_rate
    T = market_data.time_to_expiry
    q = market_data.dividend_yield
    sigma = market_data.volatility
    
    # Heston specific parameters
    v0 = 0.04                # Initial variance (e.g., 0.20^2)
    theta = 0.04             # Long-run variance
    kappa = 2.0              # Mean reversion rate
    rho = -0.7               # Correlation between asset and volatility
    sigma_v = 0.3            # Volatility of volatility
    
    print("\n--- Market Parameters ---")
    print(f"Spot (S0):      ${S:.2f}")
    print(f"Strike (K):     ${K:.2f}")
    print(f"Time (T):       {T:.2f} Years")
    print(f"Risk-Free (r):  {r * 100:.1f}%")
    print(f"Dividend (q):   {q * 100:.1f}%")
    print(f"Volatility (σ): {sigma * 100:.1f}%")

    print("\n--- Heston Parameters ---")
    print(f"Initial Var (v0): {v0:.4f}")
    print(f"Long-Run Var (θ): {theta:.4f}")
    print(f"Mean Rev (κ):     {kappa}")
    print(f"Correlation (ρ):  {rho}")
    print(f"Vol of Vol (σ_v): {sigma_v:.2f}")

    # =========================================================
    # 2. PRICING MODELS EXECUTION
    # =========================================================
    pricing_results = []

    # A. Black-Scholes Model
    start_time = time.perf_counter()
    bs_model = BlackScholesMerton(S=S, K=K, r=r, q=q, T=T, sigma=sigma)
    bs_call = bs_model.option_price(option_type='call')
    bs_put = bs_model.option_price(option_type='put')
    bs_time = (time.perf_counter() - start_time) * 1000
    pricing_results.append({"Model": "Black-Scholes (Analytical)", "Call Price": bs_call, "Put Price": bs_put, "Compute Time (ms)": bs_time})

    # B. Binomial Tree Model
    tree_steps = 500
    start_time = time.perf_counter()
    tree_model = BinomialOptionPricing(s0=S, K=K, r=r, q=q, T=T, sigma=sigma, N=tree_steps)
    _, call_tree = tree_model.option_prices(option_type='call')
    tree_call = call_tree[0, 0]
    _, put_tree = tree_model.option_prices(option_type='put')
    tree_put = put_tree[0, 0]
    tree_time = (time.perf_counter() - start_time) * 1000
    pricing_results.append({"Model": f"Binomial Tree (N={tree_steps})", "Call Price": tree_call, "Put Price": tree_put, "Compute Time (ms)": tree_time})

    # C. Monte Carlo Simulation (Control Variate)
    mc_paths = 50000
    mc_steps = 252 * int(T)
    start_time = time.perf_counter()
    mc_model = MonteCarloSimulations(s0=S, K=K, r=r, q=q, T=T, sigma=sigma, num_paths=mc_paths, time_steps=mc_steps)
    mc_call, mc_call_se = mc_model.price_with_control_variate(option_type='call')
    mc_put, mc_put_se = mc_model.price_with_control_variate(option_type='put')
    mc_time = (time.perf_counter() - start_time) * 1000
    pricing_results.append({"Model": f"Monte Carlo ({mc_paths} paths)", "Call Price": mc_call, "Put Price": mc_put, "Compute Time (ms)": mc_time})

    # D. Heston Model (Fourier Transform)
    start_time = time.perf_counter()
    heston_ft = HestonFourierModel(s0=S, v0=v0, r=r, q=q, T=T, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta)
    h_ft_call = heston_ft.option_price(K=K, option_type='call')
    h_ft_put = heston_ft.option_price(K=K, option_type='put')
    h_ft_time = (time.perf_counter() - start_time) * 1000
    pricing_results.append({"Model": "Heston (Fourier FFT)", "Call Price": h_ft_call, "Put Price": h_ft_put, "Compute Time (ms)": h_ft_time})

    # E. Heston Model (Monte Carlo)
    start_time = time.perf_counter()
    heston_mc = HestonMonteCarloModel(s0=S, v0=v0, r=r, q=q, T=T, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta)
    h_mc_call, h_mc_call_se = heston_mc.option_price(K=K, option_type='call', steps=100, paths=20000)
    h_mc_put, h_mc_put_se = heston_mc.option_price(K=K, option_type='put', steps=100, paths=20000)
    h_mc_time = (time.perf_counter() - start_time) * 1000
    pricing_results.append({"Model": "Heston (Monte Carlo)", "Call Price": h_mc_call, "Put Price": h_mc_put, "Compute Time (ms)": h_mc_time})

    # Displaying Pricing DataFrame
    print_header("PRICING RESULTS")
    df_pricing = pd.DataFrame(pricing_results)
    print(df_pricing.to_string(index=False, formatters={
        'Call Price': '${:>8.4f}'.format,
        'Put Price': '${:>8.4f}'.format,
        'Compute Time (ms)': '{:>8.2f} ms'.format
    }))
    
    print("\nMonte Carlo Standard Errors (95% Confidence):")
    print(f"  Standard MC Call: ±{mc_call_se * 1.96:.4f}")
    print(f"  Standard MC Put:  ±{mc_put_se * 1.96:.4f}")
    print(f"  Heston MC Call:   ±{h_mc_call_se * 1.96:.4f}")
    print(f"  Heston MC Put:    ±{h_mc_put_se * 1.96:.4f}")

    # =========================================================
    # 3. GREEKS EXECUTION
    # =========================================================
    print_header("NUMERICAL GREEKS COMPARISON (CALL OPTION)")
    
    # Calculating Greeks for all models
    bs_greeks = bs_model.calculate_greeks('call')
    tree_greeks = tree_model.calculate_greeks('call')
    mc_greeks = mc_model.calculate_greeks('call')
    heston_greeks = heston_ft.calculate_greeks(K, 'call')

    # Building Greeks DataFrame
    greeks_data = []
    greeks_keys = ['delta', 'gamma', 'vega', 'theta', 'rho']
    
    for greek in greeks_keys:
        greeks_data.append({
            "Greek": greek.capitalize(),
            "Black-Scholes": bs_greeks[greek],
            "Binomial Tree": tree_greeks[greek],
            "Monte Carlo": mc_greeks[greek],
            "Heston (Fourier)": heston_greeks[greek]
        })

    df_greeks = pd.DataFrame(greeks_data)
    
    # Display Greeks DataFrame
    print(df_greeks.to_string(index=False, formatters={
        'Black-Scholes': '{:>10.4f}'.format,
        'Binomial Tree': '{:>10.4f}'.format,
        'Monte Carlo': '{:>10.4f}'.format,
        'Heston (Fourier)': '{:>10.4f}'.format
    }))
    print("=" * 80 + "\n")

    # Save results to CSV 
    # df_pricing.to_csv("pricing_results.csv", index=False)
    # df_greeks.to_csv("greeks_results.csv", index=False)

if __name__ == "__main__":
    main()