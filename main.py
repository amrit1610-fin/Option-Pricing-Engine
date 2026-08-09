import pandas as pd
import time

from core.market_data import MarketData
from core.instruments import EuropeanOption, AmericanOption, AsianOption, BarrierOption

from models.black_scholes import BlackScholesEngine
from models.binomial_tree import BinomialTreeEngine
from models.monte_carlo import MonteCarloEngine
from models.heston import HestonFourierEngine, HestonMonteCarloEngine

def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def main():
    print_header("QUANTITATIVE OPTION PRICING ENGINE (OOP ARCHITECTURE)")
    
    # 1. Initialize Global Market Data
    md = MarketData(
        spot_price=105.0,
        risk_free_rate=0.07,
        time_to_expiry=2.0,
        dividend_yield=0.05,
        volatility=0.2,
        strike_price=110.0 # Kept for backward compatibility
    )
    
    # Heston specific parameters
    heston_params = {
        'v0': 0.04, 'theta': 0.04, 'kappa': 2.0, 'rho': -0.7, 'sigma_v': 0.3
    }
    
    engines = [
        ("Black-Scholes (Analytical)", BlackScholesEngine(md)),
        ("Binomial Tree (500 steps)", BinomialTreeEngine(md, N=500)),
        ("Monte Carlo (50k paths)", MonteCarloEngine(md, num_paths=50000, time_steps=500)),
        ("Heston (Fourier FFT)", HestonFourierEngine(md, **heston_params)),
        ("Heston (Monte Carlo)", HestonMonteCarloEngine(md, steps=100, paths=20000, **heston_params))
    ]

    test_options = [
        ("Standard European Call", EuropeanOption(strike=110.0, option_type='call')),
        ("American Put (Early Exercise)", AmericanOption(strike=110.0, option_type='put')),
        ("Asian Arithmetic Call", AsianOption(strike=110.0, option_type='call', averaging_type='arithmetic')),
        ("Up-and-Out Barrier Call", BarrierOption(strike=110.0, option_type='call', barrier_level=130.0, barrier_type='up-and-out'))
    ]

    for test_name, option in test_options:
        print_header(f"TESTING: {test_name}")
        pricing_results = []
        
        for engine_name, engine in engines:
            # Dynamically check if engine can price this contract!
            if not engine.check_compatibility(option):
                print(f"[SKIP] {engine_name} does not support {option.style} {option.exotic_type} options.")
                continue
                
            start_time = time.perf_counter()
            
            # Price the option
            if 'Monte Carlo' in engine_name:
                price, se = engine.calculate_price(option, return_se=True)
                price_display = f"${price:>7.4f} (±{se*1.96:.4f})"
            else:
                price = engine.calculate_price(option)
                price_display = f"${price:>7.4f}"
                
            calc_time = (time.perf_counter() - start_time) * 1000
            
            # Calculate Greeks
            greeks = engine.calculate_greeks(option)
            
            pricing_results.append({
                "Engine": engine_name,
                "Price": price_display,
                "Compute (ms)": calc_time,
                "Delta": greeks['delta'],
                "Gamma": greeks['gamma'],
                "Vega": greeks['vega']
            })
            
        # Display the results using Pandas
        if pricing_results:
            df = pd.DataFrame(pricing_results)
            print("\n" + df.to_string(index=False, formatters={
                'Compute (ms)': '{:>8.2f} ms'.format,
                'Delta': '{:>7.4f}'.format,
                'Gamma': '{:>7.4f}'.format,
                'Vega': '{:>7.4f}'.format,
            }))

if __name__ == "__main__":
    main()