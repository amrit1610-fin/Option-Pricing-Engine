import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Importing your custom pricing engine modules
from core.market_data import MarketData
from models.binomial_tree import BinomialOptionPricing
from models.black_scholes import BlackScholesMerton
from models.monte_carlo import MonteCarloSimulations
from models.heston_stoch_vol import HestonMonteCarloModel, HestonFourierModel

st.set_page_config(
    page_title="Quant Option Pricing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject beautiful styling
st.markdown("""
<style>
    .metric-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4da6ff;
        margin-bottom: 0rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
    st.sidebar.title("Model Parameters")
    
    st.sidebar.header("1. Asset & Option Details")
    option_type = st.sidebar.selectbox("Option Type", options=["Call", "Put"]).lower()
    spot = st.sidebar.number_input("Spot Price (S0)", value=105.0, step=1.0)
    strike = st.sidebar.number_input("Strike Price (K)", value=110.0, step=1.0)
    ttm = st.sidebar.slider("Time to Maturity (Years)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    
    st.sidebar.header("2. Market Rates")
    rate = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.20, value=0.07, step=0.01, format="%.2f")
    div_yield = st.sidebar.slider("Dividend Yield (q)", min_value=0.0, max_value=0.20, value=0.05, step=0.01, format="%.2f")
    vol = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
    
    st.sidebar.header("3. Heston Model Dynamics")
    v0 = st.sidebar.number_input("Initial Variance (v0)", value=0.04, step=0.01, format="%.4f")
    theta = st.sidebar.number_input("Long-Run Variance (θ)", value=0.04, step=0.01, format="%.4f")
    kappa = st.sidebar.slider("Mean Reversion (κ)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    rho = st.sidebar.slider("Correlation (ρ)", min_value=-0.99, max_value=0.99, value=-0.70, step=0.05)
    sigma_v = st.sidebar.slider("Vol of Vol (σ_v)", min_value=0.01, max_value=1.0, value=0.30, step=0.05)

    st.sidebar.header("4. Compute Settings")
    mc_paths = st.sidebar.selectbox("Monte Carlo Paths", options=[10000, 25000, 50000, 100000], index=1)
    tree_steps = st.sidebar.slider("Binomial Tree Steps", min_value=100, max_value=1000, value=500, step=100)
    
    st.markdown('<p class="main-header">Quantitative Option Pricing Engine</p>', unsafe_allow_html=True)
    st.markdown("A modular derivatives pricing library comparing analytical, tree-based, simulation, and stochastic volatility models.")
    st.divider()

    # Calculate Black-Scholes Baseline immediately for top metrics
    bs_model = BlackScholesMerton(S=spot, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol)
    bs_call = bs_model.option_price('call')
    bs_put = bs_model.option_price('put')

    # Display Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Black-Scholes Call", f"${bs_call:.4f}")
    col2.metric("Black-Scholes Put", f"${bs_put:.4f}")
    col3.metric("Moneyness (S/K)", f"{(spot/strike):.4f}")
    col4.metric("Time to Expiry", f"{ttm} Years")

    st.divider()

    if st.button("Calculate Prices & Greeks", type="primary", use_container_width=True):
        with st.spinner("Running numerical engines and simulations..."):
            
            pricing_results = []
            greeks_data = []
            
            # --- 1. Black Scholes ---
            start_time = time.perf_counter()
            bs_call = bs_model.option_price('call')
            bs_put = bs_model.option_price('put')
            bs_greeks = bs_model.calculate_greeks(option_type)
            bs_time = (time.perf_counter() - start_time) * 1000
            pricing_results.append({"Model": "Black-Scholes (Analytical)", "Call Price": bs_call, "Put Price": bs_put, "Compute Time (ms)": bs_time})

            # --- 2. Binomial Tree ---
            start_time = time.perf_counter()
            tree_model = BinomialOptionPricing(s0=spot, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol, N=tree_steps)
            _, call_tree = tree_model.option_prices('call')
            _, put_tree = tree_model.option_prices('put')
            tree_greeks = tree_model.calculate_greeks(option_type)
            tree_time = (time.perf_counter() - start_time) * 1000
            pricing_results.append({"Model": f"Binomial Tree (N={tree_steps})", "Call Price": call_tree[0,0], "Put Price": put_tree[0,0], "Compute Time (ms)": tree_time})

            # --- 3. Monte Carlo ---
            start_time = time.perf_counter()
            mc_steps = max(int(ttm * 252), 100) # 252 trading days per year
            mc_model = MonteCarloSimulations(s0=spot, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol, num_paths=mc_paths, time_steps=mc_steps)
            mc_call, mc_call_se = mc_model.price_with_control_variate('call')
            mc_put, mc_put_se = mc_model.price_with_control_variate('put')
            mc_greeks = mc_model.calculate_greeks(option_type)
            mc_time = (time.perf_counter() - start_time) * 1000
            pricing_results.append({"Model": f"Monte Carlo ({mc_paths} paths)", "Call Price": mc_call, "Put Price": mc_put, "Compute Time (ms)": mc_time})

            # --- 4. Heston Fourier ---
            start_time = time.perf_counter()
            heston_ft = HestonFourierModel(s0=spot, v0=v0, r=rate, q=div_yield, T=ttm, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta)
            h_ft_call = heston_ft.option_price(K=strike, option_type='call')
            h_ft_put = heston_ft.option_price(K=strike, option_type='put')
            heston_greeks = heston_ft.calculate_greeks(K=strike, option_type=option_type)
            h_ft_time = (time.perf_counter() - start_time) * 1000
            pricing_results.append({"Model": "Heston (Fourier Transform)", "Call Price": h_ft_call, "Put Price": h_ft_put, "Compute Time (ms)": h_ft_time})

            # Construct Pricing DataFrame
            df_pricing = pd.DataFrame(pricing_results)
            
            # Construct Greeks DataFrame
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

            # --- Display Results ---
            st.subheader("Model Pricing Comparison")
            
            # Use Pandas Styler for beautiful table rendering
            styled_pricing = df_pricing.style.format({
                "Call Price": "${:.4f}",
                "Put Price": "${:.4f}",
                "Compute Time (ms)": "{:.2f} ms"
            }).background_gradient(subset=["Compute Time (ms)"], cmap="Reds")
            
            st.dataframe(styled_pricing, use_container_width=True, hide_index=True)
            
            st.caption(f"*Monte Carlo Standard Errors (95% CI): Call ±{mc_call_se*1.96:.4f} | Put ±{mc_put_se*1.96:.4f}*")

            st.subheader(f"Risk Sensitivities ({option_type.capitalize()} Greeks)")
            styled_greeks = df_greeks.style.format({
                "Black-Scholes": "{:.4f}",
                "Binomial Tree": "{:.4f}",
                "Monte Carlo": "{:.4f}",
                "Heston (Fourier)": "{:.4f}"
            })
            st.dataframe(styled_greeks, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Interactive Visualizations")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Option Price Curve (Spot vs Price)
                S_range = np.linspace(max(0.1, spot * 0.5), spot * 1.5, 100)
                bs_prices_curve = [BlackScholesMerton(S=s, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol).option_price(option_type) for s in S_range]
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=S_range, y=bs_prices_curve, mode='lines', name=f'BS {option_type.capitalize()} Price', line=dict(color='#00CC96', width=3)))
                
                current_price = bs_call if option_type == 'call' else bs_put
                fig1.add_trace(go.Scatter(x=[spot], y=[current_price], mode='markers', name='Current Spot', marker=dict(color='#EF553B', size=12)))
                
                fig1.update_layout(title="Option Price vs. Underlying Spot", xaxis_title="Spot Price", yaxis_title=f"{option_type.capitalize()} Premium ($)", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig1, use_container_width=True)

            with col_chart2:
                # Volatility Skew / Heston Convergence (Spot vs Price difference)
                heston_prices_curve = [HestonFourierModel(s0=s, v0=v0, r=rate, q=div_yield, T=ttm, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta).option_price(strike, option_type) for s in S_range]
                price_diff = np.array(heston_prices_curve) - np.array(bs_prices_curve)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=S_range, y=price_diff, name='Heston - BS Premium', marker_color='#AB63FA'))
                fig2.add_hline(y=0, line_dash="dash", line_color="white")
                fig2.update_layout(title=f"Heston Model Premium (Skew Effect for {option_type.capitalize()})", xaxis_title="Spot Price", yaxis_title="Price Difference ($)", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()