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
    page_title="European Option Pricing Engine",
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
    
    st.markdown('<p class="main-header">European Option Pricing Engine</p>', unsafe_allow_html=True)
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
            st.subheader("Interactive Analytics & Visualizations")
            
            # --- 4. Create Tabs for different analytical views ---
            tab1, tab2, tab3 = st.tabs([
                "Price Sensitivity (Spot)", 
                "Numerical Convergence (MC & Tree)", 
                "Heston Volatility Skew"
            ])
            
            with tab1:
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
                    # Greeks Surface (Spot vs Delta)
                    bs_deltas_curve = [BlackScholesMerton(S=s, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol).calculate_greeks(option_type)['delta'] for s in S_range]
                    
                    fig_delta = go.Figure()
                    fig_delta.add_trace(go.Scatter(x=S_range, y=bs_deltas_curve, mode='lines', name=f'{option_type.capitalize()} Delta', line=dict(color='#FFA15A', width=3)))
                    fig_delta.update_layout(title=f"Delta Sensitivity Profile", xaxis_title="Spot Price", yaxis_title="Delta", template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig_delta, use_container_width=True)

            with tab2:
                col_conv1, col_conv2 = st.columns(2)
                
                with col_conv1:
                    # Monte Carlo Convergence
                    # We run a fast simulation to track running average
                    mc_fast = MonteCarloSimulations(s0=spot, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol, num_paths=mc_paths, time_steps=max(int(ttm * 252), 50))
                    paths = mc_fast._generate_paths()
                    terminal_prices = paths[:, -1]
                    
                    if option_type == 'call':
                        sim_payoffs = np.maximum(terminal_prices - strike, 0)
                    else:
                        sim_payoffs = np.maximum(strike - terminal_prices, 0)
                        
                    discounted_payoffs = np.exp(-rate * ttm) * sim_payoffs
                    
                    # Calculate cumulative average
                    cumulative_avg = np.cumsum(discounted_payoffs) / np.arange(1, mc_paths + 1)
                    
                    # Subsample for plotting performance (max 1000 points)
                    plot_step = max(1, mc_paths // 1000)
                    x_paths = np.arange(1, mc_paths + 1, plot_step)
                    y_prices = cumulative_avg[::plot_step]
                    
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Scatter(x=x_paths, y=y_prices, mode='lines', name='MC Estimate', line=dict(color='#AB63FA', width=2)))
                    fig_mc.add_hline(y=bs_call if option_type=='call' else bs_put, line_dash="dash", line_color="white", annotation_text="Analytical Price")
                    fig_mc.update_layout(title="Monte Carlo Convergence", xaxis_title="Number of Paths", yaxis_title="Estimated Price", template="plotly_dark")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                with col_conv2:
                    # Binomial Tree Oscillation
                    tree_steps_range = np.arange(10, 150, 5)
                    tree_prices = []
                    for n in tree_steps_range:
                        model = BinomialOptionPricing(s0=spot, K=strike, r=rate, q=div_yield, T=ttm, sigma=vol, N=n)
                        _, t_tree = model.option_prices(option_type)
                        tree_prices.append(t_tree[0,0])
                        
                    fig_tree = go.Figure()
                    fig_tree.add_trace(go.Scatter(x=tree_steps_range, y=tree_prices, mode='lines+markers', name='Tree Price', line=dict(color='#19D3F3', width=2)))
                    fig_tree.add_hline(y=bs_call if option_type=='call' else bs_put, line_dash="dash", line_color="white", annotation_text="Analytical Price")
                    fig_tree.update_layout(title="Binomial Tree 'Even/Odd' Oscillation", xaxis_title="Number of Steps (N)", yaxis_title="Estimated Price", template="plotly_dark")
                    st.plotly_chart(fig_tree, use_container_width=True)

            with tab3:
                # Heston Volatility Skew (Price Diff across Strikes)
                st.markdown("##### The Heston Skew Effect vs. Black-Scholes")
                st.markdown("Because Black-Scholes assumes constant volatility, it often misprices deep Out-of-the-Money (OTM) options. The Heston model, using your $\\rho$ (correlation) and $\\sigma_v$ (vol-of-vol) inputs, captures this 'Volatility Smile/Skew'.")
                
                K_range = np.linspace(spot * 0.7, spot * 1.3, 40)
                
                skew_bs_prices = [BlackScholesMerton(S=spot, K=k, r=rate, q=div_yield, T=ttm, sigma=vol).option_price(option_type) for k in K_range]
                skew_heston_prices = [HestonFourierModel(s0=spot, v0=v0, r=rate, q=div_yield, T=ttm, sigma=sigma_v, rho=rho, kappa=kappa, theta=theta).option_price(K=k, option_type=option_type) for k in K_range]
                
                skew_diff = np.array(skew_heston_prices) - np.array(skew_bs_prices)
                
                fig_skew = go.Figure()
                fig_skew.add_trace(go.Bar(x=K_range, y=skew_diff, name='Heston Price - BS Price', marker_color='#FF6692'))
                fig_skew.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="At-The-Money (ATM)")
                fig_skew.update_layout(
                    title=f"Heston Mispricing vs. Black-Scholes across Strikes ({option_type.capitalize()})", 
                    xaxis_title="Strike Price (K)", 
                    yaxis_title="Price Difference ($)", 
                    template="plotly_dark", 
                    hovermode="x unified"
                )
                st.plotly_chart(fig_skew, use_container_width=True)

if __name__ == "__main__":
    main()