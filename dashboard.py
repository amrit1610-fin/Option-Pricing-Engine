import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from core.market_data import MarketData
from core.instruments import EuropeanOption, AmericanOption, AsianOption, BarrierOption

from models.black_scholes import BlackScholesEngine
from models.binomial_tree import BinomialTreeEngine
from models.monte_carlo import MonteCarloEngine
from models.heston_stoch_vol import HestonFourierEngine, HestonMonteCarloEngine

st.set_page_config(
    page_title="Quant Option Pricing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
    st.sidebar.title("Contract & Market Params")
    
    st.sidebar.header("1. Option Contract")
    option_type = st.sidebar.selectbox("Option Type", options=["Call", "Put"]).lower()
    style = st.sidebar.selectbox("Exercise Style", options=["European", "American"])
    exotic = st.sidebar.selectbox("Exotic Feature", options=["None", "Asian", "Barrier"])
    
    # Conditional exotic parameters
    averaging_type = 'arithmetic'
    barrier_level = 0.0
    barrier_type = 'up-and-out'
    
    if exotic == "Asian":
        averaging_type = st.sidebar.selectbox("Averaging Type", ["Arithmetic", "Geometric"]).lower()
    elif exotic == "Barrier":
        barrier_type = st.sidebar.selectbox("Barrier Direction", ["Up-and-Out", "Up-and-In", "Down-and-Out", "Down-and-In"]).lower()
        barrier_level = st.sidebar.number_input("Barrier Level", value=120.0, step=1.0)
        
    spot = st.sidebar.number_input("Spot Price (S0)", value=105.0, step=1.0)
    strike = st.sidebar.number_input("Strike Price (K)", value=110.0, step=1.0)
    ttm = st.sidebar.slider("Time to Maturity (Years)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    
    if exotic == "Asian":
        option = AsianOption(strike=strike, option_type=option_type, averaging_type=averaging_type)
    elif exotic == "Barrier":
        option = BarrierOption(strike=strike, option_type=option_type, barrier_level=barrier_level, barrier_type=barrier_type)
    elif style == "American":
        option = AmericanOption(strike=strike, option_type=option_type)
    else:
        option = EuropeanOption(strike=strike, option_type=option_type)

    st.sidebar.header("2. Market Rates")
    rate = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.20, value=0.07, step=0.01)
    div_yield = st.sidebar.slider("Dividend Yield (q)", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
    vol = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
    
    st.sidebar.header("3. Heston Dynamics")
    v0 = st.sidebar.number_input("Initial Variance (v0)", value=0.04, step=0.01, format="%.4f")
    theta = st.sidebar.number_input("Long-Run Variance (θ)", value=0.04, step=0.01, format="%.4f")
    kappa = st.sidebar.slider("Mean Reversion (κ)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    rho = st.sidebar.slider("Correlation (ρ)", min_value=-0.99, max_value=0.99, value=-0.70, step=0.05)
    sigma_v = st.sidebar.slider("Vol of Vol (σ_v)", min_value=0.01, max_value=1.0, value=0.30, step=0.05)

    st.sidebar.header("4. Compute Settings")
    mc_paths = st.sidebar.selectbox("Monte Carlo Paths", options=[5000, 10000, 25000, 50000], index=1)
    tree_steps = st.sidebar.slider("Tree/Sim Steps", min_value=100, max_value=1000, value=250, step=50)

    # Create Market Data Object
    md = MarketData(
        spot_price=spot, 
        risk_free_rate=rate, 
        time_to_expiry=ttm, 
        dividend_yield=div_yield, 
        volatility=vol,
        strike_price=strike # Passed for backwards compatibility if needed
    )

    ENGINE_MAPPING = {
        "Black-Scholes (Analytical)": BlackScholesEngine,
        "Binomial Tree (Discrete)": BinomialTreeEngine,
        "Monte Carlo (GBM)": MonteCarloEngine,
        "Heston (Fourier FFT)": HestonFourierEngine,
        "Heston (Monte Carlo)": HestonMonteCarloEngine
    }
    
    # Ask the classes if they can handle the created 'option' object
    compatible_models = {
        name: cls for name, cls in ENGINE_MAPPING.items() 
        if cls.check_compatibility(option)
    }

    st.sidebar.header("5. Select Compatible Models")
    if not compatible_models:
        st.sidebar.error(f"No engines currently support {style} {exotic} options.")
        selected_model_names = []
    else:
        st.sidebar.success(f"{len(compatible_models)} models compatible!")
        selected_model_names = st.sidebar.multiselect(
            "Models to run:", 
            list(compatible_models.keys()), 
            default=list(compatible_models.keys())
        )

    st.markdown('<p class="main-header">Quantitative Option Pricing Engine</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Currently pricing a <b>{style} {exotic if exotic != "None" else "Standard"} {option_type.capitalize()}</b> option.</p>', unsafe_allow_html=True)
    st.divider()

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moneyness (S/K)", f"{(spot/strike):.4f}")
    col2.metric("Time to Expiry", f"{ttm} Years")
    col3.metric("Style", style)
    col4.metric("Exotic Feature", exotic if exotic != "None" else "Vanilla")

    st.divider()

    if st.button("Calculate Prices & Greeks", type="primary", use_container_width=True):
        if not selected_model_names:
            st.warning("Please select at least one compatible model from the sidebar.")
            return

        with st.spinner(f"Pricing {style} {option_type.capitalize()} using selected engines..."):
            pricing_results = []
            greeks_data = []
            
            # Fast Engine factory
            def get_engine_instance(name):
                cls = compatible_models[name]
                if cls == BlackScholesEngine:
                    return cls(md)
                elif cls == BinomialTreeEngine:
                    return cls(md, N=tree_steps)
                elif cls == MonteCarloEngine:
                    return cls(md, num_paths=mc_paths, time_steps=max(int(ttm * 252), 50))
                elif cls == HestonFourierEngine:
                    return cls(md, v0=v0, rho=rho, kappa=kappa, theta=theta, sigma_v=sigma_v)
                elif cls == HestonMonteCarloEngine:
                    return cls(md, v0=v0, rho=rho, kappa=kappa, theta=theta, sigma_v=sigma_v, steps=max(int(ttm * 252), 50), paths=mc_paths)

            # Loop through selected models
            for name in selected_model_names:
                engine = get_engine_instance(name)
                
                # Pricing
                start_time = time.perf_counter()
                
                # If Monte Carlo, we can extract Standard Error
                if 'Monte Carlo' in name:
                    price, se = engine.calculate_price(option, return_se=True)
                    price_str = f"${price:.4f} (±{se*1.96:.4f})"
                else:
                    price = engine.calculate_price(option)
                    price_str = f"${price:.4f}"
                    
                calc_time = (time.perf_counter() - start_time) * 1000
                
                pricing_results.append({
                    "Model": name, 
                    f"{option_type.capitalize()} Price": price_str, 
                    "Compute Time (ms)": calc_time
                })
                
                # Greeks
                greeks = engine.calculate_greeks(option)
                greeks["Model"] = name
                greeks_data.append(greeks)

            # Construct DataFrames
            df_pricing = pd.DataFrame(pricing_results)
            df_greeks = pd.DataFrame(greeks_data).set_index("Model")
            
            # Display Pricing
            st.subheader("Model Pricing Comparison")
            styled_pricing = df_pricing.style.format({
                "Compute Time (ms)": "{:.2f} ms"
            }).background_gradient(subset=["Compute Time (ms)"], cmap="Reds")
            st.dataframe(styled_pricing, use_container_width=True, hide_index=True)
            
            # Display Greeks
            st.subheader(f"Risk Sensitivities ({option_type.capitalize()} Greeks)")
            styled_greeks = df_greeks.style.format("{:.4f}")
            st.dataframe(styled_greeks, use_container_width=True)

            st.divider()

            st.subheader("Interactive Analytics")
            tab1, tab2 = st.tabs(["Price Sensitivity (Spot)", "Model Speed vs Complexity"])
            
            with tab1:
                st.markdown("##### Price Curve across Spot Values")
                # Intelligently pick the fastest model for the curve drawing to avoid freezing
                fastest_models = [m for m in selected_model_names if "Monte Carlo" not in m]
                model_to_plot = fastest_models[0] if fastest_models else selected_model_names[0]
                
                st.caption(f"*Generating curve using {model_to_plot} for performance.*")
                
                engine_for_plot = get_engine_instance(model_to_plot)
                S_range = np.linspace(max(0.1, spot * 0.5), spot * 1.5, 30)
                prices_curve = []
                
                for s in S_range:
                    md_temp = MarketData(spot_price=s, risk_free_rate=rate, time_to_expiry=ttm, dividend_yield=div_yield, volatility=vol, strike_price=strike)
                    
                    # Cleanly instantiate based on the model type without double-passing market_data
                    if isinstance(engine_for_plot, BlackScholesEngine):
                        temp_engine = BlackScholesEngine(md_temp)
                    elif isinstance(engine_for_plot, BinomialTreeEngine):
                        temp_engine = BinomialTreeEngine(md_temp, N=tree_steps)
                    elif isinstance(engine_for_plot, MonteCarloEngine):
                        temp_engine = MonteCarloEngine(md_temp, num_paths=mc_paths, time_steps=max(int(ttm * 252), 50))
                    elif isinstance(engine_for_plot, HestonFourierEngine):
                        temp_engine = HestonFourierEngine(md_temp, v0=v0, rho=rho, kappa=kappa, theta=theta, sigma_v=sigma_v)
                    elif isinstance(engine_for_plot, HestonMonteCarloEngine):
                        temp_engine = HestonMonteCarloEngine(md_temp, v0=v0, rho=rho, kappa=kappa, theta=theta, sigma_v=sigma_v, steps=max(int(ttm * 252), 50), paths=mc_paths)
                    
                    try:
                        p = temp_engine.calculate_price(option)
                        prices_curve.append(p[0] if isinstance(p, tuple) else p)
                    except:
                        prices_curve.append(np.nan)

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=S_range, y=prices_curve, mode='lines', name=f'{option_type.capitalize()} Price', line=dict(color='#00CC96', width=3)))
                fig1.update_layout(xaxis_title="Spot Price", yaxis_title=f"{option_type.capitalize()} Premium ($)", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                # Bar chart comparing compute times
                fig2 = go.Figure(data=[
                    go.Bar(x=df_pricing['Model'], y=df_pricing['Compute Time (ms)'], marker_color='#AB63FA')
                ])
                fig2.update_layout(title="Compute Time (ms) by Engine", template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()