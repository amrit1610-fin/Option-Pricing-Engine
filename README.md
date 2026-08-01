<div align="center">

# 📈 European Option Pricing Engine

A comprehensive, modular derivatives (*european options*) pricing library and interactive dashboard built in Python. This project compares traditional constant-volatility models against advanced stochastic volatility models and numerical approximation methods.

### 🚀 Live Interactive Dashboard

Experience the engine directly in your browser: [Launch Streamlit Dashboard](https://european-option-pricing-engine.streamlit.app/)

The web application allows you to dynamically adjust market parameters (Spot, Strike, Time, Risk-Free Rate) and Heston dynamics (Vol-of-Vol, Correlation, Mean Reversion) to instantly visualize pricing convergence, Greeks, and volatility skew.

### 🧠 Mathematical Models Implemented

This engine implements four distinct quantitative pricing models to calculate European Call and Put premiums, as well as their respective Greeks ($\Delta, \Gamma, \nu, \Theta, \rho$).

**Black-Scholes-Merton (Analytical)**

The industry standard closed-form solution assuming log-normal asset distributions and constant volatility.

**Cox-Ross-Rubinstein Binomial Tree (Discrete)**

An iterative, lattice-based model allowing for step-by-step risk-neutral valuation and visualization of asset price evolution.

**Monte Carlo Simulations (Stochastic)**

Uses Geometric Brownian Motion (GBM) path generation.

Advanced Feature: Implements Control Variates (using the expected terminal stock price) and Antithetic Variates to drastically reduce standard error and computational variance.

**Heston Stochastic Volatility Model**

Drops the constant volatility assumption of Black-Scholes, utilizing the Cox-Ingersoll-Ross (CIR) process for variance.

*Fourier Transform Method*: Fast, analytical pricing using Carr-Madan characteristic function integration.

*Monte Carlo Method*: Full Truncation Euler scheme to handle the Feller condition and prevent negative variance simulation.

### 📂 Project Architecture

The repository is built with a strictly modular, object-oriented design, separating core financial logic from numerical engines and the UI.

Option-Pricing-Engine/
│
├── core/                   
│   └── market_data.py      # Dataclasses standardizing Market & Instrument inputs
│
├── models/                 # Core Pricing Engines
│   ├── black_scholes.py    
│   ├── binomial_tree.py    
│   ├── monte_carlo.py      
│   └── heston_stoch_vol.py 
│
├── app.py                  # Streamlit Interactive Web Dashboard
├── main.py                 # CLI Execution & Profiling Script
└── requirements.txt        # Deployment dependencies


### ⚙️ Installation & Local Usage

To run this engine locally on your machine, follow these steps:

1. Clone the repository

git clone https://github.com/amrit1610-fin/Option-Pricing-Engine.git
cd Option-Pricing-Engine


2. Install dependencies
It is recommended to use a virtual environment.

pip install -r requirements.txt


3. Run the CLI Engine
To output a highly formatted terminal table comparing model prices, Greeks, and compute times:

python main.py


4. Launch the Web Dashboard
To boot up the interactive Streamlit application:

python -m streamlit run app.py


### 🧪 Model Validation & Accuracy

To ensure mathematical rigor, the models have been cross-validated against theoretical boundary conditions:

*Convergence*: The Binomial Tree and Monte Carlo models mathematically converge to the Black-Scholes analytical price as $N \to \infty$.

*Put-Call Parity*: All models strictly respect $C - P = S_0 e^{-qT} - K e^{-rT}$.

*Heston Collapse Test*: When Vol-of-Vol ($\sigma_v \to 0$) and Correlation ($\rho \to 0$), the Heston Fourier pricing exactly matches the Black-Scholes constant volatility pricing.

*Standardized Greeks*: All numerical derivatives (calculated via Finite Difference "Bump and Revalue") are scaled to standard trading metrics (e.g., Vega per 1% vol, Theta per 1 day).

👨‍💻 Author

**Amritanshu**

Quantitative Finance Enthusiast | Data Scientist
