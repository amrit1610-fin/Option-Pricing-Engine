from dataclasses import dataclass

@dataclass
class MarketData:
    spot_price : float
    risk_free_rate : float
    time_to_expiry : float
    dividend_yield : float
    volatility : float
    strike_price : float

data = MarketData(
    spot_price = 105.0,
    risk_free_rate = 0.07,
    time_to_expiry = 2.0,
    dividend_yield = 0.05,
    volatility = 0.2,
    strike_price = 110.0
)

# print(data.spot_price)