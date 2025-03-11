import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import OrderedDict 
from arch.bootstrap import StationaryBootstrap

# Set the working commodity
working_commodity = "BRENT CRUDE"

# Extract the close prices of the working commodity
df = pd.read_csv("commodity_futures.csv", header=0, index_col=0, thousands=",")
df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
df = df.sort_index()
commodities = df.columns
commodity_px = df[working_commodity]
commodity_px = commodity_px.dropna()

# def prices_to_returns(prices):
#     # Converts raw prices to returns
#     return prices.pct_change()

# commodity_returns = prices_to_returns(commodity_px)

commodity_returns = commodity_px.pct_change()

def returns_to_prices(returns, initial_value=1):
    # convert returns to a price series, starting at an (optional) given value
    return (1 + returns).cumprod() * initial_value

def generate_stationary_bootstrap_sample(stationary_time_series: pd.Series, num_samples: int = 1, seed: int = 0) -> pd.DataFrame:
    # Generate sample data from a stationary time-series using the block boostrapping method of Politis & Romano
    optimal_block_size = int(np.ceil(stationary_time_series.shape[0]**(1/3)))
    sbs = StationaryBootstrap(optimal_block_size, commodity_returns, seed=seed)
    count = 0
    bootstrapped_samples = OrderedDict()
    while count < num_samples:
        ts = next(sbs.bootstrap(1))[0][0]
        ts.index = stationary_time_series.index
        bootstrapped_samples[f"sample_{count}"] = ts
        count += 1
    return pd.DataFrame(bootstrapped_samples)

# Generate stationary bootstrap samples
commodity_returns_samples = generate_stationary_bootstrap_sample(commodity_returns, num_samples=5, seed=1000)


commodity_initial_px = commodity_px.values[0]
commodity_px_samples = commodity_returns_samples.apply(lambda s: returns_to_prices(s, commodity_initial_px))

ax = commodity_px_samples.plot(figsize=(20, 4))
ax = commodity_px.plot(figsize=(20, 4), ax=ax, linestyle="--", color='k', label=working_commodity)
ax.legend()
plt.show()