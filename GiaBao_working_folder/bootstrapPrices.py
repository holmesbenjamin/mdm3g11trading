import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import OrderedDict 
from arch.bootstrap import StationaryBootstrap

# Extract the close prices of the working commodity
def close_price_extraction(working_commodity):
    df = pd.read_csv("commodity_futures.csv", header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
    df = df.sort_index()
    commodity_px = df[working_commodity]
    commodity_px = commodity_px.dropna()
    return commodity_px

def prices_to_returns(prices):
    # Converts raw prices to returns
    return np.log(prices / prices.shift(1))

def returns_to_prices(returns, intial_value=1):
    # convert returns to a price series, starting at an (optional) given value
    log_price = returns.cumsum() + np.log(intial_value)
    return np.exp(log_price)

def generate_stationary_bootstrap_sample(stationary_time_series: pd.Series, num_samples: int = 1, seed: int = 0) -> pd.DataFrame:
    # Generate sample data from a stationary time-series using the block boostrapping method of Politis & Romano
    optimal_block_size = int(np.ceil(stationary_time_series.shape[0]**(1/3)))
    sbs = StationaryBootstrap(optimal_block_size, stationary_time_series, seed=seed)
    count = 0
    bootstrapped_samples = OrderedDict()
    while count < num_samples:
        ts = next(sbs.bootstrap(1))[0][0]
        ts.index = stationary_time_series.index
        bootstrapped_samples[f"sample_{count}"] = ts
        count += 1
    return pd.DataFrame(bootstrapped_samples)