import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

commodity_futures_differencing = 'hugh_working_folder/commodity_futures_adjusted/commodity_futures_differencing.csv'
commodity_futures_log_returns = 'hugh_working_folder/commodity_futures_adjusted/commodity_futures_log_returns.csv'
commodity_futures_percentage_returns = 'hugh_working_folder/commodity_futures_adjusted/commodity_futures_percentage_returns.csv'

df = pd.read_csv(commodity_futures_log_returns, parse_dates=['Date'], index_col='Date')

