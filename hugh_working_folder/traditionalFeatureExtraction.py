import pandas as pd
import numpy as np
import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


# --- FEATURE LIST --- #
fc_parameters = {
# Statistical Summary Features
"mean": None,
"median": None,
"standard_deviation": None,
"variance": None,
"skewness": None,
"kurtosis": None,

# Trend Features
"linear_trend": [{"attr": "slope"}],
"longest_strike_above_mean": None,
"longest_strike_below_mean": None,

# Volatility and Change Features
"abs_energy": None,  # corrected feature name
"mean_abs_change": None,
"absolute_sum_of_changes": None,
"number_peaks": [{"n": 3}],  # Adjust 'n' based on the resolution you need
"change_quantiles": [{"ql": 0.25, "qh": 0.75, "isabs": True, "f_agg": "mean"}],

# Frequency Domain Features
"fft_coefficient": [{"coeff": 0, "attr": "abs"}],

# Complexity and Autocorrelation Features
"cid_ce": [{"normalize": True}],  # Added required parameter normalize
"mean_second_derivative_central": None,
"autocorrelation": [{"lag": 1}]
}
# --- FEATURE LIST --- #





def create_tsfresh_flat_dataframe(series: pd.Series, id: str) -> pd.DataFrame:
    df = pd.DataFrame({'id': id, 'time': series.index, 'x': series, 'y': np.sign(series.shift(-1))}).dropna(how='any', axis=0).reset_index(drop=True)
    return df

def create_tsfresh_flat_dataframe_monthly_target(series: pd.Series, id: str, window: int = 21) -> pd.DataFrame:
    # Compute the cumulative log return over the next 'window' days
    monthly_future_return = series.rolling(window=window).sum().shift(-window + 1)
    # Create the dataframe; use the sign of the monthly cumulative return as the target
    df = pd.DataFrame({
        'id': id,
        'time': series.index,
        'x': series,
        'y': np.sign(monthly_future_return)
    }).dropna(how='any', axis=0).reset_index(drop=True)
    return df

def roll_and_extract(df, working_days):

    # Roll the time series
    df_rolled = roll_time_series(df, column_id="id", min_timeshift=working_days, max_timeshift=working_days)

    # Extract features
    df_features = tsf.extract_features(df_rolled, default_fc_parameters=fc_parameters, column_id="id", column_sort="time", column_value="x")
    df_features = df_features.dropna(how='all', axis=1)
    print(df_features.shape)

    # Extract the true values
    y_true = df_rolled.groupby("id").last()[['y', 'time']]
    y_true.index = pd.MultiIndex.from_tuples(y_true.index)
    X = df_features.join(y_true).drop(axis=1, labels='y').set_index('time').sort_index()
    # print(X)
    y = y_true.set_index('time').sort_index().squeeze()
    # print(y)

    return X, y