import numpy as np
import pandas as pd
from ripser import ripser
from persim import wasserstein
from gtda.diagrams import PersistenceLandscape
from scipy.stats import skew, kurtosis

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_BB_width(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return (upper_band - lower_band) / (rolling_mean + 1e-10)

def compute_ROC(series, period=10):
    return (series - series.shift(period)) / series.shift(period)

def compute_stochastic_oscillator(series, window=14):
    min_val = series.rolling(window=window).min()
    max_val = series.rolling(window=window).max()
    return 100 * (series - min_val) / (max_val - min_val + 1e-10)

def force_three_columns(dgm):
    if dgm.size == 0:
        return np.empty((0, 3))
    if dgm.shape[1] == 2:
        return np.hstack((dgm, np.zeros((dgm.shape[0], 1))))
    return dgm[:, :3]

def compute_diagram_features(dgm):
    if dgm.size == 0:
        return {
            'num_features_H1': 0,
            'total_persistence_H1': 0,
            'mean_persistence_H1': 0,
            'max_persistence_H1': 0,
            'persistence_entropy_H1': 0,
            'median_persistence_H1': 0,
            'std_persistence_H1': 0,
            'weighted_entropy_H1': 0
        }
    lifetimes = dgm[:, 1] - dgm[:, 0]
    num_features = dgm.shape[0]
    total_persistence = np.sum(lifetimes)
    mean_persistence = np.mean(lifetimes)
    max_persistence = np.max(lifetimes)
    median_persistence = np.median(lifetimes)
    std_persistence = np.std(lifetimes)
    entropy = -np.sum((lifetimes / (total_persistence + 1e-10)) * np.log(lifetimes / (total_persistence + 1e-10) + 1e-10)) if total_persistence else 0
    weighted_probs = (lifetimes**2) / (np.sum(lifetimes**2) + 1e-10) if total_persistence else np.zeros_like(lifetimes)
    weighted_entropy = -np.sum(weighted_probs * np.log(weighted_probs + 1e-10)) if total_persistence else 0
    return {
        'num_features_H1': num_features,
        'total_persistence_H1': total_persistence,
        'mean_persistence_H1': mean_persistence,
        'max_persistence_H1': max_persistence,
        'persistence_entropy_H1': entropy,
        'median_persistence_H1': median_persistence,
        'std_persistence_H1': std_persistence,
        'weighted_entropy_H1': weighted_entropy
    }

def compute_betti_curve(dgm, grid_points=50):
    if dgm.size == 0:
        return np.zeros(grid_points)
    birth, death = dgm[:, 0], dgm[:, 1]
    grid = np.linspace(np.min(birth), np.max(death), grid_points)
    return np.array([np.sum((birth <= t) & (death > t)) for t in grid])

def summarize_betti_curve(betti_curve):
    return {
        'betti_mean': np.mean(betti_curve),
        'betti_max': np.max(betti_curve),
        'betti_auc': np.trapz(betti_curve)
    }

def compute_landscape_features(landscape, bin_width=1.0):
    l1_norm = np.sum(np.abs(landscape)) * bin_width
    l2_norm = np.sqrt(np.sum(landscape**2) * bin_width)
    return {
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'max_landscape': np.max(landscape),
        'mean_landscape': np.mean(landscape),
        'std_landscape': np.std(landscape)
    }

def delay_embedding(time_series, embedding_dim, delay):
    n_points = len(time_series) - (embedding_dim - 1) * delay
    return np.array([time_series[i : i + embedding_dim * delay : delay] for i in range(n_points)])


df = pd.read_csv('commodity_futures.csv')
df['Date'] = pd.to_datetime(df['Date'])
commodity = 'GOLD'
df = df[['Date', commodity]].dropna().sort_values('Date').reset_index(drop=True)
df = df[df['Date'] <= pd.to_datetime('2023-08-04')].reset_index(drop=True)

df['log_price'] = np.log(df[commodity])
df['return'] = df['log_price'].diff()

df['smoothed_return'] = df['return'].rolling(window=3, center=True).median()
df['lag1_return'] = df['return'].shift(1)
df['lag2_return'] = df['return'].shift(2)

rolling_window = 10
df['rolling_mean'] = df['log_price'].rolling(window=rolling_window).mean()
df['rolling_std'] = df['log_price'].rolling(window=rolling_window).std()
df['momentum'] = df['log_price'] - df['log_price'].shift(rolling_window)

df['ema'] = df['log_price'].ewm(span=rolling_window, adjust=False).mean()
df['rsi'] = compute_RSI(df[commodity], period=14)
macd_line, macd_signal, macd_hist = compute_MACD(df[commodity])
df['macd_hist'] = macd_hist
df['bb_width'] = compute_BB_width(df[commodity], window=20, num_std=2)
df['roc'] = compute_ROC(df['log_price'], period=10)
df['stochastic'] = compute_stochastic_oscillator(df['log_price'], window=14)

df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['skewness'] = df['return'].rolling(window=rolling_window).apply(lambda x: skew(x), raw=False)
df['kurtosis'] = df['return'].rolling(window=rolling_window).apply(lambda x: kurtosis(x), raw=False)

df = df.dropna().reset_index(drop=True)


window_size = 90       
step_size = window_size  

embedding_dim = 3    
delay = 1              

num_windows = (len(df) - window_size) // step_size + 1
landscape_transform = PersistenceLandscape(n_bins=100, n_layers=7)

feature_vectors = []
prev_dgm_H1 = None

print(f"Extracting features for {num_windows} sliding windows...")
for i in range(num_windows):
    start = i * step_size
    end = start + window_size
    window_data = df.iloc[start:end]
    
    point_cloud = delay_embedding(window_data['smoothed_return'].values, embedding_dim, delay)
    
    result = ripser(point_cloud, maxdim=1)
    dgms = result['dgms']
    dgm_H1 = force_three_columns(dgms[1])
    dgm_H0 = force_three_columns(dgms[0])
    
    if dgm_H1.shape[0] > 0:
        min_val = np.min(dgm_H1[:, 0])
        max_val = np.max(dgm_H1[:, 1])
        grid = np.linspace(min_val, max_val, landscape_transform.n_bins)
        bin_width = grid[1] - grid[0]
        landscape_H1 = landscape_transform.fit_transform([dgm_H1])[0]
    else:
        bin_width = 1.0
        landscape_H1 = np.zeros((landscape_transform.n_layers, landscape_transform.n_bins))
    
    tda_features_H1 = compute_diagram_features(dgm_H1)
    betti_curve_H1 = compute_betti_curve(dgm_H1, grid_points=50)
    betti_summary = summarize_betti_curve(betti_curve_H1)
    landscape_feats = compute_landscape_features(landscape_H1, bin_width)
    
    if prev_dgm_H1 is None or dgm_H1.size == 0 or prev_dgm_H1.size == 0:
        wasserstein_dist = 0
    else:
        wasserstein_dist = wasserstein(prev_dgm_H1[:, :2], dgm_H1[:, :2], matching=False)
    prev_dgm_H1 = dgm_H1.copy()
    
    intrinsic_features = {
        'avg_rolling_mean': window_data['rolling_mean'].mean(),
        'avg_rolling_std': window_data['rolling_std'].mean(),
        'avg_momentum': window_data['momentum'].mean(),
        'avg_ema': window_data['ema'].mean(),
        'avg_rsi': window_data['rsi'].mean(),
        'avg_macd_hist': window_data['macd_hist'].mean(),
        'avg_bb_width': window_data['bb_width'].mean(),
        'avg_roc': window_data['roc'].mean(),
        'avg_stochastic': window_data['stochastic'].mean(),
        'avg_day_of_week': window_data['day_of_week'].mean(),
        'avg_month': window_data['month'].mean(),
        'avg_lag1_return': window_data['lag1_return'].mean(),
        'avg_lag2_return': window_data['lag2_return'].mean(),
        'skewness': window_data['skewness'].mean(),
        'kurtosis': window_data['kurtosis'].mean()
    }
    
    intrinsic_features['num_features_H0'] = dgm_H0.shape[0]
    
    if end < len(df):
        raw_return = df['log_price'].iloc[end] - window_data['log_price'].iloc[-1]
    else:
        raw_return = np.nan  
    
    features = {
        'window_mid_date': window_data['Date'].iloc[window_size // 2],
        'wasserstein_distance': wasserstein_dist,
    }
    features.update(tda_features_H1)
    features.update(betti_summary)
    features.update(landscape_feats)
    features.update(intrinsic_features)
    features['raw_return'] = raw_return  
    
    feature_vectors.append(features)

features_df = pd.DataFrame(feature_vectors).dropna().reset_index(drop=True)
print("Sample extracted features:")
print(features_df.columns)
print(features_df.head())