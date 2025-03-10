import pandas as pd
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
import ast  
from scipy.stats import skew, kurtosis
from scipy.stats import linregress

df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
commodity = 'WTI CRUDE'
data = df[commodity].dropna().values
tau = 1
timeframe_length = 90  # quarterly window (90 days)
resolutions = 100

# sliding window will be moved from index 0 to the last index where the window can fully fit.
max_start = len(data) - (timeframe_length + 2 * tau) + 1

def compute_entropy(lifetimes):
    if len(lifetimes) == 0:
        return np.nan
    total = np.sum(lifetimes)
    if total == 0:
        return np.nan
    p = lifetimes / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def compute_weighted_entropy(lifetimes):
    if len(lifetimes) == 0:
        return np.nan
    weights = lifetimes ** 2
    total = np.sum(weights)
    if total == 0:
        return np.nan
    p = weights / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def process_diagram(diag):
    diag = np.array(diag)
    if diag.ndim == 1:
        diag = diag.reshape(-1, 2)
    diag = diag[np.isfinite(diag).all(axis=1)]
    return diag

results = []

for window_start in range(max_start):
    window_data = np.array([
        data[window_start:window_start + timeframe_length],
        data[window_start + tau:window_start + tau + timeframe_length],
        data[window_start + 2 * tau:window_start + 2 * tau + timeframe_length]
    ]).T

    diagrams = ripser(window_data)['dgms']
    diag0 = process_diagram(diagrams[0])  
    diag1 = process_diagram(diagrams[1])  

    if len(diag0) > 0:
        times_0 = np.linspace(diag0[:, 0].min(), diag0[:, 1].max(), resolutions)
        betti_curve_0 = np.zeros(resolutions)
        for birth, death in diag0:
            betti_curve_0 += (times_0 >= birth) & (times_0 <= death)
    else:
        times_0 = np.linspace(0, 1, resolutions)
        betti_curve_0 = np.zeros(resolutions)

    if len(diag1) > 0:
        times_1 = np.linspace(diag1[:, 0].min(), diag1[:, 1].max(), resolutions)
        betti_curve_1 = np.zeros(resolutions)
        for birth, death in diag1:
            betti_curve_1 += (times_1 >= birth) & (times_1 <= death)
    else:
        times_1 = np.linspace(0, 1, resolutions)
        betti_curve_1 = np.zeros(resolutions)

    lifetimes = diag1[:, 1] - diag1[:, 0] if diag1.size > 0 else np.array([])
    persistence_entropy_H1 = compute_entropy(lifetimes)
    weighted_entropy_H1 = compute_weighted_entropy(lifetimes)

    window_mean = np.mean(window_data[:, 0])
    window_median = np.median(window_data[:, 0])
    window_std = np.std(window_data[:, 0])
    window_var = np.var(window_data[:, 0])
    window_min = np.min(window_data[:, 0])
    window_max = np.max(window_data[:, 0])
    window_range = window_max - window_min

    t_index = np.arange(timeframe_length)
    slope, intercept, r_value, p_value, std_err = linregress(t_index, window_data[:, 0])
    
    window_skew = skew(window_data[:, 0])
    window_kurtosis = kurtosis(window_data[:, 0])
    
    fft_vals = np.fft.fft(window_data[:, 0])
    fft_freq = np.fft.fftfreq(len(window_data[:, 0]))
    idx = np.argmax(np.abs(fft_vals[1:])) + 1  
    dominant_freq = fft_freq[idx]
    
    power_spectrum = np.abs(fft_vals) ** 2
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = -np.sum(power_spectrum_norm * np.log(power_spectrum_norm + 1e-12))
    
    momentum = window_data[-1, 0] - window_data[0, 0]

    window_dict = {
        "window_start": window_start,
        "mean": window_mean,
        "median": window_median,
        "std": window_std,
        "var": window_var,
        "min": window_min,
        "max": window_max,
        "range": window_range,
        "trend_slope": slope,
        "trend_intercept": intercept,
        "r_squared": r_value**2,
        "skew": window_skew,
        "kurtosis": window_kurtosis,
        "dominant_freq": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "momentum": momentum,
        "persistence_entropy_H1": persistence_entropy_H1,
        "weighted_entropy_H1": weighted_entropy_H1,
        "filtration_H0": times_0.tolist(),
        "betti_H0": betti_curve_0.tolist(),
        "filtration_H1": times_1.tolist(),
        "betti_H1": betti_curve_1.tolist()
    }
    results.append(window_dict)

df_combined = pd.DataFrame(results)
df_combined.to_csv("combined_metrics_lists.csv", index=False)

#EXAMPLE

df_combined_loaded = pd.read_csv("combined_metrics_lists.csv")

window_to_plot = 5000
window_df = df_combined_loaded[df_combined_loaded["window_start"] == window_to_plot]

if window_df.empty:
    print(f"No window found with window_start = {window_to_plot}.")
else:
    print("Statistics for window_start =", window_to_plot)
    print(window_df.transpose())
    
    filtration_H0 = ast.literal_eval(window_df["filtration_H0"].iloc[0])
    betti_H0 = ast.literal_eval(window_df["betti_H0"].iloc[0])
    filtration_H1 = ast.literal_eval(window_df["filtration_H1"].iloc[0])
    betti_H1 = ast.literal_eval(window_df["betti_H1"].iloc[0])
    
    plt.figure(figsize=(10, 5))
    plt.plot(filtration_H0, betti_H0, label="H0 Betti Curve", color="blue", marker='o', markersize=3)
    plt.plot(filtration_H1, betti_H1, label="H1 Betti Curve", color="red", marker='o', markersize=3)
    plt.xlabel("Filtration Value")
    plt.ylabel("Betti Number")
    plt.title(f"Betti Curves for Window starting at {window_to_plot}")
    plt.legend()
    plt.grid(True)
    plt.show()
