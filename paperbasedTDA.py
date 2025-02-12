#based on paper: https://arxiv.org/html/2405.16052v1#S4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein
from gtda.diagrams import PersistenceLandscape


def force_three_columns(dgm):
    if dgm.size == 0:
        return np.empty((0, 3))
    if dgm.shape[1] == 2:
        return np.hstack((dgm, np.zeros((dgm.shape[0], 1))))
    elif dgm.shape[1] >= 3:
        return dgm[:, :3]
    else:
        return dgm

df = pd.read_csv('commodity_futures.csv')
df['Date'] = pd.to_datetime(df['Date'])
commodityOne = 'WTI CRUDE' #set commodities to compare
commodityTwo = 'BRENT CRUDE'
df = df[['Date', commodityOne,commodityTwo]] 
df = df.dropna().sort_values('Date').reset_index(drop=True)
df['c1_return'] = np.log(df[commodityOne] / df[commodityOne].shift(1))
df['c2_return'] = np.log(df[commodityTwo] / df[commodityTwo].shift(1))
df = df.dropna().reset_index(drop=True)

window_size = 60   # number of days in each sliding window
step_size = 1      # step size for sliding window
num_windows = (len(df) - window_size) // step_size + 1

window_dates = []   
l1_norms = []       
l2_norms = []       
diagrams_list = []  

pl_transform = PersistenceLandscape(n_bins=100, n_layers=5)

print("Processing {} sliding windows...".format(num_windows))
for i in range(num_windows):
    start = i * step_size
    end = start + window_size
    window_data = df.iloc[start:end]

    point_cloud = window_data[['c1_return', 'c2_return']].values

    result = ripser(point_cloud, maxdim=1)
    dgms = result['dgms']
    dgm1 = dgms[1]  # persistence diagram for dimension 1

    dgm1 = force_three_columns(dgm1)
    diagrams_list.append(dgm1)
    
    if dgm1.shape[0] == 0:
        landscape = np.zeros((pl_transform.n_layers, pl_transform.n_bins))
    else:
        landscape = pl_transform.fit_transform([dgm1])[0]  # shape: (n_layers, n_bins)
    
    if dgm1.shape[0] > 0:
        # using the minimum birth and maximum death
        min_val = np.min(dgm1[:, 0])
        max_val = np.max(dgm1[:, 1])
        grid = np.linspace(min_val, max_val, pl_transform.n_bins)
        bin_width = grid[1] - grid[0]
    else:
        bin_width = 1.0

    l1 = np.sum(np.abs(landscape)) * bin_width
    l2 = np.sqrt(np.sum(landscape**2) * bin_width)
    
    l1_norms.append(l1)
    l2_norms.append(l2)
    
    mid_date = window_data['Date'].iloc[window_size // 2]
    window_dates.append(mid_date)

wdists = []    # wasserstein distances
wd_dates = []  

for i in range(1, len(diagrams_list)):
    dgm_prev = diagrams_list[i - 1]
    dgm_curr = diagrams_list[i]
    wd = wasserstein(dgm_prev[:, :2], dgm_curr[:, :2], matching=False)
    wdists.append(wd)
    wd_dates.append(window_dates[i])  

# threshold is set to mean + 4std 
l1_array = np.array(l1_norms)
l2_array = np.array(l2_norms)
wd_array = np.array(wdists)

l1_threshold = np.mean(l1_array) + 4 * np.std(l1_array)
l2_threshold = np.mean(l2_array) + 4 * np.std(l2_array)
wd_threshold = np.mean(wd_array) + 4 * np.std(wd_array)

print("\nThresholds:")
print("L1 Norm Threshold: {:.4e}".format(l1_threshold))
print("L2 Norm Threshold: {:.4e}".format(l2_threshold))
print("Wasserstein Distance Threshold: {:.4e}".format(wd_threshold))

extreme_dates_l1 = [d for d, v in zip(window_dates, l1_array) if v > l1_threshold]
extreme_dates_l2 = [d for d, v in zip(window_dates, l2_array) if v > l2_threshold]
extreme_dates_wd = [d for d, v in zip(wd_dates, wd_array) if v > wd_threshold]

print("\nDetected regime shifts (extreme events) based on persistence landscape norms:")
print("L1-norm spikes at: ", extreme_dates_l1)
print("L2-norm spikes at: ", extreme_dates_l2)
print("Wasserstein distance spikes at: ", extreme_dates_wd)

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(window_dates, l1_array, label='L1 Norm', color='blue')
axs[0].axhline(l1_threshold, color='black', linestyle='-', label='Threshold')
axs[0].set_title('Persistence Landscape L1 Norm (Dimension 1) Over Time')
axs[0].legend()
axs[0].set_ylabel("L1 Norm")

axs[1].plot(window_dates, l2_array, label='L2 Norm', color='green')
axs[1].axhline(l2_threshold, color='black', linestyle='-', label='Threshold')
axs[1].set_title('Persistence Landscape L2 Norm (Dimension 1) Over Time')
axs[1].legend()
axs[1].set_ylabel("L2 Norm")

axs[2].plot(wd_dates, wd_array, label='Wasserstein Distance', color='purple')
axs[2].axhline(wd_threshold, color='black', linestyle='-', label='Threshold')
axs[2].set_title('Wasserstein Distance Between Consecutive Windows (Dimension 1)')
axs[2].legend()
axs[2].set_ylabel("Wasserstein Distance")
axs[2].set_xlabel("Date")

plt.tight_layout()
plt.show()
