import pandas as pd
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "text.usetex": True,                
    "font.family": "serif",            
    "font.serif": ["Computer Modern"], 
    "axes.labelsize": 20,              
    "xtick.labelsize": 20,             
    "ytick.labelsize": 20,             
    "legend.fontsize": 20,            
    "figure.titlesize": 0,              
    "axes.titlesize": 0,              
})

df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
commodity = 'WTI CRUDE'
data = df[commodity].dropna().values

tau = 1
timeframe_length = 90  
resolutions = 100

max_start = len(data) - (timeframe_length + 2 * tau) + 1

def process_diagram(diag):
    diag = np.array(diag)
    if diag.ndim == 1:
        diag = diag.reshape(-1, 2)
    diag = diag[np.isfinite(diag).all(axis=1)]
    return diag

results = []
window_means = []  

for window_start in range(max_start):
    window_data = np.array([
        data[window_start:window_start + timeframe_length],
        data[window_start + tau:window_start + tau + timeframe_length],
        data[window_start + 2 * tau:window_start + 2 * tau + timeframe_length]
    ]).T

    window_means.append(np.mean(window_data[:, 0]))

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

    results.append({
        "window_start": window_start,
        "filtration_H0": times_0,
        "betti_H0": betti_curve_0,
        "filtration_H1": times_1,
        "betti_H1": betti_curve_1
    })

df_combined = pd.DataFrame(results)

betti_H0_matrix = np.array(df_combined["betti_H0"].tolist())
betti_H1_matrix = np.array(df_combined["betti_H1"].tolist())

filtration_H0 = df_combined["filtration_H0"].iloc[0]
filtration_H1 = df_combined["filtration_H1"].iloc[0]

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

im0 = axs[0].imshow(betti_H0_matrix, aspect='auto',
                     extent=[filtration_H0[0], filtration_H0[-1],
                             df_combined["window_start"].max(), 0],
                     cmap='viridis')
axs[0].set_title(r"\textbf{Crocker Plot -- $H_0$}")
axs[0].set_xlabel(r"Filtration Value")
axs[0].set_ylabel(r"Window Start")
fig.colorbar(im0, ax=axs[0], label="Betti Number")

# overlay price
# ax_price = axs[0].twiny()
# price_min, price_max = np.min(window_means), np.max(window_means)
# ax_price.set_xlim(price_min, price_max)
# ax_price.plot(window_means, np.arange(max_start), color='white', linewidth=2)
# ax_price.set_xlabel(r"Price Level")
# ax_price.tick_params(axis='x', labelsize=16)

im1 = axs[1].imshow(betti_H1_matrix, aspect='auto',
                     extent=[filtration_H1[0], filtration_H1[-1],
                             df_combined["window_start"].max(), 0],
                     cmap='viridis')
axs[1].set_title(r"\textbf{Crocker Plot -- $H_1$}")
axs[1].set_xlabel(r"Filtration Value")
axs[1].set_ylabel(r"Window Start")
fig.colorbar(im1, ax=axs[1], label="Betti Number")

plt.tight_layout()
plt.show()
