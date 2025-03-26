import pandas as pd
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

# --- Load Data ---
df = pd.read_csv("datasets/commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
commodity = 'GOLD'
data = df[commodity].dropna().values

tau = 1
timeframe_length = 90    
resolutions = 100        
window_start = 5050      

window_data = np.array([
    data[window_start: window_start + timeframe_length],
    data[window_start + tau: window_start + tau + timeframe_length],
    data[window_start + 2 * tau: window_start + 2 * tau + timeframe_length]
]).T

result = ripser(window_data)
dgms = result['dgms']

def process_diagram(diag):
    diag = np.array(diag)
    if diag.ndim == 1:
        diag = diag.reshape(-1, 2)
    diag = diag[np.isfinite(diag).all(axis=1)]
    return diag

dgm0 = process_diagram(dgms[0])
dgm1 = process_diagram(dgms[1])

if len(dgm0) > 0:
    t_min0 = dgm0[:, 0].min()
    t_max0 = dgm0[:, 1].max()
else:
    t_min0, t_max0 = 0, 1

if len(dgm1) > 0:
    t_min1 = dgm1[:, 0].min()
    t_max1 = dgm1[:, 1].max()
else:
    t_min1, t_max1 = 0, 1

filtration_H0 = np.linspace(t_min0, t_max0, resolutions)
filtration_H1 = np.linspace(t_min1, t_max1, resolutions)

betti_curve_0 = np.zeros(resolutions)
for birth, death in dgm0:
    betti_curve_0 += (filtration_H0 >= birth) & (filtration_H0 <= death)

betti_curve_1 = np.zeros(resolutions)
for birth, death in dgm1:
    betti_curve_1 += (filtration_H1 >= birth) & (filtration_H1 <= death)

fig_pd, ax_pd = plt.subplots(figsize=(8, 8))
ax_pd.set_xlabel(r"Birth")
ax_pd.set_ylabel(r"Death")

if len(dgm0) > 0 and len(dgm1) > 0:
    birth_min = min(dgm0[:,0].min(), dgm1[:,0].min())
    birth_max = max(dgm0[:,0].max(), dgm1[:,0].max())
    death_max = max(dgm0[:,1].max(), dgm1[:,1].max())
elif len(dgm0) > 0:
    birth_min, birth_max, death_max = dgm0[:,0].min(), dgm0[:,0].max(), dgm0[:,1].max()
elif len(dgm1) > 0:
    birth_min, birth_max, death_max = dgm1[:,0].min(), dgm1[:,0].max(), dgm1[:,1].max()
else:
    birth_min, birth_max, death_max = 0, 1, 1

ax_pd.set_xlim(birth_min - 0.1, birth_max + 0.1)
ax_pd.set_ylim(0, death_max + 0.5)
ax_pd.plot([birth_min, birth_max], [birth_min, birth_max], 'k--', lw=1)  

if len(dgm0) > 0:
    sorted_idx0 = np.argsort(dgm0[:, 1])
    dgm0_sorted = dgm0[sorted_idx0]
else:
    dgm0_sorted = np.empty((0, 2))
if len(dgm1) > 0:
    sorted_idx1 = np.argsort(dgm1[:, 1])
    dgm1_sorted = dgm1[sorted_idx1]
else:
    dgm1_sorted = np.empty((0, 2))

num_frames = max(len(dgm0_sorted), len(dgm1_sorted))

scat0 = ax_pd.scatter(np.empty((0,)), np.empty((0,)), s=100, c="blue", alpha=0.7, label=r"$H_0$")
scat1 = ax_pd.scatter(np.empty((0,)), np.empty((0,)), s=100, c="red", alpha=0.7, label=r"$H_1$")
ax_pd.legend()

def init_pd():
    scat0.set_offsets(np.empty((0, 2)))
    scat1.set_offsets(np.empty((0, 2)))
    return scat0, scat1

def update_pd(frame):
    current_points0 = dgm0_sorted[:min(frame+1, len(dgm0_sorted))]
    current_points1 = dgm1_sorted[:min(frame+1, len(dgm1_sorted))]
    scat0.set_offsets(current_points0)
    scat1.set_offsets(current_points1)
    return scat0, scat1

ani_pd = animation.FuncAnimation(fig_pd, update_pd, frames=num_frames,
                                 init_func=init_pd, interval=200, blit=True, repeat=True)
ani_pd.save("images/results/pd_gold.gif", writer="pillow", fps=5)
fig_betti, ax_betti = plt.subplots(figsize=(10, 6))
ax_betti.set_xlabel(r"Filtration Value")
ax_betti.set_ylabel(r"Betti Number")
ax_betti.set_xlim(min(filtration_H0[0], filtration_H1[0]), max(filtration_H0[-1], filtration_H1[-1]))
y_max = max(betti_curve_0.max(), betti_curve_1.max()) + 1
ax_betti.set_ylim(0, y_max)
ax_betti.grid(True)

line0, = ax_betti.plot([], [], label=r"$\beta_0$", color="blue", lw=2)
line1, = ax_betti.plot([], [], label=r"$\beta_1$", color="red", lw=2)
ax_betti.legend()

def init_betti():
    line0.set_data([], [])
    line1.set_data([], [])
    return line0, line1

def update_betti(frame):
    t_plot0 = filtration_H0[:frame+1]
    curve0 = betti_curve_0[:frame+1]
    line0.set_data(t_plot0, curve0)
    t_plot1 = filtration_H1[:frame+1]
    curve1 = betti_curve_1[:frame+1]
    line1.set_data(t_plot1, curve1)
    return line0, line1

ani_betti = animation.FuncAnimation(fig_betti, update_betti, frames=resolutions,
                                    init_func=init_betti, interval=30, blit=True, repeat=True)
ani_betti.save("images/results/betti_curve_gold.gif", writer="pillow", fps=5)

plt.show()
