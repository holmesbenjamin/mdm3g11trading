import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load data
df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

commodities = df.columns

tau = 1
timeframe_length = 50

# Define the start position
start_position = 5000

# Set epsilon for topological data analysis
epsilon = 0.5

# Plot selected commodities and 3D delay embedding
for commodity in commodities:
    if commodity in ['WTI CRUDE', 'BRENT CRUDE']:
        if df[commodity].notna().sum() == 0:
            continue
        data = df[commodity].dropna().values
        dates = df[commodity].dropna().index

        # Ensure timeframe_length and tau are within bounds
        if len(dates) < start_position + 2 * tau + timeframe_length:
            print(f"Skipping {commodity} as there are not enough data points.")
            continue

        # Debug: Check the range of dates
        print(f"Dates range for {commodity}: {dates[0]} to {dates[-1]}")

        # Plot entire time series with three timeframes highlighted
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, data, label=commodity, color='black', marker=',', linestyle='-')

        # Draw vertical lines for the timeframes
        ax.axvline(dates[start_position], color='red', linestyle='--', lw=1, label='First Timeframe')
        ax.axvline(dates[start_position + timeframe_length], color='red', linestyle='--', lw=1)

        ax.axvline(dates[start_position + tau], color='green', linestyle='--', lw=1, label='Second Timeframe')
        ax.axvline(dates[start_position + tau + timeframe_length], color='green', linestyle='--', lw=1)

        ax.axvline(dates[start_position + 2 * tau], color='blue', linestyle='--', lw=1, label='Third Timeframe')
        ax.axvline(dates[start_position + 2 * tau + timeframe_length], color='blue', linestyle='--', lw=1)

        ax.set_title(f"{commodity} Over Time")
        ax.set_xlabel("Year")
        plt.xticks(rotation=45)
        ax.set_ylabel("Price/Value")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True)
        ax.legend()
        plt.show()

        # Prepare data for 3D delay embedding
        delayed_data = np.array([
            data[start_position:start_position + timeframe_length],
            data[start_position + tau:start_position + tau + timeframe_length],
            data[start_position + 2 * tau:start_position + 2 * tau + timeframe_length]
        ]).T

        # Plot 3D delay embedding with epsilon spheres and connecting lines
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw connecting lines first so they appear on top
        for i, point in enumerate(delayed_data):
            for j, other_point in enumerate(delayed_data):
                if i != j and np.linalg.norm(point - other_point) <= 2 * epsilon:
                    ax.plot([point[0], other_point[0]], [point[1], other_point[1]], [point[2], other_point[2]], color='black', lw=0.5)

        # Draw semi-transparent spheres for epsilon neighborhood
        for point in delayed_data:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(np.pi, 0, 10)
            x = epsilon * np.outer(np.cos(u), np.sin(v)) + point[0]
            y = epsilon * np.outer(np.sin(u), np.sin(v)) + point[1]
            z = epsilon * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]
            ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.2)

        ax.scatter(delayed_data[:, 0], delayed_data[:, 1], delayed_data[:, 2], s=5, color='red')

        ax.set_xlabel(f"{commodity} at time t")
        ax.set_ylabel(f"{commodity} at time t+{tau}")
        ax.set_zlabel(f"{commodity} at time t+{2 * tau}")
        ax.set_title(f"3D Delay Embedding of {commodity} (ε={epsilon}, τ={tau}, length={timeframe_length})")
        plt.show()
