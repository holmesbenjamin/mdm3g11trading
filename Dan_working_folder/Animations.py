import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Load data
df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Get list of commodities
commodities = df.columns

# Set time delay (tau) and timeframe length
tau = 1
timeframe_length = 50

# Define the start position
start_position = 5000

# Set epsilon for topological data analysis
epsilon = 0.2

# Plot selected commodities and 2D delay embedding
for commodity in commodities:
    if commodity in ['WTI CRUDE', 'BRENT CRUDE']:
        if df[commodity].notna().sum() == 0:
            continue
        data = df[commodity].dropna().values
        dates = df[commodity].dropna().index

        # Ensure timeframe_length and tau are within bounds
        if len(dates) < start_position + tau + timeframe_length:
            print(f"Skipping {commodity} as there are not enough data points.")
            continue

        # Debug: Check the range of dates
        print(f"Dates range for {commodity}: {dates[0]} to {dates[-1]}")

        # Highlight only the two relevant timeframes from the start position
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates[start_position:], data[start_position:], label=commodity, color='black', marker=',', linestyle='-')

        # Draw vertical lines for the first timeframe
        ax.axvline(dates[start_position], color='red', linestyle='--', lw=1, label='Start of First Timeframe')
        ax.axvline(dates[start_position + timeframe_length], color='blue', linestyle='--', lw=1, label='End of First Timeframe')

        # Draw the second timeframe's vertical lines
        ax.axvline(dates[start_position + tau], color='green', linestyle='--', lw=1, label='Start of Second Timeframe')
        ax.axvline(dates[start_position + tau + timeframe_length], color='black', linestyle='--', lw=1, label='End of Second Timeframe')

        ax.set_title(f"{commodity} Over Time")
        ax.set_xlabel("Year")
        plt.xticks(rotation=45)
        ax.set_ylabel("Price/Value")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True)
        ax.legend()
        plt.show()

        # Prepare data for delay embedding
        delayed_data = np.array([data[start_position:start_position + timeframe_length],
                                 data[start_position + tau:start_position + tau + timeframe_length]]).T

        # Plot 2D delay embedding with epsilon circles and connecting lines
        plt.figure(figsize=(6, 6))
        for i, point in enumerate(delayed_data):
            circle = plt.Circle(point, epsilon, color='lightgray', fill=True)
            plt.gca().add_patch(circle)
            for j, other_point in enumerate(delayed_data):
                if i != j and np.linalg.norm(point - other_point) <= 2 * epsilon:
                    plt.plot([point[0], other_point[0]], [point[1], other_point[1]], color='black', lw=0.5)
        plt.scatter(delayed_data[:, 0], delayed_data[:, 1], s=5, color='red')
        plt.xlabel(f"{commodity} at time t")
        plt.ylabel(f"{commodity} at time t+{tau}")
        plt.title(f"2D Delay Embedding of {commodity} (ε={epsilon}, τ={tau}, length={timeframe_length})")
        plt.grid(True)
        plt.show()