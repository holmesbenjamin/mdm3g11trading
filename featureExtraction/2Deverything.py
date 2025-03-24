import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.animation import FuncAnimation
from ripser import ripser

# Load data
df = pd.read_csv("datasets/commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Get list of commodities
commodities = df.columns

# Set time delay (tau) and timeframe length
tau = 7
timeframe_length = 50

# Define the start position
start_position = 3000

# Select the commodity for animation
commodity = 'GOLD'  # Adjust for the commodity you want to visualize

# Check if the commodity has enough data
if df[commodity].notna().sum() == 0:
    raise ValueError(f"Commodity {commodity} has no valid data.")
data = df[commodity].dropna().values
dates = df[commodity].dropna().index

# Ensure timeframe_length and tau are within bounds
if len(dates) < start_position + 2 * tau + timeframe_length:
    raise ValueError(f"Not enough data points for commodity {commodity}.")

# Prepare data for 2D delay embedding
delayed_data = np.array([
    data[start_position:start_position + timeframe_length],
    data[start_position + tau:start_position + tau + timeframe_length]
]).T

# First plot: Time series of the commodity over time
fig_time = plt.figure(figsize=(8, 6))
ax_time = fig_time.add_subplot(111)

ax_time.plot(dates, data, label=commodity, color='black', marker=',', linestyle='-')
ax_time.axvline(dates[start_position], color='red', linestyle='--', lw=1, label='First Timeframe')
ax_time.axvline(dates[start_position + timeframe_length], color='red', linestyle='--', lw=1)

ax_time.axvline(dates[start_position + tau], color='green', linestyle='--', lw=1, label='Second Timeframe')
ax_time.axvline(dates[start_position + tau + timeframe_length], color='green', linestyle='--', lw=1)

ax_time.axvline(dates[start_position + 2 * tau], color='blue', linestyle='--', lw=1, label='Third Timeframe')
ax_time.axvline(dates[start_position + 2 * tau + timeframe_length], color='blue', linestyle='--', lw=1)

ax_time.set_title(f"{commodity} Over Time")
ax_time.set_xlabel("Year")
ax_time.set_ylabel("Price/Value")
ax_time.xaxis.set_major_locator(mdates.YearLocator())
ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_time.grid(True)
ax_time.legend()

# Second plot: Animated 2D Delay Embedding with growing epsilon
fig_embed = plt.figure(figsize=(8, 8))
ax_embed = fig_embed.add_subplot(111)

scatter = ax_embed.scatter([], [], s=10, color='red', label='Points')

# Function to update the plot for each frame of the animation
def update(frame):
    epsilon = frame  # Epsilon grows from 1 to 20

    ax_embed.clear()
    
    # Redraw the points
    ax_embed.scatter(delayed_data[:, 0], delayed_data[:, 1], s=10, color='red', label='Points')

    # Draw filled epsilon-radius circles around each point (2D equivalent of spheres)
    for point in delayed_data:
        u = np.linspace(0, 2 * np.pi, 100)
        x = epsilon * np.cos(u) + point[0]
        y = epsilon * np.sin(u) + point[1]
        ax_embed.fill(x, y, color='lightgrey', alpha=0.5)  # Fill the circles with light grey

    # Connect points whose circles overlap (check distance between points)
    num_points = len(delayed_data)
    for i in range(num_points):
        for j in range(i + 1, num_points):  # Only check each pair once
            dist = np.linalg.norm(delayed_data[i] - delayed_data[j])
            if dist <= 2 * epsilon:  # If the distance is less than or equal to 2*epsilon
                ax_embed.plot([delayed_data[i, 0], delayed_data[j, 0]], 
                              [delayed_data[i, 1], delayed_data[j, 1]], 
                              color='black', lw=0.5)

    ax_embed.set_xlabel(f"{commodity} at time t")
    ax_embed.set_ylabel(f"{commodity} at time t+{tau}")
    ax_embed.set_title(f"2D Delay Embedding of {commodity} (ε={epsilon}, τ={tau}, length={timeframe_length})")
    ax_embed.legend()

# Create the animation for the second plot with a slower interval
ani = FuncAnimation(fig_embed, update, frames=np.arange(1, 21), repeat=False, interval=500)  # interval=500ms for slower animation

plt.show()
