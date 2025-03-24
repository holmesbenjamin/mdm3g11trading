import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.animation import FuncAnimation
from ripser import ripser
from matplotlib import rcParams

# Update rcParams with the specified settings
rcParams.update({
    "text.usetex": False,  
    "font.family": "serif",
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 0,
    "axes.titlesize": 20,
})

# Load data
df = pd.read_csv("datasets/commodity_futures.csv", parse_dates=["Date"], dayfirst=True)
df.set_index("Date", inplace=True)

# Get list of commodities
commodities = df.columns

# Set time delay (tau) and timeframe length
tau = 7
timeframe_length = 50

# Define the start position
start_position = 3000

# Select the commodity for visualization
commodity = 'GOLD'  # Adjust for the commodity you want to visualize

# Check if the commodity has enough data
if df[commodity].notna().sum() == 0:
    raise ValueError(f"Commodity {commodity} has no valid data.")
data = df[commodity].dropna().values
dates = df[commodity].dropna().index

# Ensure timeframe_length and tau are within bounds
if len(dates) < start_position + tau + timeframe_length:
    raise ValueError(f"Not enough data points for commodity {commodity}.")

# Prepare data for 2D delay embedding
delayed_data = np.array([
    data[start_position:start_position + timeframe_length],
    data[start_position + tau:start_position + tau + timeframe_length]
]).T

# Compute fixed x and y limits
x_min, x_max = np.min(delayed_data[:, 0]), np.max(delayed_data[:, 0])
y_min, y_max = np.min(delayed_data[:, 1]), np.max(delayed_data[:, 1])

# Convert dates to Timestamp explicitly for safe arithmetic
start_date = pd.Timestamp(dates[start_position])
first_end_date = start_date + pd.Timedelta(days=timeframe_length)

# Second timeframe calculations
second_start_date = start_date + pd.Timedelta(days=tau)
second_end_date = second_start_date + pd.Timedelta(days=timeframe_length)


# Fig 1 - Time Series Plot
fig_time = plt.figure()
ax_time = fig_time.add_subplot(111)

ax_time.plot(dates, data, label=commodity, color='black', marker=',', linestyle='-')

# First timeframe (start_date -> start_date + timeframe_length)
ax_time.axvline(start_date, color='red', linestyle='--', lw=1, label='First Timeframe')
ax_time.axvline(first_end_date, color='red', linestyle='--', lw=1)

# Second timeframe (start_date + tau -> start_date + tau + timeframe_length)
ax_time.axvline(second_start_date, color='green', linestyle='--', lw=1, label='Second Timeframe')
ax_time.axvline(second_end_date, color='green', linestyle='--', lw=1)

y_min_range = data[(dates >= start_date - pd.Timedelta(days=30)) & (dates <= second_end_date + pd.Timedelta(days=30))].min()
y_max_range = data[(dates >= start_date - pd.Timedelta(days=30)) & (dates <= second_end_date + pd.Timedelta(days=30))].max()

# Set x-axis limits to +- 30 days from either end
ax_time.set_xlim(start_date - pd.Timedelta(days=30), second_end_date + pd.Timedelta(days=30))
ax_time.set_ylim(y_min_range - 5, y_max_range + 5)

ax_time.set_title(f"{commodity} Over Time")
ax_time.set_xlabel("Date")
ax_time.set_ylabel("Price/Value (€)")

# Modify x-axis to show both months and years
ax_time.xaxis.set_major_locator(mdates.MonthLocator())  # Show ticks for every month
ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format ticks as Month Year (e.g., Jan 2025)
ax_time.grid(True)
ax_time.legend()

# Save the time series plot
time_series_filename = f"commodity_{commodity}_time_series.png"
fig_time.savefig(time_series_filename)

# Fig 2 - Animated 2D Delay Embedding Plot
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

    # Set fixed axis limits
    ax_embed.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax_embed.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    ax_embed.set_xlabel(f"{commodity} at time t")
    ax_embed.set_ylabel(f"{commodity} at time t+{tau}")
    ax_embed.set_title(f"2D Delay Embedding of {commodity} (ε={epsilon}, τ={tau}, length={timeframe_length})")
    ax_embed.legend(loc='upper left')

# Create the animation for the second plot with a slower interval
ani = FuncAnimation(fig_embed, update, frames=np.arange(1, 30, 0.5), repeat=False, interval=50)  

# Save the animation
animation_filename = f"commodity_{commodity}_delay_embedding_animation.gif"
ani.save(animation_filename, writer='Pillow', fps=10)

plt.show()
