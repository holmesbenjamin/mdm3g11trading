import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib import rcParams

rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
    "axes.titlesize": 0,
})

cur_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(cur_dir, "."))
if project_root not in sys.path:
    sys.path.append(project_root)

def close_price_extraction(commodity):
    datafile_path = os.path.join(project_root, "datasets", "commodity_futures.csv")
    df = pd.read_csv(datafile_path, header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index()
    return df[commodity].dropna()

def animate_gold_price(price_series, start_date, end_date):
    mask = (price_series.index >= start_date) & (price_series.index <= end_date)
    price_series = price_series.loc[mask]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("GOLD Price Animation (2000 to 2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    ax.set_xlim(start_date, end_date)
    ax.set_ylim(price_series.min(), price_series.max())

    price_line, = ax.plot([], [], label="GOLD Price", color="black", lw=2)

    ax.xaxis_date()
    fig.autofmt_xdate()

    dates = price_series.index.to_list()
    prices = price_series.values

    def init():
        price_line.set_data([], [])
        return price_line,

    def update(frame):
        current_dates = dates[:frame+1]
        current_prices = prices[:frame+1]
        price_line.set_data(current_dates, current_prices)
        return price_line,

    frames = np.arange(0, len(dates), 30)
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                init_func=init, interval=2, blit=False, repeat=False)
    plt.legend()
    plt.show()
    return ani

def main():
    commodity = "GOLD"
    gold_prices = close_price_extraction(commodity)
    start_date = pd.to_datetime("2000-01-01")
    end_date = pd.to_datetime("2023-12-31")
    
    ani = animate_gold_price(gold_prices, start_date, end_date)
    ani.save("gold_price_animation.gif", writer="pillow", fps=30)

if __name__ == '__main__':
    main()
