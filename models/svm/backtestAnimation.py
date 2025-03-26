import os
import sys
import ast
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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
extraction_path = os.path.join(project_root, "featureExtraction")
if extraction_path not in sys.path:
    sys.path.append(extraction_path)

import featuresFromCSV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_object_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            first_elem = df[col].iloc[0]
            if isinstance(first_elem, list) or (isinstance(first_elem, str) and first_elem.startswith('[')):
                def convert(x):
                    if isinstance(x, str) and x.startswith('['):
                        return ast.literal_eval(x)
                    return x
                df[col] = df[col].apply(convert)
                list_data = pd.DataFrame(
                    df[col].tolist(), 
                    columns=[f"{col}_value_{i+1}" for i in range(len(df[col].iloc[0]))]
                )
                df = pd.concat([df.drop(columns=[col]), list_data], axis=1)
    return df

def close_price_extraction(commodity):
    datafile_path = os.path.join(project_root, "datasets", "commodity_futures.csv")
    df = pd.read_csv(datafile_path, header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index()
    commodity_px = df[commodity].dropna()
    return commodity_px

def prices_to_returns(prices):
    return np.log(prices / prices.shift(1))

def backtest_model(X, target_returns, clf, tss, mode):
    strategy_returns_list = []
    predictions_all = []
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        logging.info(f"Back-testing fold {fold + 1}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = np.sign(target_returns.loc[X_train.index])
        test_dates = X_test.index  

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        predictions_all.append(pd.Series(y_pred, index=test_dates))
        
        if mode == "buy_hold":
            signal = (y_pred > 0).astype(int)
        elif mode == "buy_short":
            signal = y_pred  
        else:
            raise ValueError("Unknown mode. Use 'buy_hold' or 'buy_short'.")
        
        actual_returns = target_returns.loc[X_test.index]
        strategy_returns = actual_returns * signal
        strategy_returns_list.append(strategy_returns)
        
    strategy_returns_all = pd.concat(strategy_returns_list).sort_index()
    predictions_all = pd.concat(predictions_all).sort_index()
    return strategy_returns_all, predictions_all

def animate_daily_signals(price_series, signals, stage_name, start_date, end_date):
    mask_price = (price_series.index >= start_date) & (price_series.index <= end_date)
    mask_signals = (signals.index >= start_date) & (signals.index <= end_date)
    price_series = price_series.loc[mask_price]
    signals = signals.loc[mask_signals]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(f"{stage_name} - Daily Signals on Price (Animated)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    ax.set_xlim(start_date, end_date)
    full_y_min = price_series.min()
    full_y_max = price_series.max()
    ax.set_ylim(full_y_min, full_y_max)

    signal_dates = signals.index.to_list()
    signal_values = signals.values

    price_line, = ax.plot([], [], label='Price', color='black', lw=2)
    long_scatter = ax.scatter([], [], marker='^', color='green', s=100, label='Long')
    short_scatter = ax.scatter([], [], marker='v', color='red', s=100, label='Short')
    
    profit_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, fontsize=16,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(facecolor='white', alpha=0.5))

    ax.xaxis_date()
    fig.autofmt_xdate()

    initial_investment = 1000

    def init():
        price_line.set_data([], [])
        long_scatter.set_offsets(np.empty((0, 2)))
        short_scatter.set_offsets(np.empty((0, 2)))
        profit_text.set_text('')
        return price_line, long_scatter, short_scatter, profit_text

    def update(frame):
        for patch in ax.patches:
            patch.remove()
            
        current_time = signal_dates[frame]
        current_mask = price_series.index <= current_time
        current_dates = price_series.index[current_mask]
        current_prices = price_series[current_mask]
        price_line.set_data(current_dates, current_prices)

        long_dates = []
        long_prices = []
        short_dates = []
        short_prices = []
        for d, s in zip(signal_dates[:frame+1], signal_values[:frame+1]):
            if d in price_series.index:
                if s == 1:
                    long_dates.append(d)
                    long_prices.append(price_series.loc[d])
                elif s == -1:
                    short_dates.append(d)
                    short_prices.append(price_series.loc[d])
        
        if long_dates:
            long_xy = np.column_stack((mdates.date2num(long_dates), long_prices))
            long_scatter.set_offsets(long_xy)
        else:
            long_scatter.set_offsets(np.empty((0, 2)))
        if short_dates:
            short_xy = np.column_stack((mdates.date2num(short_dates), short_prices))
            short_scatter.set_offsets(short_xy)
        else:
            short_scatter.set_offsets(np.empty((0, 2)))

        ax.set_xlim(start_date, end_date)
        ax.set_ylim(full_y_min, full_y_max)
        
        x_end = mdates.date2num(current_time)
        x_start = mdates.date2num(current_time - pd.DateOffset(months=1))
        rect = Rectangle((x_start, full_y_min), x_end - x_start, full_y_max - full_y_min,
                         facecolor='yellow', alpha=0.2, zorder=0)
        ax.add_patch(rect)

        current_prices_series = price_series[current_mask]
        daily_returns = np.log(current_prices_series / current_prices_series.shift(1)).dropna()
        current_signals = signals.reindex(daily_returns.index, method='ffill')
        strat_returns = daily_returns * current_signals
        if not strat_returns.empty:
            equity_curve = np.exp(strat_returns.cumsum())
            current_value = initial_investment * equity_curve.iloc[-1]
            profit = current_value - initial_investment
            profit_text.set_text(f"Portfolio Value: £{current_value:.2f}\nProfit: £{profit:.2f}")
        else:
            profit_text.set_text("")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        return price_line, long_scatter, short_scatter, rect, profit_text

    ani = animation.FuncAnimation(fig, update, frames=len(signal_dates),
                                  init_func=init, interval=300, blit=False, repeat=False)
    plt.show()
    return ani

def main():
    commodity = "GOLD"
    stage = "Stage 3"
    timeframe = "monthly"   
    mode = "buy_short"      
    logging.info(f"Animating {commodity} {stage} {timeframe} {mode}...")

    commodity_px = close_price_extraction(commodity)
    commodity_returns = prices_to_returns(commodity_px).dropna()

    csv_file = os.path.join(project_root, "datasets", "BGOLDcombined_metrics_lists.csv")
    try:
        stage_df = featuresFromCSV.extract_stage_3(csv_file)
    except Exception as e:
        logging.error(f"Error extracting Stage 3 features for {commodity}: {e}")
        return

    stage_df = handle_object_columns(stage_df)
    stage_df["window_end"] = pd.to_datetime(stage_df["window_end"], dayfirst=True)
    X = stage_df.drop(columns=["window_end"])
    X.index = stage_df["window_end"]

    y = np.sign(commodity_returns).reindex(X.index).dropna()
    X = X.loc[y.index]

    tss = TimeSeriesSplit(n_splits=23)
    random_state = 42
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.01, random_state=random_state))

    strategy_returns, predictions = backtest_model(X, commodity_returns, clf, tss, mode)
    daily_signals = predictions.apply(lambda x: 1 if x > 0 else -1)
    
    start_date = pd.to_datetime("2011-01-01")
    end_date = pd.to_datetime("2011-10-15")
    
    ani = animate_daily_signals(commodity_px, daily_signals, f"{commodity} {stage}", start_date, end_date)
    
    # ani.save("gold_stage3_monthly_buy_short.gif", writer="pillow", fps=5)

if __name__ == '__main__':
    main()
