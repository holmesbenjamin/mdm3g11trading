import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import sys, os, argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ast
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

def backtest_model(X, commodity_returns, clf, tss, mode):
    strategy_returns_list = []
    predictions_all = []
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        print(f"Back-testing fold {fold + 1}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = np.sign(commodity_returns.iloc[train_idx])
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
        
        actual_returns = commodity_returns.iloc[test_idx]
        strategy_returns = actual_returns * signal
        strategy_returns_list.append(strategy_returns)
    strategy_returns_all = pd.concat(strategy_returns_list).sort_index()
    predictions_all = pd.concat(predictions_all).sort_index()
    return strategy_returns_all, predictions_all

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
                list_data = pd.DataFrame(df[col].tolist(), 
                                         columns=[f"{col}_value_{i+1}" for i in range(len(df[col].iloc[0]))])
                df = pd.concat([df.drop(columns=[col]), list_data], axis=1)
    return df

def close_price_extraction(working_commodity):
    datafile_path = os.path.join(project_root, "datasets", "commodity_futures.csv")
    df = pd.read_csv(datafile_path, header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
    df = df.sort_index()
    commodity_px = df[working_commodity].dropna()
    return commodity_px

def prices_to_returns(prices):
    return np.log(prices / prices.shift(1))

def returns_to_prices(returns, initial_value=1):
    log_price = returns.cumsum() + np.log(initial_value)
    return np.exp(log_price)

def plot_price_with_positions(price_series, signals, timeframe, mode, stage_name):
    plt.figure(figsize=(14, 7))
    plt.plot(price_series.index, price_series, label='Price', color='blue', lw=2)
    signal_dates = signals.index

    long_plotted = False
    short_plotted = False

    for date in signal_dates:
        if date not in price_series.index:
            continue
        sig = signals.loc[date]
        if mode == 'buy_hold':
            if sig == 1:
                plt.scatter(date, price_series.loc[date],
                            marker='^', color='green', s=100,
                            label='Long' if not long_plotted else "")
                long_plotted = True
        elif mode == 'buy_short':
            if sig == 1:
                plt.scatter(date, price_series.loc[date],
                            marker='^', color='green', s=100,
                            label='Long' if not long_plotted else "")
                long_plotted = True
            elif sig == -1:
                plt.scatter(date, price_series.loc[date],
                            marker='v', color='red', s=100,
                            label='Short' if not short_plotted else "")
                short_plotted = True

    plt.title(f"{stage_name} - {timeframe.capitalize()} Signals on Price")
    plt.xticks(rotation=90)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_monthly_signals(price_series, signals, stage_name, timeframe, start_date, end_date):
    mask_price = (price_series.index >= start_date) & (price_series.index <= end_date)
    mask_signals = (signals.index >= start_date) & (signals.index <= end_date)
    price_series = price_series.loc[mask_price]
    signals = signals.loc[mask_signals]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(price_series.index, price_series, label='Price', color='black', lw=2)
    ax.set_title(f"{stage_name} - {timeframe.capitalize()} Signals on Price (Animated)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    signal_dates = signals.index.to_list()
    signal_values = signals.values

    def update(frame):
        ax.collections.clear()  

        current_dates = signal_dates[:frame + 1]
        current_values = signal_values[:frame + 1]

        long_dates = []
        long_prices = []
        short_dates = []
        short_prices = []
        for d, s in zip(current_dates, current_values):
            if d in price_series.index:
                if s > 0:
                    long_dates.append(d)
                    long_prices.append(price_series.loc[d])
                elif s < 0:
                    short_dates.append(d)
                    short_prices.append(price_series.loc[d])
        
        if long_dates and long_prices:
            ax.scatter(long_dates, long_prices, marker='^', color='green', s=100, label='Long')
        if short_dates and short_prices:
            ax.scatter(short_dates, short_prices, marker='v', color='red', s=100, label='Short')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        return ax.collections

    ani = animation.FuncAnimation(fig, update, frames=len(signal_dates), interval=500, blit=False, repeat=False)
    plt.show()
    return ani

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest trading strategy with different signal modes and timeframes.")
    parser.add_argument("--timeframe", choices=["daily", "monthly"], default="monthly",
                        help="Timeframe for signals: daily or monthly.")
    parser.add_argument("--mode", choices=["buy_hold", "buy_short"], default="buy_short",
                        help="Position mode: 'buy_hold' (only long or flat) or 'buy_short' (long and short).")
    return parser.parse_args()

def main():
    args = parse_args()
    timeframe = args.timeframe
    mode = args.mode
    print(f"Running backtest with timeframe: {timeframe}, mode: {mode}")

    working_commodity = "GOLD"
    csv_file = f"datasets/GOLDcombined_metrics_lists.csv"

    stage4_df = featuresFromCSV.extract_stage_4(csv_file)
    stages = {"Stage 4": stage4_df}

    commodity_px = close_price_extraction(working_commodity)
    commodity_returns = prices_to_returns(commodity_px).dropna()

    for stage in stages:
        df = stages[stage]
        df = handle_object_columns(df)
        num_rows = len(df)
        df["window_start"] = commodity_px.index[:num_rows]
        stages[stage] = df

    tss = TimeSeriesSplit(n_splits=23)
    random_state = 42
    svm_models = {"Stage 4": SVC(kernel="linear", C=0.01, random_state=random_state)}

    results = []
    print("Commodity returns range:", commodity_returns.index.min(), "to", commodity_returns.index.max())

    for stage_name, df in stages.items():
        print(f"\n--- Training on {stage_name} ---")
        X = df.drop(columns=["window_start"])
        X.index = pd.to_datetime(df["window_start"])
        y = np.sign(commodity_returns).reindex(X.index).dropna()

        clf = make_pipeline(StandardScaler(), svm_models[stage_name])
        strategy_returns, predictions = backtest_model(X, commodity_returns, clf, tss, mode)
        predictions.index = pd.to_datetime(predictions.index)

        monthly_signal = predictions.resample('ME').apply(
            lambda x: 1 if (x > 0).sum() > (x <= 0).sum() else -1
        )
        
        monthly_log_returns = commodity_returns.resample('ME').sum().dropna()
        common_index = monthly_signal.index.intersection(monthly_log_returns.index)
        if common_index.empty:
            print(f"No overlapping monthly dates for {stage_name}.")
            equity_curve_monthly = pd.Series(dtype=float)
        else:
            strat_monthly = monthly_log_returns.loc[common_index] * monthly_signal.loc[common_index]
            equity_curve_monthly = np.exp(strat_monthly.cumsum())
            cumulative_return_monthly = equity_curve_monthly.iloc[-1] - 1
            monthly_mean = strat_monthly.mean()
            monthly_std = strat_monthly.std()
            annualized_return_monthly = np.exp(12 * monthly_mean) - 1
            annualized_sharpe_monthly = (monthly_mean / monthly_std) * np.sqrt(12) if monthly_std != 0 else np.nan
            print(f"Monthly Backtest for {stage_name} - Cumulative Return: {cumulative_return_monthly:.2%}, " +
                  f"Annualized Return: {annualized_return_monthly:.2%}, Annualized Sharpe: {annualized_sharpe_monthly:.2f}")
        
        equity_curve_monthly = equity_curve_monthly * 1000

        plot_price_with_positions(commodity_px, monthly_signal, timeframe, mode, stage_name)

        start_date = pd.to_datetime("2010-01-01")
        end_date = pd.to_datetime("2015-12-31")
        ani = animate_monthly_signals(commodity_px, monthly_signal, stage_name, timeframe, start_date, end_date)
        #ani.save("monthly_signals.gif", writer="pillow", fps=2)

        f1_scores, accuracy_scores, precision_scores, recall_scores = [], [], [], []
        max_f1_scores, max_precision_scores = [], []
        for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
            print(f"Training fold {fold + 1} for {stage_name}...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average="weighted", zero_division=1))
            recall_scores.append(recall_score(y_test, y_pred, average="weighted", zero_division=1))
            max_f1_scores.append(f1_scores[-1])
            max_precision_scores.append(precision_scores[-1])
        results.append((stage_name,
                        np.mean(f1_scores),
                        np.mean(accuracy_scores),
                        np.mean(precision_scores),
                        np.mean(recall_scores),
                        np.max(max_f1_scores),
                        np.max(max_precision_scores)))
        print(f"{stage_name} Max F1-score: {np.max(max_f1_scores):.4f}")
        print(f"{stage_name} Max Accuracy: {np.mean(accuracy_scores):.4f}")
        print(f"{stage_name} Mean Precision: {np.mean(precision_scores):.4f}")
        print(f"{stage_name} Mean Recall: {np.mean(recall_scores):.4f}")
        print(f"{stage_name} Max Precision: {np.max(max_precision_scores):.4f}")

    results_df = pd.DataFrame(results, columns=["Feature Stage", "Mean F1-Score", "Mean Accuracy", 
                                                  "Mean Precision", "Mean Recall", "Max F1-Score", "Max Precision"])
    print("\nFinal Results:")
    print(results_df)

    plt.figure(figsize=(10,6))
    plt.plot(equity_curve_monthly.index, equity_curve_monthly.values, label="Monthly Signals", color='orange')
    plt.title(f"Portfolio Value for Stage 4")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (£)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
