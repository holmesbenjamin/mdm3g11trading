import os
import sys
import argparse
import logging
import ast
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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


def backtest_model(X, target_returns, clf, tss, mode):
    strategy_returns_list = []
    predictions_all = []
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        train_max = X.index[train_idx].max() + pd.Timedelta(days=90)
        test_min = X.index[test_idx].min()
        assert train_max <= test_min, (
            f"Data leakage detected in fold {fold+1}: "
            f"train window ends at {train_max} but test starts at {test_min}"
        )
        
        print(f"Back-testing fold {fold + 1}...")
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
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index()
    commodity_px = df[working_commodity].dropna()
    return commodity_px


def prices_to_returns(prices):
    return np.log(prices / prices.shift(1))


def returns_to_prices(returns, initial_value=1):
    log_price = returns.cumsum() + np.log(initial_value)
    return np.exp(log_price)


def compute_buy_and_hold_return(prices):
    return (prices.iloc[-1] / prices.iloc[0]) - 1


def main():
    timeframes = ['daily', 'monthly']
    modes = ['buy_hold', 'buy_short']
    commodities = ["CORN", "SUGAR", "GOLD"]
    stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

    stage_extractors = {
        "Stage 1": featuresFromCSV.extract_stage_1,
        "Stage 2": featuresFromCSV.extract_stage_2,
        "Stage 3": featuresFromCSV.extract_stage_3,
        "Stage 4": featuresFromCSV.extract_stage_4,
    }

    results = []

    for commodity in commodities:
        logging.info(f"Processing commodity: {commodity}")
        try:
            commodity_px = close_price_extraction(commodity)
        except Exception as e:
            logging.error(f"Error extracting prices for {commodity}: {e}")
            continue

        commodity_returns = prices_to_returns(commodity_px).dropna()
        bh_return = compute_buy_and_hold_return(commodity_px)
        logging.info(f"  Buy & Hold Return for {commodity}: {bh_return:.2%}")

        for stage in stages:
            logging.info(f"  Processing {stage} for {commodity}")
            csv_file = os.path.join(project_root, "datasets", f"{commodity.upper()}combined_metrics_lists.csv")
            try:
                stage_df = stage_extractors[stage](csv_file)
            except Exception as e:
                logging.error(f"Error extracting {stage} features for {commodity}: {e}")
                continue

            stage_df = handle_object_columns(stage_df)
            num_rows = len(stage_df)
            stage_df["window_start"] = commodity_px.index[:num_rows]
            X = stage_df.drop(columns=["window_start"])
            X.index = pd.to_datetime(stage_df["window_start"])
            y = np.sign(commodity_returns).reindex(X.index).dropna()
            X = X.loc[y.index]

            tss = TimeSeriesSplit(n_splits=23, gap=90)
            random_state = 42
            clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.01, random_state=random_state))

            for timeframe in timeframes:
                for mode in modes:
                    logging.info(f"    Running backtest with timeframe: {timeframe}, mode: {mode}")
                    strategy_returns, predictions = backtest_model(X, commodity_returns, clf, tss, mode)

                    if timeframe == "daily":
                        equity_curve = np.exp(strategy_returns.cumsum())
                        cumulative_return = equity_curve.iloc[-1] - 1
                        daily_return = strategy_returns.mean()
                        daily_std = strategy_returns.std()
                        sharpe_ratio = (daily_return / daily_std) * np.sqrt(252) if daily_std != 0 else np.nan
                    elif timeframe == "monthly":
                        if mode == "buy_hold":
                            monthly_signal = predictions.resample('ME').apply(
                                lambda x: 1 if (x > 0).sum() > (x <= 0).sum() else 0
                            )
                        elif mode == "buy_short":
                            monthly_signal = predictions.resample('ME').apply(
                                lambda x: 1 if (x > 0).sum() > (x <= 0).sum() else -1
                            )
                        monthly_log_returns = commodity_returns.resample('ME').sum().dropna()
                        common_index = monthly_signal.index.intersection(monthly_log_returns.index)
                        if common_index.empty:
                            cumulative_return = np.nan
                            sharpe_ratio = np.nan
                        else:
                            strat_monthly = monthly_log_returns.loc[common_index] * monthly_signal.loc[common_index]
                            equity_curve = np.exp(strat_monthly.cumsum())
                            cumulative_return = equity_curve.iloc[-1] - 1
                            monthly_return = strat_monthly.mean()
                            monthly_std = strat_monthly.std()
                            sharpe_ratio = (monthly_return / monthly_std) * np.sqrt(12) if monthly_std != 0 else np.nan
                    else:
                        cumulative_return = np.nan
                        sharpe_ratio = np.nan

                    f1_scores, accuracy_scores, precision_scores, recall_scores = [], [], [], []
                    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
                        accuracy_scores.append(accuracy_score(y_test, y_pred))
                        precision_scores.append(precision_score(y_test, y_pred, average="weighted", zero_division=1))
                        recall_scores.append(recall_score(y_test, y_pred, average="weighted", zero_division=1))

                    metrics = {
                        "Commodity": commodity,
                        "Stage": stage,
                        "Timeframe": timeframe,
                        "Mode": mode,
                        "Mean F1-Score": np.mean(f1_scores),
                        "Mean Accuracy": np.mean(accuracy_scores),
                        "Mean Precision": np.mean(precision_scores),
                        "Mean Recall": np.mean(recall_scores),
                        "Max F1-Score": np.max(f1_scores),
                        "Max Precision": np.max(precision_scores),
                        "Sharpe Ratio": sharpe_ratio,
                        "Cumulative Return": cumulative_return,
                        "BuyHold Return": bh_return,
                        "Num Samples": len(X)
                    }
                    logging.info(f"      Results: {metrics}")
                    results.append(metrics)

    results_df = pd.DataFrame(results)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\nFinal Summary of Backtest Metrics:")
    print(results_df)

    output_file = os.path.join(project_root, "30day_LINEAR_backtest_results_summary_all.csv")
    results_df.to_csv(output_file, index=False)
    logging.info(f"Summary saved to {output_file}")


if __name__ == '__main__':
    main()
