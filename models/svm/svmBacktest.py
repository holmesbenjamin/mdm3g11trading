import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from featuresFromCSV import extract_stage_1, extract_stage_2, extract_stage_3, extract_stage_4
import ast
import matplotlib.pyplot as plt

def backtest_model(X, commodity_returns, clf, tss):
    # Collect daily strategy returns and predictions from each fold.
    strategy_returns_list = []
    predictions_all = []
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        print(f"Back-testing fold {fold + 1}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = np.sign(commodity_returns.iloc[train_idx])
        test_dates = X_test.index  # Expecting a DatetimeIndex

        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Store predictions with their dates
        predictions_all.append(pd.Series(y_pred, index=test_dates))

        # Trading signal: invest if prediction > 0, else 0.
        signal = (y_pred > 0).astype(int)
        actual_returns = commodity_returns.iloc[test_idx]
        strategy_returns = actual_returns * signal
        strategy_returns_list.append(strategy_returns)
    # Combine the results across folds.
    strategy_returns_all = pd.concat(strategy_returns_list).sort_index()
    predictions_all = pd.concat(predictions_all).sort_index()
    return strategy_returns_all, predictions_all

def handle_object_columns(df):
    # Expand columns containing lists (or strings that look like lists) into separate numeric columns.
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
    df = pd.read_csv("datasets/commodity_futures.csv", header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
    df = df.sort_index()
    commodity_px = df[working_commodity].dropna()
    return commodity_px

def prices_to_returns(prices):
    # Compute daily log returns.
    return np.log(prices / prices.shift(1))

def returns_to_prices(returns, initial_value=1):
    log_price = returns.cumsum() + np.log(initial_value)
    return np.exp(log_price)

def main():
    working_commodity = "SILVER"
    csv_file = "datasets/SILVERcombined_metrics_lists.csv"

    # Extract features for each stage.
    stage1_df = extract_stage_1(csv_file)
    stage2_df = extract_stage_2(csv_file)
    stage3_df = extract_stage_3(csv_file)
    stage4_df = extract_stage_4(csv_file)
    stages = {
        "Stage 1": stage1_df,
        "Stage 2": stage2_df,
        "Stage 3": stage3_df,
        "Stage 4": stage4_df
    }

    # Extract commodity futures prices and compute daily log returns.
    commodity_px = close_price_extraction(working_commodity)
    commodity_returns = prices_to_returns(commodity_px).dropna()

    # Process each stage to expand any list columns.
    for stage in stages:
        stages[stage] = handle_object_columns(stages[stage])

    # Map the "window_start" in each stage to real dates from commodity_px.
    for stage in stages:
        df = stages[stage]
        num_rows = len(df)
        # Use the first num_rows dates from commodity_px.index.
        df["window_start"] = commodity_px.index[:num_rows]
        stages[stage] = df

    # Set up TimeSeriesSplit.
    tss = TimeSeriesSplit(n_splits=23)
    random_state = 42
    svm_models = {
        "Stage 1": SVC(kernel="linear", C=0.01, random_state=random_state),
        "Stage 2": SVC(kernel="linear", C=0.01, random_state=random_state),
        "Stage 3": SVC(kernel="linear", C=0.01, random_state=random_state),
        "Stage 4": SVC(kernel="linear", C=0.01, random_state=random_state)
    }

    # Dictionaries to store equity curves for plotting.
    daily_equity_curves = {}
    monthly_equity_curves = {}

    results = []

    # Diagnostic: print date ranges.
    stage2_dates = pd.to_datetime(stage2_df["window_start"])
    print("Stage 2 window_start range (before remapping):", stage2_dates.min(), "to", stage2_dates.max())
    print("Commodity returns range:", commodity_returns.index.min(), "to", commodity_returns.index.max())

    # Loop over each stage.
    for stage_name, df in stages.items():
        print(f"\n--- Training on {stage_name} ---")
        # Build feature matrix; set index using "window_start".
        X = df.drop(columns=["window_start"])
        X.index = pd.to_datetime(df["window_start"])

        # Use the sign of daily log returns as the target.
        y = np.sign(commodity_returns).dropna()

        # Define classifier pipeline.
        clf = make_pipeline(StandardScaler(), svm_models[stage_name])

        # Run daily backtest.
        strategy_returns, predictions = backtest_model(X, commodity_returns, clf, tss)
        predictions.index = pd.to_datetime(predictions.index)

        # -------------------------------
        # Monthly Aggregation (One Trade per Month)
        # -------------------------------
        monthly_signal = predictions.resample('ME').apply(
            lambda x: 1 if (x > 0).sum() > (x <= 0).sum() else -1
        )
        monthly_log_returns = commodity_returns.resample('ME').sum().dropna()
        common_index = monthly_signal.index.intersection(monthly_log_returns.index)
        if common_index.empty:
            print(f"No overlapping monthly dates between predictions and returns for {stage_name}.")
            equity_curve_monthly = pd.Series(dtype=float)
        else:
            strategy_monthly_log_returns = monthly_log_returns.loc[common_index] * monthly_signal.loc[common_index]
            equity_curve_monthly = np.exp(strategy_monthly_log_returns.cumsum())
            cumulative_return_monthly = equity_curve_monthly.iloc[-1] - 1
            monthly_mean = strategy_monthly_log_returns.mean()
            monthly_std = strategy_monthly_log_returns.std()
            annualized_return_monthly = np.exp(12 * monthly_mean) - 1
            annualized_sharpe_monthly = (monthly_mean / monthly_std) * np.sqrt(12) if monthly_std != 0 else np.nan
            print(f"Monthly Backtest for {stage_name} - Cumulative Return: {cumulative_return_monthly:.2%}, " +
                  f"Annualized Return: {annualized_return_monthly:.2%}, Annualized Sharpe Ratio: {annualized_sharpe_monthly:.2f}")

        # -------------------------------
        # Daily Backtest Metrics (Including Annualized Return)
        # -------------------------------
        equity_curve_daily = np.exp(strategy_returns.cumsum())
        cumulative_return_daily = equity_curve_daily.iloc[-1] - 1
        daily_return = strategy_returns.mean()
        daily_std = strategy_returns.std()
        sharpe_ratio_daily = (daily_return / daily_std) * np.sqrt(252) if daily_std != 0 else np.nan
        annualized_return_daily = np.exp(daily_return * 252) - 1
        print(f"Daily Backtest for {stage_name} - Cumulative Return: {cumulative_return_daily:.2%}, " +
              f"Annualized Return: {annualized_return_daily:.2%}, Sharpe Ratio: {sharpe_ratio_daily:.2f}")

        daily_equity_curves[stage_name] = equity_curve_daily * 1000
        monthly_equity_curves[stage_name] = equity_curve_monthly * 1000

        # -------------------------------
        # Cross-Validation Performance Metrics (Daily)
        # -------------------------------
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

    # -------------------------------
    # Plot Portfolio Value for Daily and Monthly Signals for each stage on separate plots.
    # -------------------------------
    for stage in daily_equity_curves.keys():
        plt.figure(figsize=(10,6))
        if not daily_equity_curves[stage].empty:
            plt.plot(daily_equity_curves[stage].index, daily_equity_curves[stage].values, label="Daily Signals", color='blue')
        if stage in monthly_equity_curves and not monthly_equity_curves[stage].empty:
            plt.plot(monthly_equity_curves[stage].index, monthly_equity_curves[stage].values, label="Monthly Signals", color='orange')
        plt.title(f"Portfolio Value for {stage}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (£)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
