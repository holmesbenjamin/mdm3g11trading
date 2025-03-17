import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from featuresFromCSV import (
    extract_stage_1, extract_stage_2, extract_stage_3, extract_stage_4
)
import ast

# Function to handle object-type columns (extract values from lists and convert them to numeric)
def handle_object_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is of type 'object'
            if isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):  # Check if the first element is a string representing a list
                # Convert string representations of lists into actual lists
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
                
                # Expand the lists into separate columns
                list_data = pd.DataFrame(df[col].to_list(), columns=[f"{col}_value_{i+1}" for i in range(len(df[col].iloc[0]))])
                
                # Drop the original object column and concatenate the new columns
                df = pd.concat([df.drop(columns=[col]), list_data], axis=1)
                
    return df

# Functions to extract prices and returns
def close_price_extraction(working_commodity):
    df = pd.read_csv("commodity_futures.csv", header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
    df = df.sort_index()
    commodity_px = df[working_commodity]
    commodity_px = commodity_px.dropna()
    return commodity_px

def prices_to_returns(prices):
    return np.log(prices / prices.shift(1))

def returns_to_prices(returns, initial_value=1):
    log_price = returns.cumsum() + np.log(initial_value)
    return np.exp(log_price)


def main():
    # --- Step 1: Load Data ---
    working_commodity = "BRENT CRUDE"
    csv_file = "combined_metrics_lists.csv"

    # Extract features for each stage
    stage1_df = extract_stage_1(csv_file)
    stage2_df = extract_stage_2(csv_file)
    stage3_df = extract_stage_3(csv_file)
    stage4_df = extract_stage_4(csv_file)

    # List of stages and their corresponding datasets
    stages = {
        "Stage 1": stage1_df,
        "Stage 2": stage2_df,
        "Stage 3": stage3_df,
        "Stage 4": stage4_df
    }

    # --- Step 2: Extract Prices and Returns ---
    commodity_px = close_price_extraction(working_commodity)
    commodity_returns = prices_to_returns(commodity_px)

    # Ensure there are no NaN values in the returns
    commodity_returns = commodity_returns.dropna()

    # --- Step 3: Train Models ---
    random_state = 42
    tss = TimeSeriesSplit(n_splits=25)

    # Define models (four linear SVMs)
    svm_models = {
        "SVM Stage 1": SVC(kernel="linear", C=0.01, random_state=random_state),
        "SVM Stage 2": SVC(kernel="linear", C=0.01, random_state=random_state),
        "SVM Stage 3": SVC(kernel="linear", C=0.01, random_state=random_state),
        "SVM Stage 4": SVC(kernel="linear", C=0.01, random_state=random_state)
    }

    # Store results
    results = []

    for stage_name, df in stages.items():
        print(f"\n--- Training on {stage_name} ---")

        # Handle any object columns that need to be expanded
        df = handle_object_columns(df)
        
        # Separate features (X) and target (y)
        X = df.drop(columns=["window_start"])  # Drop non-feature columns
        y = np.sign(commodity_returns)  # Convert returns into classification (up/down movement)

        # Ensure there are no NaN values in the target
        y = y.dropna()

        # Train the corresponding model
        model_name = f"SVM {stage_name}"
        clf = make_pipeline(StandardScaler(), svm_models[model_name])

        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        confusion_matrices = []

        max_f1_scores = []
        max_precision_scores = []

        # Apply time series split
        for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
            print(f"Training fold {fold + 1} for {stage_name}...")
            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_test)

            # Compute metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)

            # Store scores
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)

            # Track max F1 and max Precision
            max_f1_scores.append(f1)
            max_precision_scores.append(precision)

        # Store mean and max metrics for the stage
        results.append((
            stage_name,
            np.mean(f1_scores),
            np.mean(accuracy_scores),
            np.mean(precision_scores),
            np.mean(recall_scores),
            np.max(max_f1_scores),  # Max F1
            np.max(max_precision_scores)  # Max Precision
        ))

        print(f"{stage_name} Max F1-score: {np.max(max_f1_scores):.4f}")
        print(f"{stage_name} Max Accuracy: {np.mean(accuracy_scores):.4f}")
        print(f"{stage_name} Mean Precision: {np.mean(precision_scores):.4f}")
        print(f"{stage_name} Mean Recall: {np.mean(recall_scores):.4f}")
        print(f"{stage_name} Max Precision: {np.max(max_precision_scores):.4f}")

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results, columns=[
        "Feature Stage", "Mean F1-Score", "Mean Accuracy", "Mean Precision", "Mean Recall", "Max F1-Score", "Max Precision"
    ])
    print("\nFinal Results:")
    print(results_df)

if __name__ == '__main__':
    main()



#Final Results:
#  Feature Stage  Mean F1-Score  Mean Accuracy   Mean Recall  Max F1-Score  Max Precision
#0       Stage 1       0.354992       0.515652      0.515652      0.434187       0.765898
#1       Stage 2       0.418874       0.529739      0.529739      0.548626       0.756901
#2       Stage 3       0.484373       0.529217      0.529217      0.590832       0.606008
#3       Stage 4       0.489744       0.532000      0.532000      0.586264       0.606008