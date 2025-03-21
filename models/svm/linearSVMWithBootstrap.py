import models.old_models.bootstrapPrices as bsp
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import ast
from featuresFromCSV import extract_stage_1, extract_stage_2, extract_stage_3, extract_stage_4

# Function to process any object-type columns (as in your linear SVM script)
def handle_object_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if the first element is a string representing a list
            if isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                # Convert string representations to lists and expand them into separate columns
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
                list_data = pd.DataFrame(
                    df[col].to_list(),
                    columns=[f"{col}_value_{i+1}" for i in range(len(df[col].iloc[0]))]
                )
                df = pd.concat([df.drop(columns=[col]), list_data], axis=1)
    return df

def main():
    # --- Step 0: Set up constants --- #
    working_commodity = "WTI CRUDE"
    num_bootstrap_samples = 10  # number of bootstrap samples
    
    # --- Step 1: Load commodity prices and compute returns --- #
    commodity_px = bsp.close_price_extraction(working_commodity)
    commodity_returns = bsp.prices_to_returns(commodity_px).dropna()
    
    # Generate bootstrap samples of returns (the samples retain their original ordering)
    commodity_returns_samples = bsp.generate_stationary_bootstrap_sample(commodity_returns, num_bootstrap_samples)
    
    # --- Step 2: Load pre-defined features (stages) --- #
    # Use the CSV file as-is (it already contains the extracted features)
    csv_file = "combined_metrics_lists.csv"
    stages = {
        "Stage 1": extract_stage_1(csv_file),
        "Stage 2": extract_stage_2(csv_file),
        "Stage 3": extract_stage_3(csv_file),
        "Stage 4": extract_stage_4(csv_file)
    }
    
    # Optionally process each stage to expand any object-type columns
    for stage_name in stages:
        stages[stage_name] = handle_object_columns(stages[stage_name])
    
    # --- Step 3: Set up the classifier and time series splitter --- #
    svm_params = {"kernel": "linear", "C": 0.01, "random_state": 42}
    tss = TimeSeriesSplit(n_splits=25)
    
    # --- Step 4: For each bootstrap sample and each feature stage, train and evaluate --- #
    results = []  # will store tuples: (bootstrap_sample, stage, mean_score)
    
    for sample_name, sample_returns in commodity_returns_samples.items():
        print(f"\nUsing bootstrap sample: {sample_name}")
        # Compute target labels as the sign of the bootstrap sample returns
        y = np.sign(sample_returns)
        # Reset index on y so it becomes a default RangeIndex
        y = y.reset_index(drop=True)
        
        for stage_name, stage_df in stages.items():
            print(f"\n--- Training on {stage_name} for sample {sample_name} ---")
            # If the CSV has a 'window_start' column, drop it
            if "window_start" in stage_df.columns:
                X = stage_df.drop(columns=["window_start"])
            else:
                X = stage_df.copy()
            
            # Reset the index on X so it aligns by order with y
            X = X.reset_index(drop=True)
            
            # Ensure X and y have the same number of rows (if necessary, truncate to the minimum length)
            n = min(len(X), len(y))
            X_aligned = X.iloc[:n]
            y_aligned = y.iloc[:n]
            
            clf = make_pipeline(StandardScaler(), SVC(**svm_params))
            fold_scores = []
            
            # Apply time series split
            for fold, (train_idx, test_idx) in enumerate(tss.split(X_aligned)):
                print(f"Training fold {fold + 1} for {stage_name} on sample {sample_name}...")
                X_train, X_test = X_aligned.iloc[train_idx], X_aligned.iloc[test_idx]
                y_train, y_test = y_aligned.iloc[train_idx], y_aligned.iloc[test_idx]
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores)
            results.append((sample_name, stage_name, mean_score))
            print(f"Bootstrap sample: {sample_name}, {stage_name} Mean Score: {mean_score:.4f}")
    
    # --- Step 5: Show final results --- #
    results_df = pd.DataFrame(results, columns=["Bootstrap Sample", "Feature Stage", "Mean Score"])
    print("\nFinal Results:")
    print(results_df)
    
if __name__ == '__main__':
    main()
