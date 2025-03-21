import models.old_models.bootstrapPrices as bsp
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import ast
from featuresFromCSV import extract_stage_1, extract_stage_2, extract_stage_3, extract_stage_4

# Function to process any object-type columns
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
    working_commodity = "BRENT CRUDE"
    num_bootstrap_samples = 10  # number of bootstrap samples
    
    # --- Step 1: Load commodity prices and compute returns --- #
    commodity_px = bsp.close_price_extraction(working_commodity)
    commodity_returns = bsp.prices_to_returns(commodity_px).dropna()
    
    # Generate bootstrap samples of returns (the samples retain their original ordering)
    commodity_returns_samples = bsp.generate_stationary_bootstrap_sample(commodity_returns, num_bootstrap_samples)
    
    # --- Step 2: Load pre-defined features (stages) --- #
    csv_file = "combined_metrics_lists.csv"
    stages = {
        "Stage 1": extract_stage_1(csv_file),
        "Stage 2": extract_stage_2(csv_file),
        "Stage 3": extract_stage_3(csv_file),
        "Stage 4": extract_stage_4(csv_file)
    }
    
    # Process each stage to expand any object-type columns
    for stage_name in stages:
        stages[stage_name] = handle_object_columns(stages[stage_name])
    
    # --- Step 3: Set up the classifiers and time series splitter --- #
    models = {
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.01, random_state=42)),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma='scale', random_state=42)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    }
    
    tss = TimeSeriesSplit(n_splits=25)
    
    # --- Step 4: For each bootstrap sample, each feature stage, and each model, train and evaluate --- #
    results = []  # will store tuples: (bootstrap_sample, stage, model_name, mean_score)
    
    for sample_name, sample_returns in commodity_returns_samples.items():
        print(f"\nUsing bootstrap sample: {sample_name}")
        # Compute target labels as the sign of the bootstrap sample returns
        y = np.sign(sample_returns)
        # Reset index on y
        y = y.reset_index(drop=True)
        
        for stage_name, stage_df in stages.items():
            print(f"\n--- Training on {stage_name} for sample {sample_name} ---")
            # Drop window_start column if it exists
            if "window_start" in stage_df.columns:
                X = stage_df.drop(columns=["window_start"])
            else:
                X = stage_df.copy()
            
            # Reset the index on X
            X = X.reset_index(drop=True)
            
            # Ensure X and y have the same number of rows
            n = min(len(X), len(y))
            X_aligned = X.iloc[:n]
            y_aligned = y.iloc[:n]
            
            for model_name, clf in models.items():
                print(f"Testing {model_name} on {stage_name} for sample {sample_name}...")
                fold_scores = []
                
                # Apply time series split
                for fold, (train_idx, test_idx) in enumerate(tss.split(X_aligned)):
                    X_train, X_test = X_aligned.iloc[train_idx], X_aligned.iloc[test_idx]
                    y_train, y_test = y_aligned.iloc[train_idx], y_aligned.iloc[test_idx]
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                    fold_scores.append(score)
                
                mean_score = np.mean(fold_scores)
                results.append((sample_name, stage_name, model_name, mean_score))
                print(f"Bootstrap sample: {sample_name}, {stage_name}, {model_name} Mean Score: {mean_score:.4f}")
    
    # --- Step 5: Show final results and create visualizations --- #
    results_df = pd.DataFrame(results, columns=["Bootstrap Sample", "Feature Stage", "Model", "Mean Score"])
    print("\nFinal Results:")
    print(results_df)
    
    # Calculate average scores across bootstrap samples for each model and stage
    avg_results = results_df.groupby(["Feature Stage", "Model"])["Mean Score"].mean().reset_index()
    
    # Create a pivot table for easier visualization
    pivot_results = avg_results.pivot(index="Feature Stage", columns="Model", values="Mean Score")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    pivot_results.plot(kind='bar')
    plt.title('Average Model Performance by Feature Stage')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Feature Stage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    avg_results.to_csv('average_model_comparison.csv', index=False)
    
    print("\nResults saved to CSVs and visualization saved as 'model_comparison.png'")
    
if __name__ == '__main__':
    main() 