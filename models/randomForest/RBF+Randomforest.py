import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import ast
from featuresFromCSV import (
    extract_stage_1, extract_stage_2, extract_stage_3, extract_stage_4
)

def handle_object_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            if isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
                list_data = pd.DataFrame(df[col].to_list(), columns=[f"{col}_value_{i+1}" for i in range(len(df[col].iloc[0]))])
                df = pd.concat([df.drop(columns=[col]), list_data], axis=1)
    return df

def close_price_extraction(working_commodity):
    df = pd.read_csv("commodity_futures.csv", header=0, index_col=0, thousands=",")
    df.index = pd.to_datetime(df.index, format='mixed', dayfirst=True)
    df = df.sort_index()
    commodity_px = df[working_commodity]
    commodity_px = commodity_px.dropna()
    return commodity_px

def prices_to_returns(prices):
    return np.log(prices / prices.shift(1))

def train_and_evaluate_model(X, y, model, tss, stage_name, model_name):
    print(f"\n--- Training {model_name} on {stage_name} ---")
    
    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        print(f"Training fold {fold + 1}...")
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average="weighted", zero_division=1))
        recall_scores.append(recall_score(y_test, y_pred, average="weighted", zero_division=1))

    return {
        'stage': stage_name,
        'model': model_name,
        'mean_f1': np.mean(f1_scores),
        'max_f1': np.max(f1_scores),
        'mean_accuracy': np.mean(accuracy_scores),
        'mean_precision': np.mean(precision_scores),
        'max_precision': np.max(precision_scores),
        'mean_recall': np.mean(recall_scores)
    }

def main():
    # --- Step 1: Load Data ---
    working_commodity = "BRENT CRUDE"
    csv_file = "combined_metrics_lists.csv"

    # Extract features for each stage
    stages = {
        "Stage 1": extract_stage_1(csv_file),
        "Stage 2": extract_stage_2(csv_file),
        "Stage 3": extract_stage_3(csv_file),
        "Stage 4": extract_stage_4(csv_file)
    }

    # --- Step 2: Extract Prices and Returns ---
    commodity_px = close_price_extraction(working_commodity)
    commodity_returns = prices_to_returns(commodity_px)
    commodity_returns = commodity_returns.dropna()

    # --- Step 3: Define Models ---
    random_state = 42
    tss = TimeSeriesSplit(n_splits=25)

    models = {
        "RBF SVM": make_pipeline(StandardScaler(), 
            SVC(kernel="rbf", C=0.1, gamma=0.1, random_state=random_state)),
        "Random Forest": make_pipeline(StandardScaler(),
            RandomForestClassifier(max_depth=5, n_estimators=20, max_features=100, random_state=random_state))
    }

    # --- Step 4: Train and Evaluate ---
    results = []

    for stage_name, df in stages.items():
        df = handle_object_columns(df)
        X = df.drop(columns=["window_start"])
        y = np.sign(commodity_returns)
        y = y.dropna()

        for model_name, model in models.items():
            result = train_and_evaluate_model(X, y, model, tss, stage_name, model_name)
            results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results grouped by model and stage
    print("\nResults grouped by model:")
    for model_name in models.keys():
        print(f"\n{model_name} Results:")
        model_results = results_df[results_df['model'] == model_name]
        print(model_results[['stage', 'mean_f1', 'max_f1', 'mean_accuracy', 'mean_precision', 'max_precision']].to_string(index=False))

    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to model_comparison_results.csv")

if __name__ == '__main__':
    main() 

#RBF SVM Results:
  #stage  mean_f1   max_f1  mean_accuracy  mean_precision  max_precision
#Stage 1 0.370240 0.520653       0.515478        0.693732       0.755463
#Stage 2 0.374655 0.488399       0.516174        0.668661       0.755463
#Stage 3 0.353814 0.434187       0.517217        0.752271       0.765898
#Stage 4 0.353814 0.434187       0.517217        0.752271       0.765898


#Random Forest Results:
  #stage  mean_f1   max_f1  mean_accuracy  mean_precision  max_precision
#Stage 1 0.439385 0.545301       0.506783        0.522578       0.751092
#Stage 2 0.512280 0.610400       0.542609        0.555792       0.751661
#Stage 3 0.479061 0.548476       0.509913        0.519536       0.750473
#Stage 4 0.479384 0.551471       0.511130        0.515668       0.618904