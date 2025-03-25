import pandas as pd

def extract_stage_1(csv_path):
    df = pd.read_csv(csv_path)
    stage1_columns = [
        "window_end", "mean", "median", "std", "var", "min", "max", "range"
    ]
    return df[stage1_columns]

def extract_stage_2(csv_path):
    df = pd.read_csv(csv_path)
    stage2_columns = [
        "window_end", "mean", "median", "std", "var", "min", "max", "range", 
        "trend_slope", "trend_intercept", "r_squared",
        "skew", "kurtosis", "dominant_freq", "spectral_entropy", "momentum"
    ]
    return df[stage2_columns]

def extract_stage_3(csv_path):
    df = pd.read_csv(csv_path)
    basic_columns = [
        "window_end", "mean", "median", "std", "var", "min", "max", "range",
        "trend_slope", "trend_intercept", "r_squared", "skew", "kurtosis",
        "dominant_freq", "spectral_entropy", "momentum"
    ]
    tda_columns = ["filtration_H0", "betti_H0", "filtration_H1", "betti_H1"]
    return df[basic_columns + tda_columns]

def extract_stage_4(csv_path):
    df = pd.read_csv(csv_path)
    basic_columns = [
        "window_end", "mean", "median", "std", "var", "min", "max", "range",
        "trend_slope", "trend_intercept", "r_squared", "skew", "kurtosis",
        "dominant_freq", "spectral_entropy", "momentum"
    ]
    tda_columns = ["filtration_H0", "betti_H0", "filtration_H1", "betti_H1"]
    entropy_columns = ["persistence_entropy_H1", "weighted_entropy_H1"]
    return df[basic_columns + tda_columns + entropy_columns]

if __name__ == '__main__':
    csv_file = "combined_metrics_lists.csv"
    
    # core basic features
    stage1_df = extract_stage_1(csv_file)
    print("Stage 1 (Core Basic Features):")
    print(stage1_df.head())
    
    # additional technical/trend features
    stage2_df = extract_stage_2(csv_file)
    print("\nStage 2 (Technical/Trend Features):")
    print(stage2_df.head())
    
    # basic features plus TDA Betti curves
    stage3_df = extract_stage_3(csv_file)
    print("\nStage 2 (Basic + TDA Betti Curves):")
    print(stage3_df.head())
    
    # all features (Basic + TDA Betti Curves + TDA Entropy)
    stage4_df = extract_stage_4(csv_file)
    print("\nStage 3 (All Features):")
    print(stage4_df.head())
