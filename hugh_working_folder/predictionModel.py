import bootstrapPrices as bsp
import traditionalFeatureExtraction as bfe
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tsfresh.feature_extraction.feature_calculators import linear_trend_timewise, mean, kurtosis, skewness, set_property

def main():
    # --- Step 0: Set the working commodity and other constants --- #
    working_commodity = "BRENT CRUDE"
    working_days = 260

    # --- Step 1: Bootstrap the prices of a commodity --- #

    # Set the number of bootstrap samples
    num_bootstrap_samples = 10

    # Extract the prices and returns of the selected commodity
    commodity_px = bsp.close_price_extraction(working_commodity)
    commodity_returns = bsp.prices_to_returns(commodity_px)

    # Generate stationary bootstrap samples
    commodity_returns_samples = bsp.generate_stationary_bootstrap_sample(commodity_returns, num_bootstrap_samples)

    # Convert the returns samples back to prices
    commodity_initial_px = commodity_px.values[0]
    commodity_px_samples = bsp.returns_to_prices(commodity_returns_samples, commodity_initial_px)

    # # Plot the prices of the samples and the original commodity
    # ax = commodity_px_samples.plot(figsize=(20, 4))
    # ax = commodity_px.plot(figsize=(20, 4), ax=ax, linestyle="--", color='k', label=working_commodity)
    # ax.legend()
    # plt.show()

    # --- Step 2: Extraction of features & prediction --- #

    random_state = 42
    tss = TimeSeriesSplit(n_splits=10)

    names = [
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
    ]

    classifiers = [
        SVC(kernel="linear", C=0.001, random_state=random_state),
        SVC(gamma=0.1, C=0.1, random_state=random_state),
        RandomForestClassifier(
            max_depth=5, n_estimators=20, max_features=100, random_state=random_state
        ),
    ]

    @set_property("input", "pd.Series")
    def last_value(x):
        # return last value in the given Series
        return x.iloc[-1]
    
    experiment_A_results = []
    for c, s in commodity_returns_samples.items():
        print(f"Using sample time-series: {c}")
        df = bfe.create_tsfresh_flat_dataframe_monthly_target(commodity_returns, f'{working_commodity} returns')
        X, y = bfe.roll_and_extract(df, working_days)

        # Use same classifiers as above
        for name, clf in zip(names, classifiers):
            clf = make_pipeline(StandardScaler(), clf)
            for i, (train, test) in enumerate(tss.split(X)):
                print(f"Training {name} on fold {i+1}")
                X_train = X.iloc[train, :]
                y_train = y.iloc[train]
                clf.fit(X_train, y_train)
                X_test = X.iloc[test, :]
                y_test = y.iloc[test]
                score = clf.score(X_test, y_test)
                experiment_A_results.append((c, name, i+1, score))

    experiment_A_results_df = pd.DataFrame(experiment_A_results, columns=['sample_num', 'classifier', 'fold', 'score']).drop(axis=1, labels='fold')
    mean_fold_score = experiment_A_results_df.groupby(["sample_num", "classifier"]).mean()

    mean_scores = experiment_A_results_df.groupby(["sample_num", "classifier"])["score"].mean()
    std_scores = experiment_A_results_df.groupby(["sample_num", "classifier"])["score"].std()

    print("Mean Fold Score:")
    print(mean_fold_score)
    print("\nStandard Deviation of Scores:")
    print(std_scores)

if __name__ == '__main__':
    main()