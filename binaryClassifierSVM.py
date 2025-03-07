import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from featureExtractionTDA import features_df

# Create the binary target: 1 if raw_return is positive, 0 otherwise.
features_df['target'] = (features_df['raw_return'] > 0).astype(int)

# Drop columns that are not used as features.
X = features_df.drop(columns=['window_mid_date', 'raw_return', 'target'])
y = features_df['target']

# Split the data preserving temporal order (no shuffling).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Build a pipeline: scale features then train an SVM with probability estimates and class weight options.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, class_weight='balanced'))
])

# Define an expanded parameter grid for tuning.
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'svm__kernel': ['rbf', 'linear'],
    'svm__class_weight': [None, 'balanced']
}

# Use TimeSeriesSplit for CV.
tscv = TimeSeriesSplit(n_splits=5)

# Set up GridSearchCV.
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))

# Evaluate on the test set.
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test set accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
