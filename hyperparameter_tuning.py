import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
preprocessed_df = pd.read_csv('preprocessed_data.csv')
X = preprocessed_df.drop('sales', axis=1)
y = preprocessed_df['sales']

# Time-series split
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print("=== Hyperparameter Tuning with GridSearchCV ===\n")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

# GridSearchCV with time series aware CV
tscv = TimeSeriesSplit(n_splits=3)

print("Tuning Random Forest...")
rf_grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
rf_grid_search.fit(X_train, y_train)

print(f"\nBest RF Parameters: {rf_grid_search.best_params_}")
print(f"Best RF CV Score (MAE): {-rf_grid_search.best_score_:.4f}")

# Define parameter grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 57]
}

print("\n\nTuning Gradient Boosting...")
gb_grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
gb_grid_search.fit(X_train, y_train)

print(f"\nBest GB Parameters: {gb_grid_search.best_params_}")
print(f"Best GB CV Score (MAE): {-gb_grid_search.best_score_:.4f}")

best_rf = rf_grid_search.best_estimator_
best_gb = gb_grid_search.best_estimator_

rf_pred = best_rf.predict(X_test)
gb_pred = best_gb.predict(X_test)

rf_test_mae = mean_absolute_error(y_test, rf_pred)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

gb_test_mae = mean_absolute_error(y_test, gb_pred)
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print("\n=== Test Set Results (Tuned Models) ===")
print(f"\nRandom Forest:")
print(f"  MAE:  {rf_test_mae:.4f}")
print(f"  RMSE: {rf_test_rmse:.4f}")

print(f"\nGradient Boosting:")
print(f"  MAE:  {gb_test_mae:.4f}")
print(f"  RMSE: {gb_test_rmse:.4f}")

with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
with open('best_gb_model.pkl', 'wb') as f:
    pickle.dump(best_gb, f)

print("\n✓ Best models saved: best_rf_model.pkl, best_gb_model.pkl")

tuning_results = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'Gradient Boosting (Tuned)'],
    'Best_Params': [str(rf_grid_search.best_params_), str(gb_grid_search.best_params_)],
    'CV_MAE': [-rf_grid_search.best_score_, -gb_grid_search.best_score_],
    'Test_MAE': [rf_test_mae, gb_test_mae],
    'Test_RMSE': [rf_test_rmse, gb_test_rmse]
})
tuning_results.to_csv('hyperparameter_tuning_results.csv', index=False)
print("✓ Tuning results saved to: hyperparameter_tuning_results.csv")

