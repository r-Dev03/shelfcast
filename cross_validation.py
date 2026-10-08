import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load preprocessed data
preprocessed_df = pd.read_csv('preprocessed_data.csv')
X = preprocessed_df.drop('sales', axis=1)
y = preprocessed_df['sales']

# Load the tuned Random Forest (run hyperparameter_tuning.py first)
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Time Series Cross-Validation (5 splits)
tscv = TimeSeriesSplit(n_splits=5)

cv_mae_scores = []
cv_rmse_scores = []
fold_number = 0

print("=== Time Series Cross-Validation Results ===\n")

for train_idx, test_idx in tscv.split(X):
    fold_number += 1
    
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]
    
    model.fit(X_train_fold, y_train_fold)
    
    y_pred = model.predict(X_test_fold)
    
    mae = mean_absolute_error(y_test_fold, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
    
    cv_mae_scores.append(mae)
    cv_rmse_scores.append(rmse)
    
    print(f"Fold {fold_number}:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

print("=== Cross-Validation Summary ===")
print(f"Mean MAE:  {np.mean(cv_mae_scores):.4f} (+/- {np.std(cv_mae_scores):.4f})")
print(f"Mean RMSE: {np.mean(cv_rmse_scores):.4f} (+/- {np.std(cv_rmse_scores):.4f})")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(cv_mae_scores) + 1), cv_mae_scores, marker='o', linewidth=2)
plt.axhline(y=np.mean(cv_mae_scores), color='r', linestyle='--', label='Mean MAE')
plt.xlabel('Fold')
plt.ylabel('MAE')
plt.title('Cross-Validation: MAE by Fold')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cv_rmse_scores) + 1), cv_rmse_scores, marker='o', linewidth=2)
plt.axhline(y=np.mean(cv_rmse_scores), color='r', linestyle='--', label='Mean RMSE')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Cross-Validation: RMSE by Fold')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=150)
print("\n✓ Cross-validation plot saved to: cross_validation_results.png")

cv_results_df = pd.DataFrame({
    'Fold': range(1, len(cv_mae_scores) + 1),
    'MAE': cv_mae_scores,
    'RMSE': cv_rmse_scores
})
cv_results_df.to_csv('cv_results.csv', index=False)
print("✓ Cross-validation results saved to: cv_results.csv")

