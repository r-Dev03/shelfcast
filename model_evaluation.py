import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load preprocessed data
preprocessed_df = pd.read_csv('preprocessed_data.csv')
X = preprocessed_df.drop('sales', axis=1)
y = preprocessed_df['sales']

split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Load trained models
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

# Make predictions
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

print("=== Random Forest Evaluation ===")
print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.4f}")

# Evaluate Gradient Boosting
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

print("\n=== Gradient Boosting Evaluation ===")
print(f"Mean Absolute Error (MAE): {gb_mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {gb_rmse:.4f}")

print("\n=== Model Comparison ===")
if rf_mae < gb_mae:
    print("✓ Random Forest performs better (lower MAE)")
    best_model = rf_model
    best_name = "Random Forest"
else:
    print("✓ Gradient Boosting performs better (lower MAE)")
    best_model = gb_model
    best_name = "Gradient Boosting"

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_test.values[:100], label='Actual', alpha=0.7)
plt.plot(rf_predictions[:100], label='RF Predictions', alpha=0.7)
plt.xlabel('Time Period')
plt.ylabel('Sales')
plt.title('Random Forest: Actual vs Predicted (First 100 samples)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_test.values[:100], label='Actual', alpha=0.7)
plt.plot(gb_predictions[:100], label='GB Predictions', alpha=0.7)
plt.xlabel('Time Period')
plt.ylabel('Sales')
plt.title('Gradient Boosting: Actual vs Predicted (First 100 samples)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print("\n✓ Comparison plot saved to: model_comparison.png")

results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'MAE': [rf_mae, gb_mae],
    'RMSE': [rf_rmse, gb_rmse]
})
results_df.to_csv('evaluation_results.csv', index=False)
print("✓ Evaluation results saved to: evaluation_results.csv")

