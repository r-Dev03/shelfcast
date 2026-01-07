import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load preprocessed data
preprocessed_df = pd.read_csv('preprocessed_data.csv')

X = preprocessed_df.drop('sales', axis=1)
y = preprocessed_df['sales']

split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Build and train Random Forest model
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("✓ Random Forest model trained!")

# Build and train Gradient Boosting model
print("\n--- Training Gradient Boosting Model ---")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train, y_train)
print("✓ Gradient Boosting model trained!")

# Save models
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

print("\n✓ Models saved to: rf_model.pkl, gb_model.pkl")

