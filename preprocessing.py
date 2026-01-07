import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')  

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'])

df = df.sort_values('date').reset_index(drop=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week
df['day_of_week'] = df['date'].dt.dayofweek

df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
df['sales_lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
df['sales_lag_2'] = df.groupby(['store', 'item'])['sales'].shift(2)
df['sales_lag_4'] = df.groupby(['store', 'item'])['sales'].shift(4)

df['sales_rolling_mean_4'] = df.groupby(['store', 'item'])['sales'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)

df = df.dropna().reset_index(drop=True)

df['store'] = df['store'].astype('category').cat.codes
df['item'] = df['item'].astype('category').cat.codes

feature_cols = ['store', 'item', 'year', 'month', 'week', 'day_of_week', 
                'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_rolling_mean_4']
X = df[feature_cols].copy()
y = df['sales'].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

preprocessed_df = X_scaled.copy()
preprocessed_df['sales'] = y.values
preprocessed_df.to_csv('preprocessed_data.csv', index=False)

print("\n✓ Preprocessing complete!")
print(f"Preprocessed data shape: {preprocessed_df.shape}")
print(f"Features: {feature_cols}")
print("\nPreprocessed data saved to: preprocessed_data.csv")

