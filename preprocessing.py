import pandas as pd
import numpy as np
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
df['sales_lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
df['sales_lag_14'] = df.groupby(['store', 'item'])['sales'].shift(14)

# Rolling averages of the PREVIOUS days only. shift(1) excludes the current
# day, whose sales are the value being predicted.
df['sales_rolling_mean_4'] = df.groupby(['store', 'item'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=4).mean()
)
df['sales_rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7).mean()
)

# Drop rows without full history, then order by date so every train/test
# split below trains on the past and tests on the future
df = df.dropna().sort_values(['date', 'store', 'item']).reset_index(drop=True)

df['store'] = df['store'].astype('category').cat.codes
df['item'] = df['item'].astype('category').cat.codes

feature_cols = ['store', 'item', 'year', 'month', 'week', 'day_of_week',
                'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_7', 'sales_lag_14',
                'sales_rolling_mean_4', 'sales_rolling_mean_7']
X = df[feature_cols].copy()
y = df['sales'].copy()

preprocessed_df = X.copy()
preprocessed_df['sales'] = y.values
preprocessed_df.to_csv('preprocessed_data.csv', index=False)

print("\n✓ Preprocessing complete!")
print(f"Preprocessed data shape: {preprocessed_df.shape}")
print(f"Features: {feature_cols}")
print("\nPreprocessed data saved to: preprocessed_data.csv")

