# Shelfcast

**Daily sales forecasting for 500 store-item pairs with Random Forest and gradient boosting.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)

## Results

Trained on January 2013 through the first days of 2017, tested on the rest of 2017 (181,200 predictions):

| Model | MAE | RMSE |
|-------|-----|------|
| Naive baseline (same day last week) | 8.85 | 11.61 |
| Random Forest | 6.53 | 8.51 |
| **Gradient Boosting** | **6.43** | **8.37** |

Gradient boosting cuts the error by **27% relative to the naive baseline**.

## Highlights

- Forecasts daily sales for 10 stores × 50 items from five years of Kaggle sales history
- Engineers lag and rolling-average features from each store-item's past sales, without letting the target leak in
- Compares Random Forest and gradient boosting against a naive baseline, with GridSearchCV tuning and time-series cross-validation

## How It Works

### Features
For every store-item pair, each day gets:
- **Calendar:** year, month, ISO week, day of week
- **Lags:** sales 1, 2, 4, 7, and 14 days earlier
- **Rolling averages:** mean sales over the previous 4 and 7 days

The rolling averages are shifted by one day, so they only cover days *before* the one being predicted:

```python
df['sales_rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(
    lambda x: x.shift(1).rolling(window=7).mean()
)
```

Without the shift, the window includes the current day's sales, which is the value being predicted.

### Splitting by time
Features are built within each store-item series in date order, then the rows are re-sorted by date. That way, every split (the 80/20 holdout, `TimeSeriesSplit` in tuning, and cross-validation) trains on earlier days and tests on later ones.

### Why the baseline matters
Sales here follow a strong weekly cycle, so "same as last week" is already a reasonable guess. Measuring against it shows how much the models actually add beyond that cycle.

## Getting Started

1. Download `train.csv` from the [Store Item Demand Forecasting dataset](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) (a free Kaggle account is required) and place it in the project root.
2. Install the dependencies and run the pipeline in order:

```bash
git clone https://github.com/r-Dev03/shelfcast.git
cd shelfcast
pip install pandas numpy scikit-learn matplotlib

python preprocessing.py          # builds features → preprocessed_data.csv
python model_training.py         # trains both models → rf_model.pkl, gb_model.pkl
python model_evaluation.py       # baseline vs models → evaluation_results.csv, model_comparison.png
python hyperparameter_tuning.py  # GridSearchCV → best_rf_model.pkl, best_gb_model.pkl, hyperparameter_tuning_results.csv
python cross_validation.py       # 5-fold time-series CV of the tuned Random Forest → cv_results.csv, cross_validation_results.png
```

Or, with Nix: `nix develop`, then run the same scripts.

Tuning and cross-validation retrain the models many times on ~900,000 rows, so expect them to take a while.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib

## Known Limitations

- **One-day-ahead only.** The lag features assume yesterday's sales are known, so the models can't forecast several days out without feeding their own predictions back in.
- **Single holdout period.** The headline numbers come from one train/test split (nearly all of 2017). Cross-validation shows how much they vary across periods.
- **No external signals.** Holidays, promotions, and prices aren't in the dataset, so spikes driven by them can't be predicted.
