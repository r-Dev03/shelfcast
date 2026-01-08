## Requirements

### Software Requirements

- Python 3.8 or higher  
- pandas  
- numpy  
- scikit‑learn  
- matplotlib  

These libraries are used for data loading, preprocessing, model training, evaluation, cross‑validation, and plotting results for the AI/ML product.

### Hardware Requirements

- Minimum 4 GB RAM (8+ GB recommended for faster model training and tuning)  
- Multi‑core CPU (e.g., Intel i5/Ryzen 5 or better)  
- A few hundred MB of free disk space for the dataset, intermediate CSV files, and output figures  

This hardware is sufficient to load the dataset, run the preprocessing steps, train the models, and perform cross‑validation and hyperparameter tuning. 


## Instructions to Run the AI/ML Application

1. **Clone the repository**

   ```bash
   git clone https://gitlab.com/wgu-gitlab-environment/student-repos/rhussa11/d683-advanced-ai-and-ml.git
   cd d683-advanced-ai-and-ml
   ```

2. **Set up the Python environment**

   Create and activate a virtual environment (or use your existing one), then install the required packages:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install pandas numpy scikit-learn matplotlib
   ```

   If you are using a Nix flake or another environment manager, make sure the versions include these libraries.

3. **Download and place the dataset**

   - Download the public retail demand dataset, for example, the [Store Item Demand Forecasting dataset from Kaggle](https://www.kaggle.com/c/demand-forecasting-kernels-only/data).
   - Save the file as `train.csv` in the root of this project (same folder as `preprocessing.py`).  
   - Ensure the file contains at least the columns: `date`, `store`, `item`, and `sales`. 

4. **Run data preprocessing (Task B1)**

   ```bash
   python preprocessing.py
   ```

   This loads `train.csv`, cleans and preprocesses the data, creates time‑based and lag features, scales the inputs, and saves the result to `preprocessed_data.csv`. 

5. **Train baseline models (Tasks B2–B3)**

   ```bash
   python model_training.py
   ```

   This script reads `preprocessed_data.csv`, performs a time‑aware train/test split, and trains the baseline models (Random Forest and Gradient Boosting). The trained models are saved locally as `.pkl` files. 

6. **Evaluate models (Task B4)**

   ```bash
   python model_evaluation.py
   ```

   This evaluates the trained models on the test set using metrics such as MAE and RMSE and generates a comparison plot of actual vs predicted values. It writes the metrics to `evaluation_results.csv` and saves the figure as `model_comparison.png`. 

7. **Run cross‑validation (Task B5)**

   ```bash
   python cross_validation.py
   ```

   This performs time series cross‑validation using `TimeSeriesSplit`, reports MAE and RMSE for each fold, and saves both the numerical results (`cv_results.csv`) and a visualization (`cross_validation_results.png`). 

8. **Run hyperparameter tuning (Task B6)**

   ```bash
   python hyperparameter_tuning.py
   ```

   This runs a reduced‑size grid search with time‑series‑aware cross‑validation to find better hyperparameters for the chosen model(s). It reports the best parameters, evaluates the tuned model on the test set, and saves a summary to `hyperparameter_tuning_results.csv`. 

9. **Review outputs**

   After all scripts have run successfully, you can inspect:

   - `preprocessed_data.csv` for the engineered features  
   - `evaluation_results.csv` and `model_comparison.png` for baseline performance  
   - `cv_results.csv` and `cross_validation_results.png` for cross‑validation behavior  
   - `hyperparameter_tuning_results.csv` for the tuned model settings and metrics  

