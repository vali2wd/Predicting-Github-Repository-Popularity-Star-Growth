import polars as pl
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# --- Config ---
VAL_PATH = 'val_dataset.parquet'
MODEL_PATH = 'lgbm_github_model.pkl'
SPECIFIC_REPO = 'tensorflow/models'

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please train the model first.")
    
    print("Loading model and validation data...")
    model = joblib.load(MODEL_PATH)
    lf = pl.scan_parquet(VAL_PATH)
    return model, lf

def get_repo_data(lf, repo_name=None):
    if repo_name:
        print(f"Fetching data for: {repo_name}")
        df = lf.filter(pl.col('repo_name') == repo_name).collect()
        if df.height == 0:
            raise ValueError(f"Repository '{repo_name}' not found in validation set.")
    else:
        print("Picking a random repository...")
        unique_repos = lf.select('repo_name').unique().collect()['repo_name'].to_list()
        repo_name = random.choice(unique_repos)
        print(f"Selected: {repo_name}")
        df = lf.filter(pl.col('repo_name') == repo_name).collect()
    
    return df.sort('day'), repo_name

# --- FIXED FUNCTION ---
def prepare_features(df, model):
    """
    Selects exactly the features the model was trained on.
    """
    # 1. Get the list of features the model expects
    required_features = model.feature_name_
    
    # 2. Verify all features exist in the dataframe
    missing_cols = [c for c in required_features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The following features are missing from the dataset: {missing_cols}")
        
    # 3. Select only those columns in the correct order
    X = df.select(required_features).to_pandas()
    
    return X, required_features

def plot_results(dates, actual_total, pred_total, actual_diff, pred_diff, repo_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Cumulative Total
    ax1.plot(dates, actual_total, label='Actual Total (Scaled)', color='black', linewidth=2)
    ax1.plot(dates, pred_total, label='Predicted Total (Reconstructed)', color='#00ff41', linestyle='--')
    ax1.set_title(f"Repository Growth: {repo_name}", fontsize=14)
    ax1.set_ylabel("Total Stars (Scaled Log1p)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily Velocity
    ax2.bar(dates, actual_diff, label='Actual Daily Change', color='gray', alpha=0.5, width=1.0)
    ax2.plot(dates, pred_diff, label='Predicted Velocity', color='red', linewidth=1.5)
    ax2.set_title("Daily Growth Velocity (Model Sensitivity)", fontsize=12)
    ax2.set_ylabel("Change in Scaled Score")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Load
    model, val_lf = load_resources()
    
    # 2. Get Data
    df, repo_name = get_repo_data(val_lf, SPECIFIC_REPO)
    
    # 3. Prepare Features (Passing 'model' to get correct columns)
    X, feature_names = prepare_features(df, model)
    
    # 4. Predict
    print(f"Running inference on {len(feature_names)} features...")
    pred_diff = model.predict(X)
    
    # 5. Reconstruct
    prev_total = df['total_stars_lag_1d_scaled'].to_numpy()
    pred_total = prev_total + pred_diff
    
    actual_total = df['total_stars_scaled'].to_numpy()
    actual_diff = actual_total - prev_total
    
    # 6. Visualize
    dates = df['day'].to_list()
    plot_results(dates, actual_total, pred_total, actual_diff, pred_diff, repo_name)