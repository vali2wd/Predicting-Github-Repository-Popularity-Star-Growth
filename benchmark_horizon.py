import polars as pl
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.metrics import mean_absolute_error

# --- Config ---
VAL_PATH = 'val_dataset.parquet'
MODEL_PATH = 'lgbm_github_model.pkl'

# Choose a repo with enough history (e.g., 'facebook/react', 'pandas-dev/pandas')
# or set to None for random
BENCHMARK_REPO = 'tensorflow/models' 
HORIZON_DAYS = 90  # How far into the future to test

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")
    
    print("Loading resources...")
    model = joblib.load(MODEL_PATH)
    lf = pl.scan_parquet(VAL_PATH)
    return model, lf

def get_long_history_repo(lf, min_days=100):
    """Finds a repo with enough validation data to run the benchmark"""
    if BENCHMARK_REPO:
        return BENCHMARK_REPO
    
    print("Finding a suitable repository...")
    # Group by repo and count rows to ensure we have enough data
    counts = lf.group_by('repo_name').len().filter(pl.col('len') > min_days).collect()
    
    if counts.height == 0:
        raise ValueError("No repositories in validation set have enough days for this horizon.")
        
    candidates = counts['repo_name'].to_list()
    return random.choice(candidates)

def prepare_features(df, model):
    required_features = model.feature_name_
    X = df.select(required_features).to_pandas()
    return X

def run_benchmark():
    model, lf = load_resources()
    repo_name = get_long_history_repo(lf, min_days=HORIZON_DAYS + 10)
    print(f"Benchmarking Horizon on: {repo_name}")
    
    # 1. Get Data
    df = (lf.filter(pl.col('repo_name') == repo_name)
            .sort('day')
            .collect())
    
    # We take the first 'HORIZON_DAYS' of the validation set to simulate "The Future"
    # (Or you can take the whole set if you want to see the full timeline)
    df_future = df.head(HORIZON_DAYS)
    
    # 2. Prepare Features & Predict Velocity
    X = prepare_features(df_future, model)
    predicted_velocities = model.predict(X)
    
    # 3. Calculate "Drifting" Forecast (Multi-Step)
    # We start with the ACTUAL total on Day 0 (The "Anchor")
    anchor_total = df_future['total_stars_scaled'][0]
    
    # Cumulative Sum of predicted changes
    # Forecast_t = Anchor + Sum(Velocities_0_to_t)
    cumulative_growth = np.cumsum(predicted_velocities)
    
    # We adjust the cumulative sum so the first prediction adds to the anchor
    # (Because predicted_velocities[0] is the change from Day -1 to Day 0)
    # Actually, to align strictly:
    # Pred_Day_0 = Lag_Day_0 + Vel_Day_0. 
    # But for a horizon test starting at t=0, we usually assume t=0 is known.
    # Let's assume we know Day 0, and we predict Day 1 onwards.
    
    actual_totals = df_future['total_stars_scaled'].to_numpy()
    
    # Initialize recursive predictions array
    recursive_preds = np.zeros(len(actual_totals))
    recursive_preds[0] = actual_totals[0] # Day 0 is ground truth
    
    # From Day 1 onwards: Pred_t = Pred_{t-1} + Velocity_t
    for t in range(1, len(actual_totals)):
        recursive_preds[t] = recursive_preds[t-1] + predicted_velocities[t]
        
    # 4. Measure Errors at Key Intervals
    days = [7, 30, 60, 90]
    print("\n--- Horizon Accuracy (Cumulative Error) ---")
    for d in days:
        if d < len(recursive_preds):
            err = recursive_preds[d] - actual_totals[d]
            print(f"Day {d}: Drift = {err:.4f} (Scaled Units)")

    # 5. Plot
    dates = df_future['day'].to_list()
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_totals, label='Actual History', color='black', linewidth=2)
    plt.plot(dates, recursive_preds, label='Recursive Forecast (Drifting)', color='red', linestyle='--')
    
    plt.title(f"Forecast Horizon Stability: {repo_name} ({HORIZON_DAYS} Days)")
    plt.ylabel("Total Stars (Scaled)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add error fill
    plt.fill_between(dates, actual_totals, recursive_preds, color='red', alpha=0.1, label='Accumulated Error')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()