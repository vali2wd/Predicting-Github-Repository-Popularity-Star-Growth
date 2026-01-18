import polars as pl
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.metrics import mean_absolute_error

# --- Config ---
VAL_PATH = 'val_dataset_linear.parquet'
MODEL_PATH = 'lgbm_github_linear.pkl'

# Choose a repo with enough history (e.g., 'facebook/react', 'pandas-dev/pandas')
# or set to None for random selection
BENCHMARK_REPO = None  # Set to None for random
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
    available_features = df.columns
    
    print(f"Model expects {len(required_features)} features")
    print(f"DataFrame has {len(available_features)} columns")
    
    # Check for missing features
    missing_features = [f for f in required_features if f not in available_features]
    
    if missing_features:
        print(f"\n❌ ERROR: {len(missing_features)} features are missing from validation data!")
        print(f"First 10 missing: {missing_features[:10]}")
        print(f"\nAvailable columns sample: {available_features[:10]}")
        raise ValueError(f"Missing features in validation dataset: {len(missing_features)} features")
    
    X = df.select(required_features).to_pandas()
    
    print(f"Selected features shape: {X.shape}")
    
    if X.empty or X.shape[0] == 0:
        raise ValueError("Feature dataframe is empty after selection")
    
    return X

def run_benchmark():
    model, lf = load_resources()
    repo_name = get_long_history_repo(lf, min_days=HORIZON_DAYS + 10)
    print(f"Benchmarking Horizon on: {repo_name}")
    
    # 1. Get Data
    df = (lf.filter(pl.col('repo_name') == repo_name)
            .sort('day')
            .collect())
    
    # Take first HORIZON_DAYS rows for testing
    df_future = df.head(min(HORIZON_DAYS, len(df)))
    
    # 2. CRITICAL: Identify which features are "future-dependent"
    feature_names = model.feature_name_
    
    future_dependent = [f for f in feature_names if 
                       ('daily_change' in f or 
                        (f.startswith('total_') and 'lag' not in f and 'rolling' not in f))]
    
    print(f"\n⚠️ WARNING: {len(future_dependent)} features require future data:")
    print(f"Examples: {future_dependent[:5]}")
    print(f"\nThis benchmark tests 'given future activity, predict stars'")
    print(f"NOT 'predict stars without knowing future activity'")
    
    # 3. Proceed with current (optimistic) test
    print(f"\nProceeding with optimistic test (using future activity data)...")
    X = prepare_features(df_future, model)
    predicted_velocities = model.predict(X)
    
    # 3. Recursive Forecast in Linear Space
    # Determine which columns exist
    if 'total_stars' in df_future.columns:
        actual_stars = df_future['total_stars'].to_numpy()
    elif 'total_stars_scaled' in df_future.columns:
        actual_stars = df_future['total_stars_scaled'].to_numpy()
    else:
        raise ValueError("No total_stars column found in dataset")
    
    # Initialize recursive predictions
    recursive_preds = np.zeros(len(actual_stars))
    recursive_preds[0] = actual_stars[0]  # Day 0 is ground truth
    
    # From Day 1 onwards: Pred_t = Pred_{t-1} + Velocity_t
    for t in range(1, len(actual_stars)):
        recursive_preds[t] = recursive_preds[t-1] + predicted_velocities[t]
        
    # 4. Measure Errors at Key Intervals
    days = [7, 30, 60, 90]
    print("\n--- Horizon Accuracy (Linear Space) ---")
    for d in days:
        if d < len(recursive_preds):
            err = recursive_preds[d] - actual_stars[d]
            pct_err = (err / actual_stars[d]) * 100 if actual_stars[d] != 0 else 0
            print(f"Day {d:2d}: Drift = {err:+.2f} stars ({pct_err:+.2f}%)")

    # 5. Plot (Single Scale)
    dates = df_future['day'].to_list()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot: Original Scale
    ax.plot(dates, actual_stars, label='Actual History', color='black', linewidth=2)
    ax.plot(dates, recursive_preds, label='Recursive Forecast', color='red', linestyle='--', linewidth=2)
    ax.fill_between(dates, actual_stars, recursive_preds, 
                     color='red', alpha=0.15)
    ax.axhline(y=actual_stars[0], color='gray', linestyle=':', alpha=0.5, label='Baseline (Day 0)')
    ax.set_title(f"Forecast Horizon Stability: {repo_name} ({len(df_future)} Days) - Linear Scale")
    ax.set_ylabel("Total Stars")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with commas for readability
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Calculate drift
    final_drift_pct = abs((recursive_preds[-1] - actual_stars[-1]) / actual_stars[-1]) * 100
    final_star_diff = abs(recursive_preds[-1] - actual_stars[-1])
    
    plt.tight_layout()
    plt.savefig('horizon_benchmark_linear.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to 'horizon_benchmark_linear.png'")
    plt.show()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"⚠️ NOTE: This is an OPTIMISTIC test using future activity data")
    print(f"Repository: {repo_name}")
    print(f"Forecast Length: {len(df_future)} days")
    print(f"Actual Final Stars: {int(actual_stars[-1]):,}")
    print(f"Predicted Final Stars: {int(recursive_preds[-1]):,}")
    print(f"Star Difference: {int(final_star_diff):,} stars")
    print(f"Final Drift: {final_drift_pct:.2f}%")
    if final_drift_pct < 5:
        print("✅ PASS: Drift < 5% (Model is stable)")
    elif final_drift_pct < 10:
        print("⚠️ MARGINAL: Drift 5-10% (Acceptable for long horizons)")
    else:
        print("❌ FAIL: Drift > 10% (Model has bias accumulation)")


if __name__ == "__main__":
    run_benchmark()
