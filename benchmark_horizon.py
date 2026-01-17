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
MODEL_PATH = 'lgbm_github_log_only.pkl'

# Choose a repo with enough history (e.g., 'facebook/react', 'pandas-dev/pandas')
# or set to None for random selection
BENCHMARK_REPO = 'Alamofire/Alamofire'  # Set to None for random
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
        print(f"\nThis means val_dataset.parquet was created from the old dataset.")
        print(f"Solution: Re-run train_model.py to regenerate train/val splits from github_features_log1pd.parquet")
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
    
    # 2. Prepare Features & Predict Velocity
    X = prepare_features(df_future, model)
    predicted_velocities = model.predict(X)
    
    # 3. Recursive Forecast in Log1p Space
    # Start with the actual log1p value on Day 0
    actual_log1p = df_future['total_stars_log1p'].to_numpy()
    
    # Initialize recursive predictions
    recursive_preds = np.zeros(len(actual_log1p))
    recursive_preds[0] = actual_log1p[0]  # Day 0 is ground truth
    
    # From Day 1 onwards: Pred_t = Pred_{t-1} + Velocity_t
    for t in range(1, len(actual_log1p)):
        recursive_preds[t] = recursive_preds[t-1] + predicted_velocities[t]
        
    # 4. Measure Errors at Key Intervals
    days = [7, 30, 60, 90]
    print("\n--- Horizon Accuracy (Drift in Log1p Space) ---")
    for d in days:
        if d < len(recursive_preds):
            err = recursive_preds[d] - actual_log1p[d]
            pct_err = (err / actual_log1p[d]) * 100 if actual_log1p[d] != 0 else 0
            print(f"Day {d:2d}: Drift = {err:+.5f} ({pct_err:+.2f}%)")

    # 5. Convert to Original Scale for Presentation
    actual_stars = np.expm1(actual_log1p)
    pred_stars = np.expm1(recursive_preds)
    
    # 6. Plot (Two Subplots)
    dates = df_future['day'].to_list()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top Plot: Log1p Space (Technical)
    ax1.plot(dates, actual_log1p, label='Actual History', color='black', linewidth=2)
    ax1.plot(dates, recursive_preds, label='Recursive Forecast', color='red', linestyle='--', linewidth=2)
    ax1.fill_between(dates, actual_log1p, recursive_preds, 
                     color='red', alpha=0.15)
    ax1.axhline(y=actual_log1p[0], color='gray', linestyle=':', alpha=0.5, label='Baseline (Day 0)')
    ax1.set_title(f"Forecast Horizon Stability: {repo_name} ({len(df_future)} Days) - Log1p Space")
    ax1.set_ylabel("Total Stars (log1p)")
    ax1.set_xlabel("Date")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: Original Scale (Presentation)
    ax2.plot(dates, actual_stars, label='Actual History', color='black', linewidth=2)
    ax2.plot(dates, pred_stars, label='Recursive Forecast', color='red', linestyle='--', linewidth=2)
    ax2.fill_between(dates, actual_stars, pred_stars, 
                     color='red', alpha=0.15)
    ax2.axhline(y=actual_stars[0], color='gray', linestyle=':', alpha=0.5, label='Baseline (Day 0)')
    ax2.set_title(f"Forecast Horizon Stability - Original Scale (Star Counts)")
    ax2.set_ylabel("Total Stars")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis with commas for readability
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Calculate drift on original scale
    final_drift_pct = abs((recursive_preds[-1] - actual_log1p[-1]) / actual_log1p[-1]) * 100
    final_star_diff = abs(pred_stars[-1] - actual_stars[-1])
    
    plt.tight_layout()
    plt.savefig('horizon_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to 'horizon_benchmark.png'")
    plt.show()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Repository: {repo_name}")
    print(f"Forecast Length: {len(df_future)} days")
    print(f"Actual Final Stars: {int(actual_stars[-1]):,}")
    print(f"Predicted Final Stars: {int(pred_stars[-1]):,}")
    print(f"Star Difference: {int(final_star_diff):,} stars")
    print(f"Final Drift (log1p): {final_drift_pct:.2f}%")
    if final_drift_pct < 5:
        print("✅ PASS: Drift < 5% (Model is stable)")
    elif final_drift_pct < 10:
        print("⚠️ MARGINAL: Drift 5-10% (Acceptable for long horizons)")
    else:
        print("❌ FAIL: Drift > 10% (Model has bias accumulation)")


if __name__ == "__main__":
    run_benchmark()