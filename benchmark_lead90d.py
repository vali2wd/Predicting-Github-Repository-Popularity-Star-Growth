"""
Benchmark 90-Day Lead Forecasting Model
Direct forecast: Predict stars 90 days ahead using only current/past data
NO FUTURE DATA - NO RECURSION - Clean forecasting
"""

import polars as pl
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# --- Config ---
VAL_PATH = 'val_dataset_lead90d.parquet'
MODEL_PATH = 'lgbm_github_lead90d.pkl'
FORECAST_HORIZON = 90

# Choose specific repo or None for random
BENCHMARK_REPO = None  # e.g., 'torvalds/linux' or None for random
NUM_REPOS = 3  # Number of repos to visualize

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print("Loading resources...")
    model = joblib.load(MODEL_PATH)
    lf = pl.scan_parquet(VAL_PATH)
    return model, lf

def get_suitable_repos(lf, min_days=120):
    """Get repos with enough data for meaningful visualization"""
    print("Finding suitable repositories...")
    
    # Count rows per repo
    counts = lf.group_by('repo_name').len().filter(pl.col('len') > min_days).collect()
    
    if counts.height == 0:
        raise ValueError("No repositories with enough validation data")
    
    candidates = counts.sort('len', descending=True)['repo_name'].to_list()
    
    if BENCHMARK_REPO and BENCHMARK_REPO in candidates:
        selected = [BENCHMARK_REPO]
        if NUM_REPOS > 1:
            selected += random.sample([r for r in candidates if r != BENCHMARK_REPO], 
                                     min(NUM_REPOS - 1, len(candidates) - 1))
        return selected
    else:
        return random.sample(candidates, min(NUM_REPOS, len(candidates)))

def prepare_features(df, model):
    """Extract features for prediction (no lead columns)"""
    required_features = model.feature_name_
    
    # Verify no lead columns in features
    lead_features = [f for f in required_features if 'lead' in f]
    if lead_features:
        raise ValueError(f"⚠️ Model contains lead features: {lead_features[:5]}")
    
    X = df.select(required_features).to_pandas()
    return X

def run_benchmark():
    model, lf = load_resources()
    repo_names = get_suitable_repos(lf, min_days=120)
    
    print(f"\nBenchmarking {len(repo_names)} repositories:")
    for repo in repo_names:
        print(f"  - {repo}")
    
    # Create subplots
    fig, axes = plt.subplots(len(repo_names), 1, figsize=(14, 5 * len(repo_names)))
    if len(repo_names) == 1:
        axes = [axes]
    
    all_stats = []
    
    for idx, repo_name in enumerate(repo_names):
        print(f"\n{'='*60}")
        print(f"Repository: {repo_name}")
        print('='*60)
        
        # Get data for this repo
        df = (lf.filter(pl.col('repo_name') == repo_name)
                .sort('day')
                .collect())
        
        # Remove rows with null targets (last 90 days won't have lead values)
        df = df.filter(pl.col(f'total_stars_lead_{FORECAST_HORIZON}d').is_not_null())
        
        if len(df) < 30:
            print(f"⚠️ Skipping {repo_name} - insufficient data")
            continue
        
        # Prepare features and predict
        X = prepare_features(df, model)
        
        # Get actual and predicted values
        actual_stars = df[f'total_stars_lead_{FORECAST_HORIZON}d'].to_numpy()
        pred_stars = model.predict(X)
        current_stars = df['total_stars'].to_numpy()
        dates = df['day'].to_list()
        
        # Calculate errors
        errors = pred_stars - actual_stars
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        
        # Safe MAPE calculation
        mask = actual_stars > 0
        if mask.sum() > 0:
            mape = (np.abs(errors[mask]) / actual_stars[mask] * 100).mean()
        else:
            mape = np.inf
        
        # Store stats
        all_stats.append({
            'repo': repo_name,
            'samples': len(df),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'final_actual': int(actual_stars[-1]),
            'final_pred': int(pred_stars[-1])
        })
        
        print(f"Samples: {len(df):,}")
        print(f"MAE: {mae:.2f} stars")
        print(f"RMSE: {rmse:.2f} stars")
        print(f"MAPE: {mape:.2f}%")
        
        # Plot
        ax = axes[idx]
        
        # Plot current stars (baseline)
        ax.plot(dates, current_stars, label='Current Stars (t=0)', 
                color='gray', alpha=0.6, linewidth=1.5)
        
        # Plot actual stars 90 days ahead
        ax.plot(dates, actual_stars, label=f'Actual Stars (t+{FORECAST_HORIZON}d)', 
                color='black', linewidth=2)
        
        # Plot predicted stars 90 days ahead
        ax.plot(dates, pred_stars, label=f'Predicted Stars (t+{FORECAST_HORIZON}d)', 
                color='red', linestyle='--', linewidth=2)
        
        # Fill between actual and predicted
        ax.fill_between(dates, actual_stars, pred_stars, 
                       color='red', alpha=0.15)
        
        # Formatting
        ax.set_title(f'{repo_name} - {FORECAST_HORIZON}-Day Forecast\n'
                    f'MAE: {mae:.0f} stars | MAPE: {mape:.1f}%')
        ax.set_ylabel('Total Stars')
        ax.set_xlabel('Date (Prediction Made)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(f'benchmark_lead{FORECAST_HORIZON}d.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to 'benchmark_lead{FORECAST_HORIZON}d.png'")
    plt.show()
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: {FORECAST_HORIZON}-Day Forecast Performance")
    print('='*80)
    print(f"{'Repository':<30} {'Samples':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>8}")
    print('-'*80)
    for stat in all_stats:
        mape_str = f"{stat['mape']:.1f}%" if stat['mape'] != np.inf else "N/A"
        print(f"{stat['repo']:<30} {stat['samples']:>10,} {stat['mae']:>10.1f} "
              f"{stat['rmse']:>10.1f} {mape_str:>8}")
    
    # Overall stats
    valid_mapes = [s['mape'] for s in all_stats if s['mape'] != np.inf]
    avg_mape = np.mean(valid_mapes) if valid_mapes else np.inf
    avg_mae = np.mean([s['mae'] for s in all_stats])
    
    mape_str = f"{avg_mape:.1f}%" if avg_mape != np.inf else "N/A"
    print('-'*80)
    print(f"{'AVERAGE':<30} {'':<10} {avg_mae:>10.1f} {'':>10} {mape_str:>8}")
    print('='*80)
    
    print(f"\n✅ This model uses ONLY current/past data - NO FUTURE DATA LEAKAGE")
    print(f"✅ Direct forecasting - NO RECURSIVE DRIFT")
    print(f"\n💡 Interpretation: Model predicts what stars will be {FORECAST_HORIZON} days from now")
    print(f"   based on current activity, lags, and rolling statistics.")

if __name__ == "__main__":
    run_benchmark()
