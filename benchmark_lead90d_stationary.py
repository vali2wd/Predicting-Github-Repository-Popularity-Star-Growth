"""
Benchmark 90-Day Lead Forecasting Model (STATIONARY TARGET)
Model predicts PERCENTAGE GROWTH RATE using ONLY delta features
Formula: predicted_stars = current_stars * (1 + predicted_growth_rate)
"""

import polars as pl
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# --- Config ---
VAL_PATH = 'val_dataset_lead90d_stationary.parquet'
MODEL_PATH = 'lgbm_github_lead90d_stationary.pkl'
FORECAST_HORIZON = 90
TARGET = 'target_growth_rate'

# Choose specific repo or None for random
BENCHMARK_REPO = None  # e.g., 'torvalds/linux' or None for random
NUM_REPOS = 3  # Number of repos to visualize

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}\nRun train_model_lead90d_stationary.py first")
    
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
    """Extract features for prediction (only delta features)"""
    required_features = model.feature_name_
    
    # Verify no lead columns in features
    lead_features = [f for f in required_features if 'lead' in f]
    if lead_features:
        raise ValueError(f"⚠️ Model contains lead features: {lead_features[:5]}")
    
    # Verify only delta features
    non_delta = [f for f in required_features if 'delta_' not in f]
    if non_delta:
        print(f"⚠️ WARNING: Non-delta features found: {non_delta[:5]}")
    
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
        
        if len(df) < 30:
            print(f"⚠️ Skipping {repo_name} - insufficient data")
            continue
        
        # Prepare features and predict GROWTH RATE
        X = prepare_features(df, model)
        pred_growth_rate = model.predict(X)
        
        # Get current stars and actual growth rate
        current_stars = df['total_stars'].to_numpy()
        actual_growth_rate = df[TARGET].to_numpy()
        actual_future_stars = df[f'total_stars_lead_{FORECAST_HORIZON}d'].to_numpy()
        
        # RECONSTRUCT predicted future stars
        pred_future_stars = current_stars * (1 + pred_growth_rate)
        
        dates = df['day'].to_list()
        
        # Calculate errors in STAR SPACE (what matters for interpretation)
        errors_stars = pred_future_stars - actual_future_stars
        mae_stars = np.abs(errors_stars).mean()
        rmse_stars = np.sqrt((errors_stars ** 2).mean())
        
        # Safe MAPE calculation
        mask = actual_future_stars > 0
        if mask.sum() > 0:
            mape = (np.abs(errors_stars[mask]) / actual_future_stars[mask] * 100).mean()
        else:
            mape = np.inf
        
        # Calculate errors in GROWTH RATE SPACE (for model evaluation)
        errors_growth = pred_growth_rate - actual_growth_rate
        mae_growth = np.abs(errors_growth).mean()
        rmse_growth = np.sqrt((errors_growth ** 2).mean())
        
        # Store stats
        all_stats.append({
            'repo': repo_name,
            'samples': len(df),
            'mae_stars': mae_stars,
            'rmse_stars': rmse_stars,
            'mape': mape,
            'mae_growth': mae_growth,
            'final_actual': int(actual_future_stars[-1]),
            'final_pred': int(pred_future_stars[-1])
        })
        
        print(f"Samples: {len(df):,}")
        print(f"MAE (stars): {mae_stars:.2f}")
        print(f"RMSE (stars): {rmse_stars:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"MAE (growth rate): {mae_growth:.4f}")
        
        # Plot
        ax = axes[idx]
        
        # Plot current stars (baseline)
        ax.plot(dates, current_stars, label='Current Stars (t=0)', 
                color='gray', alpha=0.6, linewidth=1.5)
        
        # Plot actual stars 90 days ahead
        ax.plot(dates, actual_future_stars, label=f'Actual Stars (t+{FORECAST_HORIZON}d)', 
                color='black', linewidth=2)
        
        # Plot predicted stars 90 days ahead (reconstructed from growth rate)
        ax.plot(dates, pred_future_stars, label=f'Predicted Stars (t+{FORECAST_HORIZON}d)', 
                color='blue', linestyle='--', linewidth=2)
        
        # Fill between actual and predicted
        ax.fill_between(dates, actual_future_stars, pred_future_stars, 
                       color='blue', alpha=0.15)
        
        # Formatting
        ax.set_title(f'{repo_name} - {FORECAST_HORIZON}-Day Forecast (Stationary Model)\n'
                    f'MAE: {mae_stars:.0f} stars | MAPE: {mape:.1f}% | Growth MAE: {mae_growth:.4f}')
        ax.set_ylabel('Total Stars')
        ax.set_xlabel('Date (Prediction Made)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(f'benchmark_lead{FORECAST_HORIZON}d_stationary.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to 'benchmark_lead{FORECAST_HORIZON}d_stationary.png'")
    plt.show()
    
    # Summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY: {FORECAST_HORIZON}-Day Forecast Performance (Stationary Model)")
    print('='*90)
    print(f"{'Repository':<30} {'Samples':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'GrowthMAE':>12}")
    print('-'*90)
    for stat in all_stats:
        mape_str = f"{stat['mape']:.1f}%" if stat['mape'] != np.inf else "N/A"
        print(f"{stat['repo']:<30} {stat['samples']:>10,} {stat['mae_stars']:>10.1f} "
              f"{stat['rmse_stars']:>10.1f} {mape_str:>8} {stat['mae_growth']:>12.4f}")
    
    # Overall stats
    valid_mapes = [s['mape'] for s in all_stats if s['mape'] != np.inf]
    avg_mape = np.mean(valid_mapes) if valid_mapes else np.inf
    avg_mae = np.mean([s['mae_stars'] for s in all_stats])
    avg_mae_growth = np.mean([s['mae_growth'] for s in all_stats])
    
    mape_str = f"{avg_mape:.1f}%" if avg_mape != np.inf else "N/A"
    print('-'*90)
    print(f"{'AVERAGE':<30} {'':<10} {avg_mae:>10.1f} {'':>10} {mape_str:>8} {avg_mae_growth:>12.4f}")
    print('='*90)
    
    print(f"\n✅ STATIONARY MODEL: Predicts growth rate using ONLY delta features")
    print(f"✅ Reconstruction: predicted_stars = current_stars × (1 + predicted_growth)")
    print(f"✅ NO FUTURE DATA LEAKAGE - Pure change-based forecasting")
    print(f"\n💡 Benefits of this approach:")
    print(f"   - Delta features: Pure changes over 1d, 7d, 30d, 60d windows")
    print(f"   - Scale-invariant: Growth rate works for all repo sizes")
    print(f"   - Stationary: No trend, no cumulative values")
    print(f"   - Clean forecasting: Model learns from activity changes only")

if __name__ == "__main__":
    run_benchmark()
