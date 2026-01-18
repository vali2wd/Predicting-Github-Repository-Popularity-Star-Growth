"""
Multi-Horizon Benchmark: Visualize 30d, 90d, and 180d Predictions Together
Shows all three forecast horizons on the same graph for comparison
Each model predicts different future points from the same starting point
"""

import polars as pl
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datetime import timedelta

# === Configuration ===
FORECAST_HORIZONS = [30, 90, 180]
MODEL_PATHS = {
    30: 'lgbm_github_lead30d.pkl',
    90: 'lgbm_github_lead90d.pkl',
    180: 'lgbm_github_lead180d.pkl'
}
VAL_PATHS = {
    30: 'val_dataset_lead30d.parquet',
    90: 'val_dataset_lead90d.parquet',
    180: 'val_dataset_lead180d.parquet'
}

# Visualization settings
BENCHMARK_REPO = None  # Set to specific repo name or None for random
NUM_REPOS = 3
MIN_DAYS = 200  # Minimum days of data required

# Color scheme for different horizons (dotted lines)
COLORS = {
    30: '#22c55e',   # Green
    90: '#3b82f6',   # Blue
    180: '#ef4444'   # Red
}

def load_resources():
    """Load all models and validation datasets"""
    print("Loading models and validation data...")
    
    models = {}
    val_dfs = {}
    
    for horizon in FORECAST_HORIZONS:
        model_path = MODEL_PATHS[horizon]
        val_path = VAL_PATHS[horizon]
        
        if not os.path.exists(model_path):
            print(f"⚠️  WARNING: Model not found: {model_path}")
            continue
        
        if not os.path.exists(val_path):
            print(f"⚠️  WARNING: Validation data not found: {val_path}")
            continue
        
        models[horizon] = joblib.load(model_path)
        val_dfs[horizon] = pl.scan_parquet(val_path)
        print(f"  ✓ Loaded {horizon}d model")
    
    if not models:
        raise FileNotFoundError("No models found. Run train_model_multi_horizon.py first")
    
    return models, val_dfs

def get_suitable_repos(val_dfs, min_days=MIN_DAYS):
    """Find repos with enough data in all validation sets"""
    print(f"\nFinding repositories with at least {min_days} days of data...")
    
    # Get repos from the shortest horizon (most data available)
    shortest_horizon = min(val_dfs.keys())
    lf = val_dfs[shortest_horizon]
    
    counts = lf.group_by('repo_name').len().filter(pl.col('len') > min_days).collect()
    
    if counts.height == 0:
        raise ValueError(f"No repositories with >{min_days} days of validation data")
    
    candidates = counts.sort('len', descending=True)['repo_name'].to_list()
    
    if BENCHMARK_REPO and BENCHMARK_REPO in candidates:
        selected = [BENCHMARK_REPO]
        if NUM_REPOS > 1:
            others = [r for r in candidates if r != BENCHMARK_REPO]
            selected += random.sample(others, min(NUM_REPOS - 1, len(others)))
        return selected
    else:
        return random.sample(candidates, min(NUM_REPOS, len(candidates)))

def prepare_features(df, model):
    """Extract features for prediction"""
    required_features = model.feature_name_
    
    # Verify no lead columns
    lead_features = [f for f in required_features if 'lead' in f]
    if lead_features:
        raise ValueError(f"⚠️ Model contains lead features: {lead_features[:5]}")
    
    X = df.select(required_features).to_pandas()
    return X

def run_benchmark():
    models, val_dfs = load_resources()
    available_horizons = sorted(models.keys())
    
    print(f"Available horizons: {available_horizons}")
    
    repo_names = get_suitable_repos(val_dfs)
    
    print(f"\nBenchmarking {len(repo_names)} repositories:")
    for repo in repo_names:
        print(f"  - {repo}")
    
    all_stats = {h: [] for h in available_horizons}
    
    # Create separate figure for each repository
    for repo_idx, repo_name in enumerate(repo_names):
        print(f"\n{'='*80}")
        print(f"Repository {repo_idx + 1}/{len(repo_names)}: {repo_name}")
        print('='*80)
        
        # Create figure for this repo
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # First, get the actual star trajectory from the shortest horizon dataset
        shortest_horizon = min(available_horizons)
        df_actual = (val_dfs[shortest_horizon]
                    .filter(pl.col('repo_name') == repo_name)
                    .sort('day')
                    .collect())
        
        if len(df_actual) < 30:
            print(f"⚠️ Skipping {repo_name} - insufficient data")
            continue
        
        # Plot actual stars trajectory (black solid line)
        actual_dates = df_actual['day'].to_list()
        actual_stars = df_actual['total_stars'].to_numpy()
        
        ax.plot(actual_dates, actual_stars, 
               label='Actual Stars',
               color='black', linewidth=2.5, linestyle='-', alpha=0.8, zorder=10)
        
        # Now plot predictions for each horizon
        for horizon in available_horizons:
            # Get data for this repo and horizon
            df = (val_dfs[horizon]
                  .filter(pl.col('repo_name') == repo_name)
                  .sort('day')
                  .collect())
            
            if len(df) < 30:
                print(f"⚠️ Skipping {horizon}d - insufficient data")
                continue
            
            # Prepare features and predict
            X = prepare_features(df, models[horizon])
            
            TARGET = f'total_stars_lead_{horizon}d'
            
            # Predictions and actual future values (for error calculation)
            pred_stars = models[horizon].predict(X)
            actual_future_stars = df[TARGET].to_numpy()
            
            # Dates when predictions were made
            prediction_dates = df['day'].to_list()
            
            # Target dates (when predictions are FOR)
            target_dates = [d + timedelta(days=horizon) for d in prediction_dates]
            
            # Calculate errors
            errors = pred_stars - actual_future_stars
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors ** 2).mean())
            
            mask = actual_future_stars > 0
            if mask.sum() > 0:
                mape = (np.abs(errors[mask]) / actual_future_stars[mask] * 100).mean()
            else:
                mape = np.inf
            
            # Store stats
            all_stats[horizon].append({
                'repo': repo_name,
                'samples': len(df),
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
            print(f"{horizon}d: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
            
            # Plot predicted stars at target dates (colored dotted line)
            ax.plot(target_dates, pred_stars,
                   label=f'Predicted {horizon}d ahead (MAPE: {mape:.1f}%)',
                   color=COLORS[horizon], linewidth=2, linestyle=':', alpha=0.9)
        
        # Formatting
        ax.set_title(f'{repo_name} - Multi-Horizon Forecast', 
                    fontsize=13, fontweight='bold')
        ax.set_ylabel('Total Stars', fontsize=11)
        ax.set_xlabel('Date (Prediction Made)', fontsize=11)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Save and show this repo's figure
        plt.tight_layout()
        safe_repo_name = repo_name.replace('/', '_')
        plt.savefig(f'benchmark/benchmark_{safe_repo_name}.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to 'benchmark/benchmark_{safe_repo_name}.png'")
        plt.show()
    
    # Summary tables by horizon
    print(f"\n{'='*100}")
    print("MULTI-HORIZON FORECAST PERFORMANCE SUMMARY")
    print('='*100)
    
    for horizon in available_horizons:
        if not all_stats[horizon]:
            continue
        
        print(f"\n{horizon}-DAY HORIZON:")
        print(f"{'Repository':<30} {'Samples':>10} {'MAE':>10} {'RMSE':>10} {'MAPE':>8}")
        print('-'*70)
        
        for stat in all_stats[horizon]:
            mape_str = f"{stat['mape']:.1f}%" if stat['mape'] != np.inf else "N/A"
            print(f"{stat['repo']:<30} {stat['samples']:>10,} {stat['mae']:>10.1f} "
                  f"{stat['rmse']:>10.1f} {mape_str:>8}")
        
        # Average stats
        valid_mapes = [s['mape'] for s in all_stats[horizon] if s['mape'] != np.inf]
        avg_mape = np.mean(valid_mapes) if valid_mapes else np.inf
        avg_mae = np.mean([s['mae'] for s in all_stats[horizon]])
        avg_rmse = np.mean([s['rmse'] for s in all_stats[horizon]])
        
        mape_str = f"{avg_mape:.1f}%" if avg_mape != np.inf else "N/A"
        print('-'*70)
        print(f"{'AVERAGE':<30} {'':<10} {avg_mae:>10.1f} {avg_rmse:>10.1f} {mape_str:>8}")
    
    print("\n" + "="*100)
    print("✅ Multi-horizon benchmark complete!")
    print(f"\n📊 Created {len(repo_names)} plots (one per repository):")
    for repo in repo_names:
        safe_name = repo.replace('/', '_')
        print(f"   - benchmark_{safe_name}.png")
    print("\n💡 Interpretation:")
    print("   - Black solid line: Actual star counts")
    print("   - Green dotted: 30-day predictions (shifted 30 days forward)")
    print("   - Blue dotted: 90-day predictions (shifted 90 days forward)")
    print("   - Red dotted: 180-day predictions (shifted 180 days forward)")
    print("   - X-axis shows the TARGET date (not when prediction was made)")
    print("   - Closer to black = better predictions")

if __name__ == "__main__":
    run_benchmark()
