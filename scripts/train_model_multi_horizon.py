"""
Train Multi-Horizon LightGBM Models for GitHub Star Forecasting
Trains 3 separate models: 30-day, 90-day, and 180-day forecasts
Direct forecasting approach: NO FUTURE DATA LEAKAGE
Uses repository-level random split to prevent distribution shift
"""

import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
import random

# === Configuration ===
SOURCE_PATH = 'github_features_lead.parquet'
FORECAST_HORIZONS = [30, 90, 180]
RANDOM_SEED = 42

print("="*80)
print("MULTI-HORIZON GITHUB STAR FORECASTING")
print("="*80)
print(f"Training {len(FORECAST_HORIZONS)} models: {FORECAST_HORIZONS} days ahead")
print(f"Split strategy: Repository-level random (80/20)")
print("="*80)

# === Step 1: Load and Split Data (Repository-Level Random Split) ===
print("\n[1/4] Loading data and splitting by repository...")
df = pl.read_parquet(SOURCE_PATH)

# Get all unique repositories
all_repos = df.select('repo_name').unique().to_series().to_list()
print(f"Total repositories: {len(all_repos):,}")
print(f"Total samples: {len(df):,}")

# Randomly split repositories (80% train, 20% validation)
random.seed(RANDOM_SEED)
random.shuffle(all_repos)

split_idx = int(len(all_repos) * 0.8)
train_repos = set(all_repos[:split_idx])
val_repos = set(all_repos[split_idx:])

print(f"Train repositories: {len(train_repos):,}")
print(f"Validation repositories: {len(val_repos):,}")

# Split data by repository
train_df = df.filter(pl.col('repo_name').is_in(train_repos)).sort('repo_name', 'day')
val_df = df.filter(pl.col('repo_name').is_in(val_repos)).sort('repo_name', 'day')

print(f"Train samples: {len(train_df):,}")
print(f"Val samples: {len(val_df):,}")

# === Step 2: Identify Features (Common for All Models) ===
print("\n[2/4] Identifying features...")

# BANNED columns (common for all horizons)
all_cols = train_df.columns
lead_cols = [c for c in all_cols if 'lead' in c]

BANNED = {
    'repo_name',
    'day',
    *lead_cols  # Ban ALL lead columns
}

# Select features (same for all models)
feature_cols = [c for c in all_cols if c not in BANNED]
print(f"Total features: {len(feature_cols)}")
print(f"Banned columns: {len(BANNED)} (including {len(lead_cols)} lead columns)")

# === Step 3: Train Models for Each Horizon ===
print("\n[3/4] Training models...")

results = {}

for horizon in FORECAST_HORIZONS:
    print(f"\n{'='*80}")
    print(f"TRAINING {horizon}-DAY MODEL")
    print('='*80)
    
    TARGET = f'total_stars_lead_{horizon}d'
    MODEL_PATH = f'lgbm_github_lead{horizon}d.pkl'
    TRAIN_PATH = f'train_dataset_lead{horizon}d.parquet'
    VAL_PATH = f'val_dataset_lead{horizon}d.parquet'
    
    # Check if target exists
    if TARGET not in train_df.columns:
        print(f"⚠️  WARNING: Target '{TARGET}' not found, skipping {horizon}d model")
        continue
    
    # Filter nulls for this horizon
    train_h = train_df.filter(pl.col(TARGET).is_not_null())
    val_h = val_df.filter(pl.col(TARGET).is_not_null())
    
    null_count = len(train_df) - len(train_h)
    print(f"Removed {null_count:,} samples with null target")
    print(f"Train samples: {len(train_h):,}")
    print(f"Val samples: {len(val_h):,}")
    
    # Save splits
    train_h.write_parquet(TRAIN_PATH)
    val_h.write_parquet(VAL_PATH)
    
    # Prepare data
    X_train = train_h.select(feature_cols).to_pandas()
    y_train = train_h.select(TARGET).to_pandas().values.ravel()
    
    X_val = val_h.select(feature_cols).to_pandas()
    y_val = val_h.select(TARGET).to_pandas().values.ravel()
    
    # Train model
    print(f"\nTraining LightGBM for {horizon}d horizon...")
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.7,
        subsample=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['Train', 'Valid'],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100)
        ]
    )
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved: {MODEL_PATH}")
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    rmse = root_mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    mask = y_val > 0
    mape = (abs(y_val[mask] - y_pred[mask]) / y_val[mask] * 100).mean()
    
    print(f"\n📊 {horizon}d Model Performance:")
    print(f"  RMSE: {rmse:.2f} stars")
    print(f"  MAE: {mae:.2f} stars")
    print(f"  MAPE: {mape:.2f}%")
    
    # Store results
    results[horizon] = {
        'model_path': MODEL_PATH,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'best_iteration': model.best_iteration_
    }
    
    # Feature importance
    importance_df = pl.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort('importance', descending=True)
    
    print(f"\n📋 Top 10 Features ({horizon}d):")
    for i, row in enumerate(importance_df.head(10).iter_rows(), 1):
        print(f"  {i}. {row[0]}: {row[1]:.0f}")

# === Step 4: Summary ===
print("\n" + "="*80)
print("TRAINING COMPLETE - SUMMARY")
print("="*80)

print(f"\n{'Horizon':<10} {'Model Path':<30} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}")
print("-"*80)
for horizon in FORECAST_HORIZONS:
    if horizon in results:
        r = results[horizon]
        print(f"{horizon}d{' ':<7} {r['model_path']:<30} {r['rmse']:<12.2f} {r['mae']:<12.2f} {r['mape']:<10.2f}%")

print("\n✅ All models trained successfully!")
print(f"\nNext step: Run 'benchmark_multi_horizon.py' to visualize all predictions")
