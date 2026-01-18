"""
Train LightGBM model for 90-Day GitHub Star Forecasting (STATIONARY TARGET)
Predicts PERCENTAGE GROWTH instead of absolute values to remove trend
Target: (stars_90d_ahead - current_stars) / current_stars
Uses ONLY delta features (pure changes over time windows)
"""

import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib

# === Configuration ===
SOURCE_PATH = 'github_features_lead.parquet'
TRAIN_PATH = 'train_dataset_lead90d_stationary.parquet'
VAL_PATH = 'val_dataset_lead90d_stationary.parquet'
MODEL_PATH = 'lgbm_github_lead90d_stationary.pkl'
FORECAST_HORIZON = 90

# === Step 1: Load and Split Data (Repository-Level Random Split) ===
print(f"Loading data for {FORECAST_HORIZON}-day STATIONARY forecasting...")
print("Target: PERCENTAGE GROWTH RATE (scale-invariant)")
print("Features: ONLY delta values (pure changes)")
print("Using REPOSITORY-LEVEL random split (80/20)")

# Load full dataset
df = pl.read_parquet(SOURCE_PATH)

# Get all unique repositories
all_repos = df.select('repo_name').unique().to_series().to_list()
print(f"\nTotal repositories: {len(all_repos):,}")

# Randomly split repositories (80% train, 20% validation)
import random
random.seed(42)
random.shuffle(all_repos)

split_idx = int(len(all_repos) * 0.8)
train_repos = set(all_repos[:split_idx])
val_repos = set(all_repos[split_idx:])

print(f"Train repositories: {len(train_repos):,}")
print(f"Validation repositories: {len(val_repos):,}")

# Split data by repository
train_df = df.filter(pl.col('repo_name').is_in(train_repos)).sort('repo_name', 'day')
val_df = df.filter(pl.col('repo_name').is_in(val_repos)).sort('repo_name', 'day')

print(f"Initial train samples: {len(train_df):,}")
print(f"Initial val samples: {len(val_df):,}")

# === Step 2: Create Stationary Target and Delta Features ===
print("\n=== Creating Stationary Target and Features ===")
print("Target: growth_rate = (stars_90d_ahead - current_stars) / current_stars")
print("Features: DELTA VALUES ONLY (no cumulative)")

LEAD_COL = f'total_stars_lead_{FORECAST_HORIZON}d'
TARGET = 'target_growth_rate'

# Filter out rows where:
# 1. Lead value is null (last 90 days)
# 2. Current stars is zero (can't compute percentage)
print("\nFiltering data...")
print("  - Removing rows with null lead values")
print("  - Removing rows with zero current stars (can't compute percentage)")

train_df = (train_df
    .filter(pl.col(LEAD_COL).is_not_null())
    .filter(pl.col('total_stars') > 0)
    .with_columns([
        ((pl.col(LEAD_COL) - pl.col('total_stars')) / pl.col('total_stars')).alias(TARGET)
    ])
)

val_df = (val_df
    .filter(pl.col(LEAD_COL).is_not_null())
    .filter(pl.col('total_stars') > 0)
    .with_columns([
        ((pl.col(LEAD_COL) - pl.col('total_stars')) / pl.col('total_stars')).alias(TARGET)
    ])
)

print(f"\nAfter filtering - Train: {len(train_df):,}, Val: {len(val_df):,}")

# === Create Delta Features (Remove Cumulative Values) ===
print("\n=== Engineering Delta Features ===")
print("Computing: current - lag for all cumulative metrics")

# Identify base metrics (all cumulative columns)
BASE_METRICS = [
    'total_stars', 'total_forks', 'total_issues_opened',
    'total_prs_opened', 'total_commits', 'total_issues_closed', 
    'total_prs_merged', 'total_comments'
]
LAG_PERIODS = ['1d', '7d', '30d', '60d']

delta_exprs = []
for metric in BASE_METRICS:
    for lag in LAG_PERIODS:
        lag_col = f'{metric}_lag_{lag}'
        delta_col = f'{metric}_delta_{lag}'
        
        # Check if columns exist
        if metric in train_df.columns and lag_col in train_df.columns:
            delta_exprs.append(
                (pl.col(metric) - pl.col(lag_col)).alias(delta_col)
            )

print(f"Creating {len(delta_exprs)} delta features...")

train_df = train_df.with_columns(delta_exprs)
val_df = val_df.with_columns(delta_exprs)

# Analyze target distribution
train_target_stats = train_df.select(TARGET).describe()
print("\nTarget distribution (growth rate):")
print(train_target_stats)

# Save splits
print("\nSaving train/val splits...")
train_df.write_parquet(TRAIN_PATH)
val_df.write_parquet(VAL_PATH)

# === Step 3: Feature Selection (CRITICAL: Use ONLY Delta Features) ===
print("\n=== Feature Selection: ONLY Delta Features ===")

# Start with all columns
all_cols = train_df.columns

# Select ONLY delta features
feature_cols = [c for c in all_cols if 'delta_' in c]

print(f"\n✅ Selected {len(feature_cols)} delta features ONLY")
print(f"   These represent changes over time windows (current - lag)")

if feature_cols:
    print(f"\nDelta features by metric:")
    for metric in BASE_METRICS:
        metric_deltas = [f for f in feature_cols if metric in f]
        if metric_deltas:
            print(f"  {metric}: {len(metric_deltas)} deltas")
            for feat in metric_deltas[:3]:
                print(f"    - {feat}")

# === Step 4: Prepare Training Data ===
print("\nPreparing datasets...")
X_train = train_df.select(feature_cols).to_pandas()
y_train = train_df.select(TARGET).to_pandas().values.ravel()

X_val = val_df.select(feature_cols).to_pandas()
y_val = val_df.select(TARGET).to_pandas().values.ravel()

print(f"Training samples: {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")
print(f"\nTarget range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"Target mean: {y_train.mean():.4f} (average growth rate)")

# === Step 5: Train LightGBM ===
print(f"\n=== Training LightGBM (STATIONARY {FORECAST_HORIZON}-Day Growth Model) ===")
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.7,
    subsample=0.8,
    random_state=42,
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

# === Step 6: Save Model ===
print(f"\n💾 Saving model to '{MODEL_PATH}'...")
joblib.dump(model, MODEL_PATH)

# === Step 7: Evaluation (Growth Rate Space) ===
print("\n=== Evaluation: Growth Rate Prediction ===")

pred_growth = model.predict(X_val)
actual_growth = y_val

# Metrics in growth rate space
rmse_growth = root_mean_squared_error(actual_growth, pred_growth)
mae_growth = mean_absolute_error(actual_growth, pred_growth)

print(f"RMSE (growth rate): {rmse_growth:.4f}")
print(f"MAE (growth rate): {mae_growth:.4f}")

# === Step 8: Evaluation (Reconstructed Star Space) ===
print("\n=== Evaluation: Reconstructed Star Predictions ===")

# Get current stars for reconstruction
current_stars_val = val_df.select('total_stars').to_pandas().values.ravel()
actual_future_stars = val_df.select(LEAD_COL).to_pandas().values.ravel()

# Reconstruct predictions
pred_future_stars = current_stars_val * (1 + pred_growth)

# Metrics in star space
rmse_stars = root_mean_squared_error(actual_future_stars, pred_future_stars)
mae_stars = mean_absolute_error(actual_future_stars, pred_future_stars)

# Safe MAPE
mask = actual_future_stars > 0
mape = (abs(actual_future_stars[mask] - pred_future_stars[mask]) / actual_future_stars[mask] * 100).mean()

print(f"RMSE (stars): {rmse_stars:.2f}")
print(f"MAE (stars): {mae_stars:.2f}")
print(f"MAPE: {mape:.2f}%")

# === Step 9: Feature Importance ===
print("\n=== Feature Importance (Top 20) ===")
importance_df = pl.DataFrame({
    'feature': model.feature_name_,
    'importance': model.feature_importances_
}).sort('importance', descending=True)

print(importance_df.head(20))

# Plot
plt.figure(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title(f"Feature Importance: {FORECAST_HORIZON}-Day Growth Rate Model (Stationary)")
plt.xlabel("Gain")
plt.tight_layout()
plt.savefig(f'feature_importance_lead{FORECAST_HORIZON}d_stationary.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Feature importance plot saved")
plt.show()

# === Step 10: Sanity Check ===
print("\n=== Sanity Check ===")
top_features = importance_df['feature'].head(10).to_list()
print("Top 10 features:")
for i, feat in enumerate(top_features, 1):
    print(f"  {i}. {feat}")

# Check for lead columns in top features (should be NONE)
lead_in_top = [f for f in top_features if 'lead' in f]
if lead_in_top:
    print(f"\n❌ ERROR: Lead columns found in top features: {lead_in_top}")
    print("This indicates data leakage!")
else:
    print(f"\n✅ No lead columns in top features (no data leakage)")

# Check for non-delta columns in top features (should be NONE)
non_delta_in_top = [f for f in top_features if 'delta_' not in f]

if non_delta_in_top:
    print(f"\n⚠️  WARNING: Non-delta features in top 10: {non_delta_in_top}")
else:
    print(f"✅ All top features are delta values (pure changes)")

print(f"\n✅ Training complete!")
print(f"📊 Model uses ONLY delta features (change over time windows)")
print(f"📊 Target is growth rate (scale-invariant, stationary)")
print(f"🔄 Benchmark will reconstruct absolute star predictions")
print(f"Next step: Run benchmark to visualize {FORECAST_HORIZON}-day predictions")
