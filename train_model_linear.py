"""
Train LightGBM model for GitHub Star Forecasting - LINEAR SCALE VERSION
Following the "Velocity Model" approach: Predict First Derivative in Linear Space
"""

import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta

# === Configuration ===
SOURCE_PATH = 'processed_github_features.parquet'
TRAIN_PATH = 'train_dataset_linear.parquet'
VAL_PATH = 'val_dataset_linear.parquet'
MODEL_PATH = 'lgbm_github_linear.pkl'

# === Step 1: Load and Split Data (Polars Lazy) ===
print("Loading data with Polars (lazy mode)...")
lf = pl.scan_parquet(SOURCE_PATH)

# Get date range for chronological split (80% train, 20% validation)
date_stats = lf.select([
    pl.col('day').min().alias('min_date'),
    pl.col('day').max().alias('max_date')
]).collect()

min_date = date_stats['min_date'][0]
max_date = date_stats['max_date'][0]
print(f"Date range: {min_date} to {max_date}")

# Calculate split date (80% point) using string manipulation
if isinstance(min_date, str):
    min_dt = datetime.strptime(min_date, '%Y-%m-%d')
    max_dt = datetime.strptime(max_date, '%Y-%m-%d')
else:
    min_dt = min_date
    max_dt = max_date
    
date_range = (max_dt - min_dt).days
split_dt = min_dt + timedelta(days=int(date_range * 0.8))
split_date = split_dt.strftime('%Y-%m-%d')

print(f"Split date (80%): {split_date}")

# Convert split_date to proper date type for comparison
from datetime import date
split_date_obj = date(split_dt.year, split_dt.month, split_dt.day)

# Split lazily
train_lf = lf.filter(pl.col('day') < split_date_obj).sort('repo_name', 'day')
val_lf = lf.filter(pl.col('day') >= split_date_obj).sort('repo_name', 'day')

# Save splits for benchmarking
print("Saving train/val splits...")
train_lf.sink_parquet(TRAIN_PATH)
val_lf.sink_parquet(VAL_PATH)

# === Step 2: Collect Data for Training ===
print("Collecting training data...")
train_df = pl.read_parquet(TRAIN_PATH)
val_df = pl.read_parquet(VAL_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")

# === Step 3: Create Target (Velocity = First Derivative in Linear Space) ===
# Target = total_stars_today - total_stars_yesterday
TARGET = 'target_velocity_linear'

# Check which lag column exists
has_lag_1d = 'total_stars_lag_1d' in train_df.columns
has_lag_scaled = 'total_stars_lag_1d_scaled' in train_df.columns

if has_lag_1d:
    lag_col = 'total_stars_lag_1d'
    current_col = 'total_stars'
elif has_lag_scaled:
    lag_col = 'total_stars_lag_1d_scaled'
    current_col = 'total_stars_scaled'
else:
    raise ValueError("No lag column found. Dataset should have either 'total_stars_lag_1d' or 'total_stars_lag_1d_scaled'")

print(f"Using columns: {current_col} - {lag_col}")

train_df = train_df.with_columns(
    (pl.col(current_col) - pl.col(lag_col)).alias(TARGET)
)
val_df = val_df.with_columns(
    (pl.col(current_col) - pl.col(lag_col)).alias(TARGET)
)

# === Step 4: Feature Selection (CRITICAL: Prevent Data Leakage) ===
# BANNED columns that would cause the model to cheat:
BANNED = {
    'repo_name',                      # Identifier
    'day',                            # Identifier
    'total_stars',                    # Current target (direct leakage)
    'total_stars_scaled',             # Scaled version of target
    'total_stars_lag_1d',             # Previous value (causes persistence bias)
    'total_stars_lag_1d_scaled',      # Scaled version
    'total_stars_lag_7d',             # Older lag
    'total_stars_lag_7d_scaled',      # Scaled version
    'total_stars_daily_change',       # Direct derivative of target
    'total_stars_daily_change_scaled', # Scaled version
    TARGET                            # The target itself
}

all_cols = train_df.columns
feature_cols = [c for c in all_cols if c not in BANNED]

print(f"\nTarget: {TARGET}")
print(f"Total features: {len(feature_cols)}")
print(f"\nFeature categories:")
print(f"  - Commits: {len([c for c in feature_cols if 'commit' in c])}")
print(f"  - Forks: {len([c for c in feature_cols if 'fork' in c])}")
print(f"  - Issues: {len([c for c in feature_cols if 'issue' in c])}")
print(f"  - PRs: {len([c for c in feature_cols if 'pr' in c])}")
print(f"  - Stars (rolling stats only): {len([c for c in feature_cols if 'star' in c])}")

# === Step 5: Prepare Training Data ===
print("\nPreparing datasets...")
X_train = train_df.select(feature_cols).to_pandas()
y_train = train_df.select(TARGET).to_pandas().values.ravel()

X_val = val_df.select(feature_cols).to_pandas()
y_val = val_df.select(TARGET).to_pandas().values.ravel()

# === Step 6: Train LightGBM ===
print("\n=== Training LightGBM (Velocity Model - Linear Scale) ===")
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.1,
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

# === Step 7: Save Model ===
print(f"\n💾 Saving model to '{MODEL_PATH}'...")
joblib.dump(model, MODEL_PATH)

# === Step 8: Evaluation (Reconstruction Test) ===
print("\n=== Evaluation: Reconstruction RMSE ===")

# Get actual values
prev_stars = val_df.select(lag_col).to_pandas().values.ravel()
actual_stars = val_df.select(current_col).to_pandas().values.ravel()

# Predict velocity
pred_velocity = model.predict(X_val)

# Reconstruct: Current = Previous + Velocity
pred_stars = prev_stars + pred_velocity

# Calculate metrics
rmse = root_mean_squared_error(actual_stars, pred_stars)
mae = mean_absolute_error(actual_stars, pred_stars)
print(f"RMSE (linear space): {rmse:.2f} stars")
print(f"MAE (linear space): {mae:.2f} stars")

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
plt.title("Feature Importance: Growth Velocity Model (Linear Scale)")
plt.xlabel("Gain")
plt.tight_layout()
plt.savefig('feature_importance_linear.png', dpi=150, bbox_inches='tight')
print("\n📊 Feature importance plot saved to 'feature_importance_linear.png'")
plt.show()

# === Step 10: Sanity Check ===
print("\n=== Sanity Check ===")
top_features = importance_df['feature'].head(5).to_list()
print("Top 5 features:")
for i, feat in enumerate(top_features, 1):
    print(f"  {i}. {feat}")

if any('lag_1d' in f or 'lag_7d' in f for f in top_features if 'star' in f):
    print("\n⚠️ WARNING: Model is using star lags (persistence bias detected!)")
elif any('commit' in f or 'fork' in f or 'issue' in f or 'pr' in f for f in top_features):
    print("\n✅ SUCCESS: Model is using activity features (commits, PRs, etc.)")
else:
    print("\n⚠️ WARNING: Top features are unexpected. Review feature importance.")

print("\n✅ Training complete!")
print(f"Next step: Create and run 'benchmark_horizon_linear.py' to test 90-day forecast stability")
