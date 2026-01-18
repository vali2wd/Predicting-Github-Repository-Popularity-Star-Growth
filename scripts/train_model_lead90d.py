"""
Train LightGBM model for 90-Day GitHub Star Forecasting
Direct forecasting approach: Predict stars 90 days ahead using current activity
NO FUTURE DATA LEAKAGE - Only uses current/past features
"""

import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta, date

# === Configuration ===
SOURCE_PATH = 'github_features_lead.parquet'
TRAIN_PATH = 'train_dataset_lead90d.parquet'
VAL_PATH = 'val_dataset_lead90d.parquet'
MODEL_PATH = 'lgbm_github_lead90d.pkl'
FORECAST_HORIZON = 90  # Predicting 90 days ahead

# === Step 1: Load and Split Data (Repository-Level Random Split) ===
print(f"Loading data for {FORECAST_HORIZON}-day forecasting...")
print("Using REPOSITORY-LEVEL random split (80/20)")
print("This prevents distribution shift from chronological splitting")

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

# Save splits
print("\nSaving train/val splits...")
train_df.write_parquet(TRAIN_PATH)
val_df.write_parquet(VAL_PATH)

print(f"Train samples: {len(train_df):,}")
print(f"Val samples: {len(val_df):,}")

# === Step 2: Load Data for Training ===
print("\nLoading training data...")
train_df = pl.read_parquet(TRAIN_PATH)
val_df = pl.read_parquet(VAL_PATH)

# === Step 3: Define Target ===
TARGET = f'total_stars_lead_{FORECAST_HORIZON}d'
print(f"\nTarget: {TARGET}")

# Check if target exists and has non-null values
if TARGET not in train_df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset")

null_count = train_df.select(pl.col(TARGET).is_null().sum()).item()
print(f"Null values in target: {null_count:,}")

# Drop rows with null targets (last 90 days of data won't have lead values)
train_df = train_df.filter(pl.col(TARGET).is_not_null())
val_df = val_df.filter(pl.col(TARGET).is_not_null())

print(f"After removing nulls - Train: {train_df.shape}, Val: {val_df.shape}")

# === Step 4: Feature Selection (CRITICAL: Ban ALL Lead Columns) ===
# BANNED columns:
BANNED = {
    'repo_name',                      # Identifier
    'day',                            # Identifier
    TARGET,                           # The target itself
}

# Ban ALL lead columns (they contain future information)
all_cols = train_df.columns
lead_cols = [c for c in all_cols if 'lead' in c]
BANNED.update(lead_cols)

print(f"\n🚫 Banned {len(BANNED)} columns (including {len(lead_cols)} lead columns)")

# Select features
feature_cols = [c for c in all_cols if c not in BANNED]

print(f"Total features: {len(feature_cols)}")
print(f"\nFeature categories:")
print(f"  - Current values: {len([c for c in feature_cols if 'lag' not in c and 'rolling' not in c and 'daily' not in c])}")
print(f"  - Lags (1d, 7d, 30d): {len([c for c in feature_cols if 'lag' in c])}")
print(f"  - Rolling stats: {len([c for c in feature_cols if 'rolling' in c])}")
print(f"  - Daily changes: {len([c for c in feature_cols if 'daily_change' in c])}")

# === Step 5: Prepare Training Data ===
print("\nPreparing datasets...")
X_train = train_df.select(feature_cols).to_pandas()
y_train = train_df.select(TARGET).to_pandas().values.ravel()

X_val = val_df.select(feature_cols).to_pandas()
y_val = val_df.select(TARGET).to_pandas().values.ravel()

print(f"Training samples: {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")

# === Step 6: Train LightGBM ===
print(f"\n=== Training LightGBM ({FORECAST_HORIZON}-Day Forecast Model) ===")
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=31,
    colsample_bytree=0.7,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    # reg_lambda=1.0
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

# === Step 8: Evaluation ===
print("\n=== Evaluation: Direct Forecast Accuracy ===")

# Actual stars 90 days ahead
actual_future = y_val

# Predicted stars 90 days ahead
pred_future = model.predict(X_val)

# Calculate metrics
rmse = root_mean_squared_error(actual_future, pred_future)
mae = mean_absolute_error(actual_future, pred_future)
mape = (abs(actual_future - pred_future) / actual_future * 100).mean()

print(f"RMSE: {rmse:.2f} stars")
print(f"MAE: {mae:.2f} stars")
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
plt.title(f"Feature Importance: {FORECAST_HORIZON}-Day Forecast Model")
plt.xlabel("Gain")
plt.tight_layout()
plt.savefig(f'feature_importance_lead{FORECAST_HORIZON}d.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Feature importance plot saved")
plt.show()

# === Step 10: Sanity Check ===
print("\n=== Sanity Check ===")
top_features = importance_df['feature'].head(5).to_list()
print("Top 5 features:")
for i, feat in enumerate(top_features, 1):
    print(f"  {i}. {feat}")

# Check for lead columns in top features (should be NONE)
lead_in_top = [f for f in top_features if 'lead' in f]
if lead_in_top:
    print(f"\n❌ ERROR: Lead columns found in top features: {lead_in_top}")
    print("This indicates data leakage!")
else:
    print(f"\n✅ SUCCESS: No lead columns in top features (no data leakage)")

print(f"\n✅ Training complete!")
print(f"Next step: Run benchmark to visualize {FORECAST_HORIZON}-day predictions")
