import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

# 1. Load Pre-Split Data
# We can just read_parquet now because the files are likely smaller and ready
print("Loading datasets...")
train_df = pl.read_parquet('train_dataset.parquet')
val_df = pl.read_parquet('val_dataset.parquet')

# 2. Define Features
TARGET = 'total_stars_scaled'
IGNORE_COLS = ['repo_name', 'day'] # Keep in file for analysis, drop for training

# Identify leakage columns to drop (same logic as before)
# We exclude any column that isn't a lag or rolling stat
all_cols = train_df.columns
# Columns that contain "lag" or "rolling" are safe features
feature_cols = [
    c for c in all_cols 
    if ('lag' in c or 'rolling' in c) 
    and c not in IGNORE_COLS 
    and c != TARGET
]

print(f"Training on {len(feature_cols)} features.")

# 3. Prepare X and y
# Convert to pandas for LightGBM (efficient zero-copy often possible)
X_train = train_df.select(feature_cols).to_pandas()
y_train = train_df.select(TARGET).to_pandas().values.ravel()

X_val = val_df.select(feature_cols).to_pandas()
y_val = val_df.select(TARGET).to_pandas().values.ravel()

# Clean up Polars frames to free RAM for LightGBM
del train_df, val_df

# 4. Train
model = lgb.LGBMRegressor(
    n_estimators=2000,        # Increased since we have early stopping
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.8,     # Randomly select 80% of features per tree (prevents overfitting)
    subsample=0.8,            # Randomly select 80% of data per tree
    random_state=42,
    n_jobs=-1
)

callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=100)
]

print("Starting training...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['Train', 'Valid'],
    eval_metric='rmse',
    callbacks=callbacks
)

# 5. Visualize
lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10,6))
plt.title("Feature Importance (Gain)")
plt.tight_layout()
plt.show()