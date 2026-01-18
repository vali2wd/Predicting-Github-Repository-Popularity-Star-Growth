import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import joblib
import os

# Configuration
TRAIN_PATH = 'train_dataset.parquet'
VAL_PATH = 'val_dataset.parquet'
MODEL_PATH = 'lgbm_github_model.pkl'

# 1. Load Data
# We always load data because we need it for evaluation/plotting
print("Loading data...")
train_df = pl.read_parquet(TRAIN_PATH)
val_df = pl.read_parquet(VAL_PATH)

# 2. Create the "Difference" Target
# We predict (Current_Scaled - Prev_Scaled) to model growth velocity
TARGET_DIFF = 'target_diff'

train_df = train_df.with_columns(
    (pl.col('total_stars_scaled') - pl.col('total_stars_lag_1d_scaled')).alias(TARGET_DIFF)
)
val_df = val_df.with_columns(
    (pl.col('total_stars_scaled') - pl.col('total_stars_lag_1d_scaled')).alias(TARGET_DIFF)
)

# 3. Feature Selection
IGNORE_COLS = ['repo_name', 'day', TARGET_DIFF]

# Exclude cumulative features that would cause leakage/persistence bias
cumulative_cols = [
    'total_stars_scaled', 
    'total_stars_lag_1d_scaled', 
    'total_stars_lag_7d_scaled',
    'total_stars_daily_change_scaled'
]

all_cols = train_df.columns
feature_cols = [
    c for c in all_cols 
    if c not in IGNORE_COLS 
    and c not in cumulative_cols
]

print(f"Target: {TARGET_DIFF}")
print(f"Features: {len(feature_cols)}")

# 4. Prepare X and y for LightGBM
# We create the datasets for validation even if we don't train, 
# because we need them to calculate the score of the loaded model.
X_val = val_df.select(feature_cols).to_pandas()
y_val = val_df.select(TARGET_DIFF).to_pandas().values.ravel()

# 5. Model Logic (Load or Train)
if os.path.exists(MODEL_PATH):
    print(f"✅ Found existing model at '{MODEL_PATH}'. Loading...")
    model = joblib.load(MODEL_PATH)
    
    # Optional: Verify the loaded model expects the same number of features
    if model.n_features_ != len(feature_cols):
        print(f"⚠️ Warning: Loaded model expects {model.n_features_} features, but we have {len(feature_cols)}.")
else:
    print(f"❌ No model found at '{MODEL_PATH}'. Training new model...")
    
    # Only convert training data if we actually need to train
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(TARGET_DIFF).to_pandas().values.ravel()
    
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['Train', 'Valid'],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Save the model
    print(f"💾 Saving model to '{MODEL_PATH}'...")
    joblib.dump(model, MODEL_PATH)

# Clean up memory
del train_df

# 6. Evaluation & Reconstruction
print("\n--- Evaluation ---")

# Actual previous values (needed to reverse the math)
prev_scaled = val_df.select('total_stars_lag_1d_scaled').to_pandas().values.ravel()
actual_scaled = val_df.select('total_stars_scaled').to_pandas().values.ravel()

# Predict the Diff
pred_diff = model.predict(X_val)

# Reconstruct: Predicted_Total = Previous + Predicted_Diff
pred_total_scaled = prev_scaled + pred_diff

# Calculate RMSE on the 'Scaled Total' using the new function
rmse_scaled = root_mean_squared_error(actual_scaled, pred_total_scaled)
print(f"RMSE (Scaled Total Space): {rmse_scaled:.5f}")

# 7. Plotting
lgb.plot_importance(model, max_num_features=15, importance_type='gain', figsize=(10,6))
plt.title("Feature Importance (Growth Velocity Model)")
plt.tight_layout()
plt.show()