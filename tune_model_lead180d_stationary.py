"""
Hyperparameter Tuning for 180-Day Stationary Growth Model
Uses Optuna with GroupKFold cross-validation by repo_name
Optimizes LightGBM hyperparameters to minimize RMSE
"""

import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np
import json

# === Configuration ===
SOURCE_PATH = 'github_features_lead.parquet'
BEST_PARAMS_PATH = 'best_params_lead180d_stationary.json'
MODEL_PATH = 'lgbm_github_lead180d_stationary_tuned.pkl'
FORECAST_HORIZON = 180

# Optuna settings
N_TRIALS = 100
N_FOLDS = 5
RANDOM_SEED = 42

print("="*80)
print(f"HYPERPARAMETER TUNING: {FORECAST_HORIZON}-Day Stationary Growth Model")
print("="*80)
print(f"Optimizer: Optuna TPE")
print(f"Trials: {N_TRIALS}")
print(f"CV Strategy: GroupKFold with {N_FOLDS} folds (by repo_name)")
print(f"Objective: Minimize RMSE (growth rate)")
print("="*80)

# === Step 1: Load and Prepare Data ===
print("\nLoading data...")
df = pl.read_parquet(SOURCE_PATH)

LEAD_COL = f'total_stars_lead_{FORECAST_HORIZON}d'
TARGET = 'target_growth_rate'

# Filter and create target
df = (df
    .filter(pl.col(LEAD_COL).is_not_null())
    .filter(pl.col('total_stars') > 0)
    .with_columns([
        ((pl.col(LEAD_COL) - pl.col('total_stars')) / pl.col('total_stars')).alias(TARGET)
    ])
)

print(f"Total samples after filtering: {len(df):,}")

# === Step 2: Create Delta Features ===
print("\nCreating delta features...")

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
        
        if metric in df.columns and lag_col in df.columns:
            delta_exprs.append(
                (pl.col(metric) - pl.col(lag_col)).alias(delta_col)
            )

df = df.with_columns(delta_exprs)

# === Step 3: Select Features ===
feature_cols = [c for c in df.columns if 'delta_' in c]
print(f"Using {len(feature_cols)} delta features")

# === Step 4: Prepare Arrays for Training ===
print("\nPreparing training data...")
X = df.select(feature_cols).to_pandas()
y = df.select(TARGET).to_pandas().values.ravel()
groups = df.select('repo_name').to_pandas().values.ravel()

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(y):,}")
print(f"Unique repos: {len(np.unique(groups)):,}")

# === Step 5: Define Optuna Objective ===
def objective(trial):
    """Optuna objective function with GroupKFold CV"""
    
    # Suggest hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        
        # Tunable parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': 2000,
    }
    
    # GroupKFold cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        cv_scores.append(rmse)
    
    # Return mean CV score
    mean_rmse = np.mean(cv_scores)
    return mean_rmse

# === Step 6: Run Optimization ===
print("\n" + "="*80)
print("STARTING HYPERPARAMETER SEARCH")
print("="*80)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# === Step 7: Display Results ===
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)

print(f"\nBest trial: {study.best_trial.number}")
print(f"Best RMSE (CV): {study.best_value:.6f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Save best parameters
best_params_full = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'n_estimators': 2000,
    **study.best_params
}

with open(BEST_PARAMS_PATH, 'w') as f:
    json.dump(best_params_full, f, indent=2)

print(f"\n💾 Best parameters saved to '{BEST_PARAMS_PATH}'")

# === Step 8: Visualize Optimization ===
print("\nGenerating optimization plots...")

fig1 = plot_optimization_history(study)
fig1.write_image('optuna_optimization_history.png', width=1200, height=600)
print("  - optuna_optimization_history.png")

fig2 = plot_param_importances(study)
fig2.write_image('optuna_param_importances.png', width=1200, height=600)
print("  - optuna_param_importances.png")

# === Step 9: Retrain with Best Parameters on Full Data ===
print("\n" + "="*80)
print("RETRAINING WITH BEST PARAMETERS")
print("="*80)

# Use 80/20 repo split for final validation
all_repos = np.unique(groups)
np.random.seed(RANDOM_SEED)
np.random.shuffle(all_repos)

split_idx = int(len(all_repos) * 0.8)
train_repos = set(all_repos[:split_idx])
val_repos = set(all_repos[split_idx:])

train_mask = np.isin(groups, list(train_repos))
val_mask = np.isin(groups, list(val_repos))

X_train, X_val = X[train_mask], X[val_mask]
y_train, y_val = y[train_mask], y[val_mask]

print(f"Train samples: {len(y_train):,}")
print(f"Val samples: {len(y_val):,}")

# Train final model
final_model = lgb.LGBMRegressor(**best_params_full)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['Train', 'Valid'],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100)
    ]
)

# Save model
joblib.dump(final_model, MODEL_PATH)
print(f"\n💾 Tuned model saved to '{MODEL_PATH}'")

# === Step 10: Final Evaluation ===
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

y_pred = final_model.predict(X_val)
rmse_growth = root_mean_squared_error(y_val, y_pred)
mae_growth = mean_absolute_error(y_val, y_pred)

print(f"\nGrowth Rate Metrics:")
print(f"  RMSE: {rmse_growth:.6f}")
print(f"  MAE: {mae_growth:.6f}")

# Reconstruct to star space
df_val = df.filter(pl.col('repo_name').is_in(val_repos))
current_stars = df_val.select('total_stars').to_pandas().values.ravel()
actual_future_stars = df_val.select(LEAD_COL).to_pandas().values.ravel()
pred_future_stars = current_stars * (1 + y_pred)

rmse_stars = root_mean_squared_error(actual_future_stars, pred_future_stars)
mae_stars = mean_absolute_error(actual_future_stars, pred_future_stars)

mask = actual_future_stars > 0
mape = (np.abs(actual_future_stars[mask] - pred_future_stars[mask]) / actual_future_stars[mask] * 100).mean()

print(f"\nStar Prediction Metrics:")
print(f"  RMSE: {rmse_stars:.2f} stars")
print(f"  MAE: {mae_stars:.2f} stars")
print(f"  MAPE: {mape:.2f}%")

# === Step 11: Feature Importance ===
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Top 20)")
print("="*80)

importance_df = pl.DataFrame({
    'feature': final_model.feature_name_,
    'importance': final_model.feature_importances_
}).sort('importance', descending=True)

print(importance_df.head(20))

plt.figure(figsize=(10, 8))
lgb.plot_importance(final_model, max_num_features=20, importance_type='gain')
plt.title(f"Feature Importance: {FORECAST_HORIZON}-Day Tuned Model")
plt.tight_layout()
plt.savefig(f'feature_importance_lead{FORECAST_HORIZON}d_tuned.png', dpi=150)
print(f"\n📊 Feature importance plot saved")
plt.close()

print("\n" + "="*80)
print("✅ TUNING COMPLETE!")
print("="*80)
print(f"\nNext steps:")
print(f"  1. Review optimization plots in current directory")
print(f"  2. Run benchmark_lead180d_stationary.py with MODEL_PATH = '{MODEL_PATH}'")
print(f"  3. Compare tuned model vs baseline performance")
