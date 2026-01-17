# Developer Context & Project State: GitHub Star Forecasting

**Last Updated:** Jan 2026
**Project Goal:** Forecast cumulative GitHub stars 90+ days out.
**Primary Tech Stack:** Python, Polars (Data Processing), LightGBM (Modeling).

---

## 1. The Core Problem & Solution

We are predicting `total_stars`, which is a **cumulative, non-stationary** metric.

* **The Trap (Naive Model):** If you feed `total_stars_lag_1d` into a model, it learns . This yields great 1-day accuracy (RMSE) but fails completely at forecasting trends (flatlines in the future).
* **The Solution (Velocity Model):** We predict the **First Derivative** (Rate of Change) in Log Space.


* This forces the model to look at *activity features* (commits, PRs) to determine if growth is accelerating or decelerating.



## 2. Data Pipeline Architecture

We use a **Hybrid Polars/Pandas** approach to handle the large dataset (10 years/repo) on resource-constrained hardware.

* **Step 1: Lazy Loading (Polars)**
* We use `pl.scan_parquet()` to define transformations lazily.
* Sorting and Splitting are done *before* materializing data to RAM.
* *Constraint:* Do not load the full 10GB+ dataset into Pandas at once.


* **Step 2: Feature Engineering (Log-Space)**
* **Previous Mistake:** We used `MaxAbsScaler` on top of `log1p`.
* *Result:* Massive drift. A 0.01 change meant "1 star" for small repos but "100 stars" for large repos.


* **Current Rule:** **NO SCALERS.** Use `log1p` raw values only. LightGBM handles the magnitude differences via tree splits.
* **Input Features:** `_log1p` of lags (1d, 7d) and rolling means (7d, 30d).



## 3. Strict Feature Selection Rules

To prevent **Data Leakage**, the following columns are **BANNED** from `X` (Input Features):

| Column Type | Status | Reason |
| --- | --- | --- |
| `repo_name`, `day` | **DROP** | Identifiers. |
| `total_stars_log1p` (Current) | **DROP** | The Target. |
| `total_stars_lag_1d_log1p` | **DROP** | **Leakage Source.** If included, the model ignores activity features and just copies the previous value. |
| `total_stars_daily_change` | **DROP** | Directly correlated with the target; prevents learning the *cause* of the change. |

**Allowed Features:**

* Lags of *Activity* (Commits, PRs, Issues, Forks).
* Rolling Statistics of Activity.
* Rolling Statistics of Stars (e.g., `rolling_std_7d`) are okay, as they describe volatility, not absolute count.

## 4. Current File Structure

* `processed_github_features.parquet`: Raw source (contains `_log1p` cols).
* `train_dataset.parquet` / `val_dataset.parquet`: Chronologically split (First 80% time vs Last 20%).
* `lgbm_github_log_only.pkl`: The trained LightGBM model artifact.
* `train_model.py`: Training script (Handles the Target diff calculation).
* `benchmark_horizon.py`: The "Drift Test" script.

## 5. Benchmarking & Evaluation

We use two distinct evaluation methods:

1. **Reconstruction RMSE (Validation):**
* Formula: 
* *Purpose:* Checks one-step-ahead accuracy.


2. **Recursive Drift Test (Horizon):**
* We feed predictions back into the model for 90 days.
* *Pass Criteria:* The forecast curve must follow the *shape* of the actual history, not just the level.
* *Fail Condition:* The curve diverges linearly (e.g., shoots up to infinity or crashes to zero) due to bias accumulation.



## 6. Immediate Next Steps

1. **Data Verification:** Ensure the new `processed_github_features.parquet` has `total_commits_log1p`, `total_stars_log1p`, etc.
2. **Retrain:** Run `train_model.py`.
* *Check:* Verify that feature importance shows `commits`, `forks`, or `issues` at the top, NOT `stars`.


3. **Visualization:** Run `benchmark_horizon.py` on 'facebook/react' (or similar large repo).
* *Goal:* Drift < 5% after 30 days.



---

**Technical Note on Polars:**
If you need to debug the data, use:

```python
import polars as pl
lf = pl.scan_parquet('train_dataset.parquet')
print(lf.head(5).collect())

```

Do not use `pd.read_parquet()` on the full file.