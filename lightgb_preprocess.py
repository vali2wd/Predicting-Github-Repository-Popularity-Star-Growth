import polars as pl

# Config
INPUT_FILE = 'processed_github_features.parquet'
TRAIN_OUTPUT = 'train_dataset.parquet'
VAL_OUTPUT = 'val_dataset.parquet'
TIME_COL = 'day'

# 1. Scan the raw file
lf = pl.scan_parquet(INPUT_FILE)

# 2. Cast and prepare time column
# We cast to Date to ensure correct sorting and splitting
lf = lf.with_columns(pl.col(TIME_COL).cast(pl.Date))

# 3. Calculate Split Date
# We fetch only the unique dates to determine the 80% cutoff
# This is very memory efficient.
print("Calculating time split...")
dates = (
    lf.select(TIME_COL)
    .unique()
    .collect()
    .get_column(TIME_COL)
    .sort()
)
split_idx = int(len(dates) * 0.80)
split_date = dates[split_idx]
print(f"Data will be split at: {split_date}")

# 4. Sort Globally
# Sorting implies the data is ready for time-series modeling (e.g., rolling windows)
# Note: Sorting is expensive, so we define it here to happen once before saving.
lf_sorted = lf.sort(TIME_COL)

# 5. Define Splits (Lazy)
# We keep repo_name and day in the file for debugging, 
# but we will drop them during training.
train_lf = lf_sorted.filter(pl.col(TIME_COL) < split_date)
val_lf = lf_sorted.filter(pl.col(TIME_COL) >= split_date)

# 6. Stream to Disk (Sink)
# sink_parquet processes the query graph and writes to disk in chunks.
# It does NOT load the whole dataframe into RAM.
print(f"Streaming {TRAIN_OUTPUT} to disk...")
train_lf.sink_parquet(TRAIN_OUTPUT)

print(f"Streaming {VAL_OUTPUT} to disk...")
val_lf.sink_parquet(VAL_OUTPUT)

print("Processing complete. Files saved.")