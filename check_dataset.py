import polars as pl

# Load the new dataset
lf = pl.scan_parquet('github_features_log1pd.parquet')
schema = lf.collect_schema()

print(f"Total columns: {len(schema.names())}")
print(f"Total rows: {lf.select(pl.len()).collect().item()}")

# Find log1p columns
log_cols = [c for c in schema.names() if 'log1p' in c]
print(f"\nlog1p columns ({len(log_cols)}):")
for c in sorted(log_cols):
    print(f"  - {c}")

# Show first few rows
print("\nFirst 3 rows (selected columns):")
sample_cols = ['repo_name', 'day'] + log_cols[:5]
print(lf.select(sample_cols).head(3).collect())
