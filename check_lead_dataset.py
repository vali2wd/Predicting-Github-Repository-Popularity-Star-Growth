import polars as pl

# Load the lead dataset
lf = pl.scan_parquet('github_features_lead.parquet')
schema = lf.collect_schema()

print(f"Total columns: {len(schema.names())}")

# Categorize columns
lead_cols = [c for c in schema.names() if 'lead' in c]
log1p_cols = [c for c in schema.names() if 'log1p' in c and 'lead' not in c]
regular_cols = [c for c in schema.names() if 'lead' not in c and 'log1p' not in c]

print(f"\nLead columns ({len(lead_cols)}):")
for c in sorted(lead_cols)[:20]:
    print(f"  {c}")
if len(lead_cols) > 20:
    print(f"  ... and {len(lead_cols) - 20} more")

print(f"\nLog1p columns (non-lead) ({len(log1p_cols)}):")
for c in sorted(log1p_cols)[:10]:
    print(f"  {c}")

print(f"\nSample of first 3 rows:")
sample = lf.select(['repo_name', 'day', 'total_stars_log1p'] + 
                   [c for c in lead_cols if 'total_stars' in c][:3]).head(3).collect()
print(sample)
