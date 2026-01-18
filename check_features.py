import polars as pl

df = pl.read_parquet('train_dataset.parquet')
cols = df.columns

# Find all star-related columns
star_cols = [c for c in cols if 'star' in c.lower()]

print('All star-related columns in dataset:')
for c in sorted(star_cols):
    print(f'  {c}')

print(f'\nTotal: {len(star_cols)} star columns')

# Check if there are non-log1p versions
non_log_stars = [c for c in star_cols if 'log1p' not in c]
print(f'\nNon-log1p star columns ({len(non_log_stars)}):')
for c in non_log_stars:
    print(f'  {c}')
