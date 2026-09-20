"""
Extract indoor/outdoor temperature pairs from EFUS 2017 temp CSVs,
using bedroom sensors only (BED1 > BED2 > BED3 priority).

Observations where no bedroom sensor is present are dropped entirely —
unlike efus_extract.py which falls back to LR/HALL.

Output: efus_indoor_outdoor_bedroom.parquet
Columns: CaseID, hour, T_in (bedroom), Location (BED1/BED2/BED3), T_out
"""

import pandas as pd
import numpy as np
import os

import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

TEMPDIR  = "/home/teaching/heating/test_data/ukda_9434_csv_r/csv/tempdata_csv/"
OUT_PATH = "/home/teaching/heating/efus_indoor_outdoor_bedroom.parquet"

BED_PRIORITY  = ["BED1", "BED2", "BED3"]
SUMMER_MONTHS = [5, 6, 7, 8, 9, 10]

files = sorted([f for f in os.listdir(TEMPDIR) if f.endswith(".csv") and not f.endswith("Identifier")])
print(f"Processing {len(files)} files...")


def process_file(fpath: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(
        fpath,
        chunksize=200_000,
        usecols=["OB_TIME", "value", "Location", "CaseID"],
        parse_dates=["OB_TIME"],
    ):
        chunk = chunk[chunk["OB_TIME"].dt.month.isin(SUMMER_MONTHS)]
        if chunk.empty:
            continue

        chunk["hour"] = chunk["OB_TIME"].dt.round("h")

        ext = (chunk[chunk["Location"] == "EXT"][["CaseID", "hour", "value"]]
               .rename(columns={"value": "T_out"}))
        bed = chunk[chunk["Location"].isin(BED_PRIORITY)].copy()

        if ext.empty or bed.empty:
            continue

        bed["rank"] = bed["Location"].map({loc: i for i, loc in enumerate(BED_PRIORITY)})
        bed = (bed.sort_values("rank")
                  .groupby(["CaseID", "hour"])
                  .first()
                  .reset_index()
                  .rename(columns={"value": "T_in"})
               [["CaseID", "hour", "T_in", "Location"]])

        ext_hr = ext.groupby(["CaseID", "hour"])["T_out"].mean().reset_index()

        merged = bed.merge(ext_hr, on=["CaseID", "hour"], how="inner")
        parts.append(merged)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


all_parts = []
for i, fname in enumerate(files):
    fpath = os.path.join(TEMPDIR, fname)
    print(f"  [{i+1}/{len(files)}] {fname}", end="", flush=True)
    part = process_file(fpath)
    if not part.empty:
        all_parts.append(part)
        print(f" → {len(part):,} rows")
    else:
        print(" → (no bedroom/summer data)")

print("\nConcatenating all parts...")
df = pd.concat(all_parts, ignore_index=True)

# Across-file dedup: keep highest-priority bedroom sensor per (CaseID, hour)
df["rank"] = df["Location"].map({loc: i for i, loc in enumerate(BED_PRIORITY)})
df = (df.sort_values("rank")
        .drop_duplicates(subset=["CaseID", "hour"])
        .drop(columns="rank")
        .sort_values(["CaseID", "hour"])
        .reset_index(drop=True))

print(f"\nFinal shape: {df.shape}")
print(f"CaseIDs with bedroom data: {df['CaseID'].nunique()}")
print(f"Time range: {df['hour'].min()} to {df['hour'].max()}")
print(f"T_in range: {df['T_in'].min():.1f} to {df['T_in'].max():.1f}")
print(f"T_out range: {df['T_out'].min():.1f} to {df['T_out'].max():.1f}")
print(f"Bedroom sensors used:\n{df['Location'].value_counts()}")

df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
