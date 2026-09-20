"""
Build analysis dataset for the mean-temperature night criterion (bedroom only).

Mean-temperature bedroom criterion: a bedroom overheats if the mean indoor temperature
during 22:00–06:59 exceeds the threshold on more than 7 nights in May–Sep.

This script computes per-night mean T_in and flags night_mean_overheat_{26,27}.
Predictor for GEE is T_2DMMT (2-day running mean outdoor Tmax), aligned to
the evening date (same convention as below).

Night assignment: a night 22:00–06:59 is assigned to the EVENING date
(e.g. night starting 2017-08-03 22:00 → date 2017-08-03).

Outputs (to analysis/<region_prefix>/bedroom/mean_criterion/):
  daily_overheating.parquet       -- one row per dwelling × evening date
  dwelling_overheating_summary.parquet  -- one row per dwelling

Usage:
  python python/efus_overheating_datasets_meannight.py --region london
  python python/efus_overheating_datasets_meannight.py --region southwest
"""

import argparse
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--region", type=str, default="london",
                    choices=["london", "southwest"])
args = parser.parse_args()

REGION     = args.region
REGION_GOR = {"london": 7, "southwest": 8}[REGION]
MONTHS     = list(range(5, 10))   # May–Sep
THRESHOLDS = [26, 27]

# Night hours: 22, 23 (evening date), 0–6 (next calendar date)
NIGHT_HOURS_EVENING = [22, 23]        # hours on the evening date
NIGHT_HOURS_MORNING = list(range(7))  # hours 0–6 on the following morning date

_reg_pre = "" if REGION == "london" else REGION
_subdir  = "/".join(filter(None, [_reg_pre, "bedroom/mean_criterion"]))
ANALYSIS_DIR = os.path.join(ROOT, "analysis", _subdir)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Load parquet ──────────────────────────────────────────────────────────────
print(f"Loading bedroom temperature data ({REGION})...")
df = pd.read_parquet(os.path.join(ROOT, "efus_indoor_outdoor_bedroom.parquet"))
df = df[df["T_in"].between(-10, 40) & df["T_out"].between(-10, 40)].copy()

BLDG_COLS = [
    "CaseID", "dwtype_efus", "dwage_efus", "WallType2x_efus",
    "InsulatedWalls_efus", "FullyDblGlz_efus", "floor6x_efus",
    "EPceeb12e_efus", "gorEHS_efus", "AnyCooling",
]
bldg = pd.read_csv(
    os.path.join(ROOT,
        "test_data/ukda_9434_csv_r/csv/selected_interview_responses_caseid.csv"),
    usecols=BLDG_COLS,
)
df = df.merge(bldg, on="CaseID", how="left").dropna(subset=BLDG_COLS[1:])
for c in BLDG_COLS[1:]:
    df[c] = df[c].astype(int)

df = df[df["gorEHS_efus"] == REGION_GOR].copy()

# Minimum coverage filter
sizes = df.groupby("CaseID").size()
df    = df[df["CaseID"].isin(sizes[sizes >= 20].index)].reset_index(drop=True)

df["date"]        = df["hour"].dt.floor("D")
df["hour_of_day"] = df["hour"].dt.hour
df["month"]       = df["hour"].dt.month

# Building chars lookup
BLDG_CHAR_COLS = [
    "CaseID", "dwtype_efus", "dwage_efus", "WallType2x_efus",
    "InsulatedWalls_efus", "FullyDblGlz_efus", "floor6x_efus",
    "EPceeb12e_efus", "AnyCooling",
]
dwelling_chars = df[BLDG_CHAR_COLS].drop_duplicates("CaseID").set_index("CaseID")

# ── Assign each hour to an evening date ───────────────────────────────────────
# Hours 22–23 on date D → evening_date = D
# Hours 0–6 on date D   → evening_date = D - 1 day
df["evening_date"] = df["date"]
morning_mask = df["hour_of_day"].isin(NIGHT_HOURS_MORNING)
df.loc[morning_mask, "evening_date"] = df.loc[morning_mask, "date"] - pd.Timedelta(days=1)

# Keep only night hours (22, 23, 0–6)
night_mask = df["hour_of_day"].isin(NIGHT_HOURS_EVENING + NIGHT_HOURS_MORNING)
df_night   = df[night_mask].copy()

# Filter to evenings whose month is in May–Sep (by evening_date month)
df_night["ev_month"] = df_night["evening_date"].dt.month
df_night = df_night[df_night["ev_month"].isin(MONTHS)].copy()

n_dwellings = df_night["CaseID"].nunique()
print(f"{REGION.capitalize()} May-Sep nights: {len(df_night):,} obs, {n_dwellings} dwellings")

# ── Step 1: per-night mean T_in ───────────────────────────────────────────────
print("Computing per-night mean T_in...")
night_tin = (
    df_night.groupby(["CaseID", "evening_date"])
    .agg(
        mean_tin_night=("T_in", "mean"),
        n_night_hours =("T_in", "count"),
    )
    .reset_index()
    .rename(columns={"evening_date": "date"})
)

for thr in THRESHOLDS:
    night_tin[f"night_mean_overheat_{thr}"] = (night_tin["mean_tin_night"] >= thr).astype(int)

# ── Step 2: T_2DMMT from daytime outdoor data (May–Sep days) ─────────────────
print("Computing T_2DMMT...")
df_day = df[df["month"].isin(MONTHS)].copy()
daily_tout = (
    df_day.groupby(["CaseID", "date"])
    .agg(daily_max_Tout=("T_out", "max"))
    .reset_index()
)
daily_tout = daily_tout.sort_values(["CaseID", "date"])
daily_tout["T_2DMMT"] = (
    daily_tout
    .groupby("CaseID")["daily_max_Tout"]
    .transform(lambda x: x.rolling(2, min_periods=1).mean())
)

# ── Step 3: assemble nightly dataset ─────────────────────────────────────────
print("Assembling nightly dataset...")
daily = night_tin.merge(daily_tout[["CaseID", "date", "T_2DMMT", "daily_max_Tout"]],
                        on=["CaseID", "date"], how="inner")

daily = daily.dropna(subset=["T_2DMMT"]).copy()
daily = daily.sort_values(["CaseID", "date"])
daily["day_index"] = daily.groupby("CaseID").cumcount()
daily = daily.join(dwelling_chars, on="CaseID")
daily["CavityWall"] = (daily["WallType2x_efus"] == 2).astype(int)

col_order = (
    ["CaseID", "date", "day_index"]
    + ["mean_tin_night", "n_night_hours"]
    + [f"night_mean_overheat_{t}" for t in THRESHOLDS]
    + ["T_2DMMT", "daily_max_Tout"]
    + list(dwelling_chars.columns) + ["CavityWall"]
)
daily = daily[col_order]

out_path = os.path.join(ANALYSIS_DIR, "daily_overheating.parquet")
daily.to_parquet(out_path, index=False)
print(f"Saved: {out_path}  ({len(daily):,} rows, {daily['CaseID'].nunique()} dwellings)")

# ── Step 4: dwelling-level May-Sep summary ────────────────────────────────────
print("Building dwelling-level summary...")
summary = (
    daily.groupby("CaseID")
    .agg(
        n_nights             =("date", "count"),
        mean_tin_night_mean  =("mean_tin_night", "mean"),
        mean_T_2DMMT         =("T_2DMMT", "mean"),
        **{f"nights_above_{t}": (f"night_mean_overheat_{t}", "sum") for t in THRESHOLDS},
    )
    .reset_index()
)

# Seasonal overheating: >7 nights above threshold
OVERHEAT_NIGHTS_THRESHOLD = 7
for thr in THRESHOLDS:
    summary[f"season_overheat_{thr}"] = (
        (summary[f"nights_above_{thr}"] > OVERHEAT_NIGHTS_THRESHOLD).astype(int)
    )

summary = summary.join(dwelling_chars, on="CaseID")
summary["CavityWall"] = (summary["WallType2x_efus"] == 2).astype(int)

out_path = os.path.join(ANALYSIS_DIR, "dwelling_overheating_summary.parquet")
summary.to_parquet(out_path, index=False)
print(f"Saved: {out_path}  ({len(summary)} dwellings)")

print("\n── Seasonal overheating rates (mean-temperature criterion) ─────────────")
for thr in THRESHOLDS:
    n_fail = summary[f"season_overheat_{thr}"].sum()
    pct    = 100 * n_fail / len(summary)
    print(f"  {thr}°C (>7 nights): {n_fail}/{len(summary)} dwellings ({pct:.1f}%)")

print("\nDone.")
