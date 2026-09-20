"""
Build analysis datasets for overheating impact profiles.

Outputs (saved to analysis/):
  daily_overheating.parquet       -- one row per dwelling × calendar day
  dwelling_overheating_summary.parquet  -- one row per dwelling (May-Sep totals)

See overheating_analysis.md for full specification.
"""

import argparse
import os
import numpy as np
import pandas as pd

# ── Room configuration ────────────────────────────────────────────────────────
ROOM_CONFIG = {
    "livingroom": {
        "parquet":    "efus_indoor_outdoor_livingroom.parquet",
        "occ_hours":  list(range(7, 22)),               # 07:00–21:00, 15 h/day
        "crit_daily": 0.03,                             # >3% of day's occ hours
        "crit_seas":  3.0,                              # >3% of season's occ hours
    },
    "bedroom": {
        "parquet":    "efus_indoor_outdoor_bedroom.parquet",
        "occ_hours":  list(range(22, 24)) + list(range(0, 7)),  # 22:00–06:59, 9 h/day
        "crit_daily": 0.01,                             # >1% of day's occ hours
        "crit_seas":  1.0,                              # >1% of season's occ hours
    },
}

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--months", nargs="+", type=int, default=list(range(5, 10)),
                    help="Calendar months to include (default: 5-9)")
parser.add_argument("--subdir", type=str, default="",
                    help="Time-based subdirectory suffix, e.g. 'august'")
parser.add_argument("--room", type=str, default="livingroom",
                    choices=list(ROOM_CONFIG.keys()),
                    help="Room type (default: livingroom)")
parser.add_argument("--region", type=str, default="london",
                    choices=["london", "southwest"],
                    help="EFUS region filter (default: london)")
args = parser.parse_args()

MONTHS    = args.months
ROOM      = args.room
RCFG      = ROOM_CONFIG[ROOM]
OCC_HOURS = RCFG["occ_hours"]
REGION    = args.region

REGION_GOR = {"london": 7, "southwest": 8}[REGION]

# Analysis dir: region prefix (empty for london, region name for others) + room/subdir
_room_sub  = f"{ROOM}/{args.subdir}" if args.subdir else ROOM
_reg_pre   = "" if REGION == "london" else REGION
_subdir    = "/".join(filter(None, [_reg_pre, _room_sub]))
ANALYSIS_DIR = os.path.join(ROOT, "analysis", _subdir) if _subdir else os.path.join(ROOT, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

THRESHOLDS = [26, 27, 28]

# ── Load and filter ───────────────────────────────────────────────────────────
print(f"Loading temperature data ({ROOM})...")
df = pd.read_parquet(os.path.join(ROOT, RCFG["parquet"]))

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

# Region filter + requested months
df = df[df["gorEHS_efus"] == REGION_GOR].copy()
df = df[df["hour"].dt.month.isin(MONTHS)].copy()

# Minimum coverage filter (≥20 obs)
sizes   = df.groupby("CaseID").size()
df      = df[df["CaseID"].isin(sizes[sizes >= 20].index)].reset_index(drop=True)

df["date"]        = df["hour"].dt.floor("D")
df["hour_of_day"] = df["hour"].dt.hour
df["occupied"]    = df["hour_of_day"].isin(OCC_HOURS)

n_dwellings = df["CaseID"].nunique()
print(f"London May-Sep: {len(df):,} obs, {n_dwellings} dwellings")

# ── Building characteristics lookup (one row per dwelling) ────────────────────
BLDG_CHAR_COLS = [
    "CaseID", "dwtype_efus", "dwage_efus", "WallType2x_efus",
    "InsulatedWalls_efus", "FullyDblGlz_efus", "floor6x_efus",
    "EPceeb12e_efus", "AnyCooling",
]
dwelling_chars = df[BLDG_CHAR_COLS].drop_duplicates("CaseID").set_index("CaseID")

# ── Step 1: daily T_in statistics ─────────────────────────────────────────────
print("Computing daily T_in statistics...")

daily_tin = (
    df.groupby(["CaseID", "date"])
    .agg(
        daily_max_Tin=("T_in", "max"),
        daily_min_Tin=("T_in", "min"),
        daily_mean_Tin=("T_in", "mean"),
    )
    .reset_index()
)

# ── Step 2: daily T_out statistics (outdoor climate covariates) ───────────────
print("Computing daily T_out statistics...")

daily_tout = (
    df.groupby(["CaseID", "date"])
    .agg(
        daily_max_Tout=("T_out", "max"),
        daily_min_Tout=("T_out", "min"),
        daily_mean_Tout=("T_out", "mean"),
    )
    .reset_index()
)

# T_2DMMT: 2-day running mean of daily outdoor max (per dwelling)
daily_tout = daily_tout.sort_values(["CaseID", "date"])
daily_tout["T_2DMMT"] = (
    daily_tout
    .groupby("CaseID")["daily_max_Tout"]
    .transform(lambda x: x.rolling(2, min_periods=1).mean())
)

# ── Step 3: daily degree-hours and exceedance (occupied hours only) ───────────
print("Computing daily degree-hours and exceedance proportions...")

df_occ = df[df["occupied"]].copy()

dh_parts = []
for thr in THRESHOLDS:
    df_occ[f"exceed_{thr}"]  = (df_occ["T_in"] >= thr).astype(int)
    df_occ[f"dh_{thr}"]      = (df_occ["T_in"] - thr).clip(lower=0)

agg_dict = {"T_in": "count"}   # count occupied hours
for thr in THRESHOLDS:
    agg_dict[f"exceed_{thr}"] = "sum"
    agg_dict[f"dh_{thr}"]     = "sum"

daily_occ = (
    df_occ
    .groupby(["CaseID", "date"])
    .agg(**{
        "occ_hours":                  ("T_in",            "count"),
        **{f"exceed_count_{t}":       (f"exceed_{t}",     "sum")   for t in THRESHOLDS},
        **{f"daily_dh_{t}":           (f"dh_{t}",         "sum")   for t in THRESHOLDS},
    })
    .reset_index()
)

for thr in THRESHOLDS:
    daily_occ[f"daily_exceed_prop_{thr}"] = (
        daily_occ[f"exceed_count_{thr}"] / daily_occ["occ_hours"]
    )
    daily_occ[f"daily_overheat_{thr}"] = (
        (daily_occ[f"daily_exceed_prop_{thr}"] > RCFG["crit_daily"]).astype(int)
    )

daily_occ = daily_occ.drop(
    columns=[f"exceed_count_{t}" for t in THRESHOLDS]
)

# ── Step 4: assemble daily dataset ────────────────────────────────────────────
print("Assembling daily dataset...")

daily = (
    daily_tin
    .merge(daily_tout, on=["CaseID", "date"])
    .merge(daily_occ,  on=["CaseID", "date"])
)

# Integer day index within each dwelling's series (needed for AR(1) in nlme)
daily = daily.sort_values(["CaseID", "date"])
daily["day_index"] = (
    daily.groupby("CaseID").cumcount()
)

# Join building characteristics
daily = daily.join(dwelling_chars, on="CaseID")

# Friendly binary label for CavityWall (used in models)
daily["CavityWall"] = (daily["WallType2x_efus"] == 2).astype(int)

col_order = (
    ["CaseID", "date", "day_index"]
    + ["daily_max_Tin", "daily_min_Tin", "daily_mean_Tin"]
    + [f"daily_dh_{t}"           for t in THRESHOLDS]
    + [f"daily_exceed_prop_{t}"  for t in THRESHOLDS]
    + [f"daily_overheat_{t}"         for t in THRESHOLDS]
    + ["occ_hours"]
    + ["T_2DMMT", "daily_max_Tout", "daily_min_Tout", "daily_mean_Tout"]
    + [c for c in dwelling_chars.columns] + ["CavityWall"]
)
daily = daily[col_order]

out_path = os.path.join(ANALYSIS_DIR, "daily_overheating.parquet")
daily.to_parquet(out_path, index=False)
print(f"Saved: {out_path}  ({len(daily):,} rows, {daily['CaseID'].nunique()} dwellings)")

# ── Step 5: dwelling-level May-Sep summary ────────────────────────────────────
print("Building dwelling-level May-Sep summary...")

# Aggregate daily metrics to dwelling level
summary_agg = (
    daily.groupby("CaseID")
    .agg(
        mean_daily_max_Tin   =("daily_max_Tin",  "mean"),
        mean_daily_min_Tin   =("daily_min_Tin",  "mean"),
        n_days               =("date",           "count"),
        mean_T_2DMMT         =("T_2DMMT",        "mean"),
        **{f"total_dh_{t}":     (f"daily_dh_{t}",          "sum")  for t in THRESHOLDS},
        **{f"overheat_days_{t}":    (f"daily_overheat_{t}",         "sum")  for t in THRESHOLDS},
    )
    .reset_index()
)

# Exceedance % over the whole month: use occupied-hour totals from df_occ
monthly_occ = (
    df_occ
    .assign(**{f"exceed_{t}": (df_occ["T_in"] >= t).astype(int) for t in THRESHOLDS})
    .groupby("CaseID")
    .agg(
        total_occ_hours=("T_in", "count"),
        **{f"exceed_total_{t}": (f"exceed_{t}", "sum") for t in THRESHOLDS},
    )
    .reset_index()
)
for thr in THRESHOLDS:
    monthly_occ[f"exceed_pct_{thr}"] = (
        100 * monthly_occ[f"exceed_total_{thr}"] / monthly_occ["total_occ_hours"]
    )
    monthly_occ[f"season_overheat_{thr}"] = (
        (monthly_occ[f"exceed_pct_{thr}"] > RCFG["crit_seas"]).astype(int)
    )
monthly_occ = monthly_occ.drop(columns=[f"exceed_total_{t}" for t in THRESHOLDS])

summary = summary_agg.merge(monthly_occ, on="CaseID")

# Log-transform degree-hours (zero-inflated, right-skewed)
for thr in THRESHOLDS:
    summary[f"log_dh_{thr}"] = np.log1p(summary[f"total_dh_{thr}"])

# Join building characteristics
summary = summary.join(dwelling_chars, on="CaseID")
summary["CavityWall"] = (summary["WallType2x_efus"] == 2).astype(int)

out_path = os.path.join(ANALYSIS_DIR, "dwelling_overheating_summary.parquet")
summary.to_parquet(out_path, index=False)
print(f"Saved: {out_path}  ({len(summary)} dwellings)")

# ── Step 6: print summary statistics ─────────────────────────────────────────
print("\n── Daily dataset columns ──────────────────────────────────────────────")
print(daily.dtypes.to_string())

print("\n── Dwelling summary: overheating metrics ──────────────────────────────")
metric_cols = (
    ["mean_daily_max_Tin", "mean_daily_min_Tin"]
    + [f"total_dh_{t}" for t in THRESHOLDS]
    + [f"exceed_pct_{t}" for t in THRESHOLDS]
    + [f"season_overheat_{t}" for t in THRESHOLDS]
)
print(summary[metric_cols].describe().round(2).to_string())

print("\n── Overheating criterion rates ──────────────────────────────────────────────────")
for thr in THRESHOLDS:
    n_fail = summary[f"season_overheat_{thr}"].sum()
    pct    = 100 * n_fail / len(summary)
    print(f"  {thr}°C: {n_fail}/{len(summary)} dwellings ({pct:.1f}%) classified as overheating")

print("\nDone.")
