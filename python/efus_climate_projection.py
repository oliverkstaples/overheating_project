"""
Project expected % of occupied hours exceeding threshold under UKCP18 RCP8.5
for London/Southwest dwellings by EPC band.

Approach
--------
Apply the GEE model directly to GCM daily T_2DMMT values for the fixed-threshold overheating
assessment season, with a per-member additive mean bias correction plus a
cross-scale correction to align the GCM and GEE centring scales.

The GEE model was fitted with:
    T_2DMMT_c = T_2DMMT_obs - T2DMMT_OBS_MEAN      (centred on observed mean)

The GCM bias correction uses:
    t2c(d) = T_2DMMT_gcm(d) - GCM_baseline(member) - correction

where:
    GCM_baseline(m) = member m's SEASON_MONTHS mean T_2DMMT over 1981-2010
    correction      = T2DMMT_OBS_MEAN - mean(GCM_baseline over all members)

The correction bridges the gap between the GCM's historical baseline and the
EFUS observed mean for the same season.  Without it, GCM t2c=0 would correspond
to pre-warming conditions while the GEE model expects t2c=0 = the EFUS season mean.

After correction:
  - GCM 1981-2010 period: mean t2c ~ -correction (pre-EFUS climate)
  - GCM ~EFUS year:       mean t2c ~ 0 (aligns with observed validation)
  - GCM future:           t2c = warming above EFUS baseline

This is additive mean bias correction: only the mean offset is removed.
Variance, distributional shape, and extremes are taken directly from the GCM.

Season months
-------------
May-Sep (4month runs, --subdir ""):  SEASON_MONTHS = [5,6,7,8,9]
August only (1month runs, --subdir august): SEASON_MONTHS = [8]

T_2DMMT is computed identically to the dataset builder: rolling(2, min_periods=1)
over the season days only, so the first day of each season uses itself alone.

Dwelling profile
----------------
The exceedance-proportion figure (climate_projection.svg) shows the metric two ways
per EPC band, both at the 26 degC threshold:
  Left  panel — modal dwelling: the modal type/age in the EFUS sample, no cooling.
  Right panel — stock average: the per-dwelling prediction (each dwelling's own
                type/age/cooling) averaged over all EFUS dwellings in the EPC band.
Modal values are derived at run time from analysis/daily_overheating.parquet.

Reference period
----------------
1981-2010: each member's own season-months mean T_2DMMT.

Metric
------
The projected quantity is:

    E[seasonal exceedance %] = mean(E[daily_exceed_prop]) x 100

where daily_exceed_prop = fraction of occupied hours on a given day above the
threshold.  This is an expected exceedance percentage, NOT a probability of
fixed-threshold criterion failure.  A line crossing the fixed-threshold criterion threshold means:

    the expected percentage of occupied hours above the threshold exceeds X%

not that there is a given probability that the dwelling exceeds the criterion.

Validation markers
------------------
Two 2018 markers are shown per EPC band:
  Solid diamond   = observed 2018 exceedance % (weighted by occupied hours)
  Hollow diamond  = model-predicted 2018 exceedance % at the observed T_2DMMT
The gap between them indicates model fit at the EFUS 2018 climate state.

Overheating-criterion line
-------------------
Livingroom: 3% (>3% of occupied hours 07:00–21:00 above threshold)
Bedroom:    1% (>1% of occupied hours 21:00–06:00 above threshold)

Extrapolation
-------------
The model is fitted to EFUS 2018 data.  Future GCM T_2DMMT will increasingly
exceed the observed training range.  An extrapolation diagnostic is printed
showing what fraction of future daily t2c values fall outside the training range.

Outputs
-------
  analysis/{room}/[subdir/]climate_projection.csv
  plots/efus2017/{region}_{room}_{period}/impact_profiles/climate_projection.svg
"""

import argparse, os, glob, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import scienceplots  # noqa: F401
from scipy import stats as _scipy_stats

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ── Arguments ─────────────────────────────────────────────────────────────────
REGION_CONFIG = {
    "london":    {"ukcp18_region": 4},
    "southwest": {"ukcp18_region": 9},
}
parser = argparse.ArgumentParser()
parser.add_argument("--room",   type=str, default="livingroom",
                    choices=["livingroom", "bedroom"])
parser.add_argument("--region", type=str, default="london",
                    choices=list(REGION_CONFIG.keys()))
parser.add_argument("--subdir", type=str, default="",
                    help="Time-based subdirectory suffix, e.g. 'august'")
args = parser.parse_args()

ROOM   = args.room
REGION = args.region

_reg_pre  = "" if REGION == "london" else REGION
_room_sub = f"{ROOM}/{args.subdir}" if args.subdir else ROOM
_ana_sub  = "/".join(filter(None, [_reg_pre, _room_sub]))
ANALYSIS  = os.path.join(ROOT, "analysis", _ana_sub)

_month_label = "1month" if args.subdir == "august" else "4month"
OUTDIR = os.path.join(ROOT, "plots", "efus2017",
                      f"{REGION}_{ROOM}_{_month_label}", "impact_profiles")
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use(["science", "nature", "bright"])
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
def dark(c):
    return tuple(np.array(mcolors.to_rgb(c)) * 0.45)

# ── Configuration ─────────────────────────────────────────────────────────────
UKCP18_REGION  = REGION_CONFIG[REGION]["ukcp18_region"]

# Season months must match the --subdir used when fitting the models
if args.subdir == "august":
    SEASON_MONTHS   = [8]
    MIN_SEASON_DAYS = 20    # August has 30-31 days
    SEASON_LABEL    = "August"
else:
    SEASON_MONTHS   = list(range(5, 10))
    MIN_SEASON_DAYS = 100   # May-Sep has 150-153 days
    SEASON_LABEL    = "May--Sep"

# Room-specific overheating-criterion line
CRIT_LINE = 1.0 if ROOM == "bedroom" else 3.0

BASELINE_START = 1981
BASELINE_END   = 2010
HIST_START     = 1981
HIST_END       = 2019
PROJ_START     = 2020
PROJ_END       = 2080   # UKCP18 land-rcm record ends 2080-11-30
SMOOTH_WINDOW  = 6
THRESHOLDS     = [26]
EPC_LABELS     = {1: "C+", 2: "D", 3: "E", 4: "F/G"}
DWTYPE_LABELS  = {1: "Detached", 2: "Semi-detached", 3: "End-terrace",
                  4: "Mid-terrace", 5: "Bungalow", 6: "Flat"}
DWAGE_LABELS   = {1: "pre-1919", 2: "1919–44", 3: "1944–64",
                  4: "1964–80", 5: "1980–90", 6: "1990–2002",
                  7: "post-2002"}

NC_DIR  = os.path.join(ROOT, "test_data", "RCM")
NC_GLOB = os.path.join(NC_DIR, "tasmax_rcp85_land-rcm_uk_region_*_day_*.nc")

# GEE factor reference levels — must match levels= coding in efus_overheating_models.R
GEE_REF_EPC    = 1   # EPC C+
GEE_REF_DWTYPE = 6   # Flat
GEE_REF_DWAGE  = 7   # Post-2002
GEE_REF_COOL   = 0   # No cooling

# ── Load GEE coefficients ─────────────────────────────────────────────────────
gee_coefs       = pd.read_csv(os.path.join(ANALYSIS, "gee_coefs.csv"))
centres         = pd.read_csv(os.path.join(ANALYSIS, "climate_centres.csv"))
T2DMMT_OBS_MEAN = float(centres.loc[
    centres.variable == "T_2DMMT", "centre"].iloc[0])

print(f"GEE centring value (observed {SEASON_LABEL} mean T_2DMMT): "
      f"{T2DMMT_OBS_MEAN:.2f} degC")

# Infer available EPC levels from the fitted model (avoids projecting absent levels,
# e.g. EPC4 may be dropped in Southwest where no F/G dwellings exist in the sample).
# Check all threshold models and assert they have the same EPC levels.
def _epc_levels_for(model_name):
    rows = gee_coefs[gee_coefs.model == model_name]
    non_ref = sorted(
        int(t[3:]) for t in rows.term
        if t.startswith("EPC") and t[3:].isdigit()
    )
    return sorted({GEE_REF_EPC} | set(non_ref))

_epc_by_model = {f"gee_excprop_{thr}": _epc_levels_for(f"gee_excprop_{thr}")
                 for thr in THRESHOLDS}
_unique_epc_sets = set(tuple(v) for v in _epc_by_model.values())
if len(_unique_epc_sets) > 1:
    raise ValueError(
        f"EPC levels differ across threshold models: {_epc_by_model}. "
        f"Check gee_coefs.csv for inconsistent factor level dropping."
    )
EPC_LEVELS = list(_epc_by_model[f"gee_excprop_{THRESHOLDS[0]}"])
print(f"EPC levels in fitted model: {[EPC_LABELS[e] for e in EPC_LEVELS]}")

# ── Load daily data (modal profile + validation points) ───────────────────────
daily_df = pd.read_parquet(os.path.join(ANALYSIS, "daily_overheating.parquet"))

dwelling_chars = (
    daily_df[["CaseID", "dwtype_efus", "dwage_efus"]]
    .drop_duplicates("CaseID")
)
MODAL_DWTYPE = int(dwelling_chars["dwtype_efus"].mode().iloc[0])
MODAL_DWAGE  = int(dwelling_chars["dwage_efus"].mode().iloc[0])
MODAL_COOL   = 0   # no cooling (modal; cooling rare in UK stock)

print(f"Modal dwelling profile: dwtype={MODAL_DWTYPE}, dwage={MODAL_DWAGE}, "
      f"cooling={MODAL_COOL}")

# ── GEE coefficient lookup ────────────────────────────────────────────────────
def gee_coef(model, term, required=True):
    """
    Look up a GEE coefficient from gee_coefs.

    required=True  (default): raises ValueError if the term is absent.
                   Use for intercept, climate slope, and all EPC terms.
    required=False: returns 0.0 if absent (used for dwtype/dwage/cooling, which
                   the R model may have dropped under a fallback formula).
    """
    row = gee_coefs[(gee_coefs.model == model) & (gee_coefs.term == term)]
    if len(row) == 0:
        if required:
            raise ValueError(
                f"Missing required GEE coefficient: model={model!r}, term={term!r}. "
                f"Check gee_coefs.csv or whether a fallback formula was used."
            )
        return 0.0
    if len(row) > 1:
        raise ValueError(
            f"Duplicated GEE coefficient: model={model!r}, term={term!r}"
        )
    return float(row.estimate.iloc[0])

# Warn once at startup if optional terms for the modal profile are absent
# (indicates a fallback formula was used during R model fitting)
print("Checking optional GEE coefficients for modal dwelling profile...")
for _thr in THRESHOLDS:
    _m = f"gee_excprop_{_thr}"
    for _t in [f"dwtype{MODAL_DWTYPE}", f"dwage{MODAL_DWAGE}", "cooling1"]:
        if not len(gee_coefs[(gee_coefs.model == _m) & (gee_coefs.term == _t)]):
            print(f"  NOTE: {_m!r}/{_t!r} absent from fitted model "
                  f"(fallback formula applied during R fitting) → treated as 0.0")

def predict_profile(t2dmmt_c_arr, epc_level, model,
                    dwtype=MODAL_DWTYPE, dwage=MODAL_DWAGE, cooling=MODAL_COOL):
    """
    E[daily_exceed_prop] for a given building profile, vectorised over T_2DMMT_c.
    Reference-level coefficients are zero by definition.
    Non-reference EPC terms are required; non-reference dwtype/dwage/cooling are
    optional (the R model may have dropped them under a fallback formula).
    """
    b0       = gee_coef(model, "(Intercept)",                          required=True)
    b_clim   = gee_coef(model, "T_2DMMT_c",                           required=True)
    b_epc    = 0.0 if epc_level == GEE_REF_EPC    else gee_coef(model, f"EPC{epc_level}",            required=True)
    b_int    = 0.0 if epc_level == GEE_REF_EPC    else gee_coef(model, f"T_2DMMT_c:EPC{epc_level}",  required=True)
    b_dwtype = 0.0 if dwtype    == GEE_REF_DWTYPE else gee_coef(model, f"dwtype{dwtype}",             required=False)
    b_dwage  = 0.0 if dwage     == GEE_REF_DWAGE  else gee_coef(model, f"dwage{dwage}",               required=False)
    b_cool   = 0.0 if cooling   == GEE_REF_COOL   else gee_coef(model, "cooling1",                    required=False)
    eta = (b0 + (b_clim + b_int) * t2dmmt_c_arr
           + b_epc + b_dwtype + b_dwage + b_cool)
    return 1.0 / (1.0 + np.exp(-eta))

def smooth(arr):
    """Centred rolling mean for plotting (module-level so both figures share it)."""
    return (pd.Series(arr)
            .rolling(SMOOTH_WINDOW, center=True, min_periods=5)
            .mean().values)

# ── Per-dwelling profile parameters (for the stock-averaged view) ──────────────
# Every EFUS dwelling carries its own intercept/climate-slope under the fitted GEE,
# built from its actual EPC band, dwelling type, age and cooling.  These drive both
# the stock-averaged exceedance % (Figure 1, right panel) and the stock-level
# P(criterion met) model (Figure 2).
dw_prof = (
    daily_df.groupby("CaseID")
    .agg(
        epc     = ("EPceeb12e_efus", "first"),
        dwtype  = ("dwtype_efus",    "first"),
        dwage   = ("dwage_efus",     "first"),
        cooling = ("AnyCooling",     "first"),
        occ_hrs = ("occ_hours",      "median"),
    )
    .reset_index()
)
dw_prof = dw_prof[dw_prof["epc"].isin(EPC_LEVELS)].reset_index(drop=True)
N_DW    = len(dw_prof)
print(f"\nStock-level model: {N_DW} EFUS dwellings")

def _dw_params(model):
    """Intercept and climate-slope arrays (N_DW,) for every dwelling."""
    b0_arr = np.zeros(N_DW)
    bc_arr = np.zeros(N_DW)
    for i in range(N_DW):
        r      = dw_prof.iloc[i]
        epc    = int(r["epc"]);   dwtype = int(r["dwtype"])
        dwage  = int(r["dwage"]); cool   = int(r["cooling"])
        b0     = gee_coef(model, "(Intercept)",           required=True)
        b_c    = gee_coef(model, "T_2DMMT_c",             required=True)
        b_epc  = (0.0 if epc    == GEE_REF_EPC    else
                  gee_coef(model, f"EPC{epc}",           required=True))
        b_int  = (0.0 if epc    == GEE_REF_EPC    else
                  gee_coef(model, f"T_2DMMT_c:EPC{epc}", required=True))
        b_dw   = (0.0 if dwtype == GEE_REF_DWTYPE else
                  gee_coef(model, f"dwtype{dwtype}",     required=False))
        b_da   = (0.0 if dwage  == GEE_REF_DWAGE  else
                  gee_coef(model, f"dwage{dwage}",       required=False))
        b_co   = (0.0 if cool   == GEE_REF_COOL   else
                  gee_coef(model, "cooling1",            required=False))
        b0_arr[i] = b0 + b_epc + b_dw + b_da + b_co
        bc_arr[i] = b_c + b_int
    return b0_arr, bc_arr

_dw_params_cache = {f"gee_excprop_{thr}": _dw_params(f"gee_excprop_{thr}")
                    for thr in THRESHOLDS}
_dw_occ = dw_prof["occ_hrs"].values.clip(1)
_dw_epc = dw_prof["epc"].values

def stock_exceed_pct(t2c_arr, b0_arr, bc_arr):
    """Per-dwelling seasonal mean exceedance % (vectorised over days), shape (N_DW,).

    E[daily_exceed_prop] is evaluated for every dwelling on every season day, then
    averaged over days.  Averaging the result over the dwellings in an EPC band gives
    the stock-averaged exceedance % for that band (equal-weighted across dwellings).
    """
    eta = b0_arr[:, None] + bc_arr[:, None] * t2c_arr[None, :]
    p   = 1.0 / (1.0 + np.exp(-eta))
    return p.mean(axis=1) * 100.0

# ── Extract GCM season-months daily T_2DMMT ───────────────────────────────────
CFTIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

def extract_season_daily(nc_path, region=UKCP18_REGION):
    """
    Returns {year: array_of_daily_T2DMMT} for the configured SEASON_MONTHS.

    T_2DMMT_d = (Tmax_d + Tmax_{d-1}) / 2, computed only within the season days.
    The first day of each season uses its own Tmax as both values — matching the
    dataset builder's rolling(2, min_periods=1) behaviour on the filtered months.
    """
    ds     = xr.open_dataset(nc_path, decode_times=CFTIME_CODER)
    tasmax = ds.sel(region=region).isel(ensemble_member=0)["tasmax"].values
    times  = ds.time.values
    ds.close()

    months = np.array([t.month for t in times])
    years  = np.array([t.year  for t in times])

    result = {}
    for yr in np.unique(years):
        season_idx = np.where(np.isin(months, SEASON_MONTHS) & (years == yr))[0]

        if len(season_idx) < MIN_SEASON_DAYS:
            continue

        # Replicate rolling(2, min_periods=1): first day of season uses itself alone
        all_vals = np.concatenate([[tasmax[season_idx[0]]], tasmax[season_idx]])
        t2dmmt   = 0.5 * (all_vals[:-1] + all_vals[1:])
        result[yr] = t2dmmt
    return result

# ── Training range for extrapolation diagnostic ───────────────────────────────
train_t2c     = daily_df["T_2DMMT"].dropna().values - T2DMMT_OBS_MEAN
train_t2c_min = float(train_t2c.min())
train_t2c_max = float(train_t2c.max())

# ── True observed validation: actual exceedance % from daily_overheating ──────
obs_actual_rows = []
for thr in THRESHOLDS:
    pcol = f"daily_exceed_prop_{thr}"
    for epc in EPC_LEVELS:
        sub = daily_df[daily_df["EPceeb12e_efus"] == epc].copy()
        if sub.empty:
            continue
        obs_pct = 100.0 * np.average(sub[pcol].fillna(0), weights=sub["occ_hours"])
        obs_actual_rows.append({"threshold": thr, "epc": epc, "obs_exceed_pct": obs_pct})
obs_actual_df = pd.DataFrame(obs_actual_rows)

# ── Process all ensemble members ──────────────────────────────────────────────
nc_files = sorted(glob.glob(NC_GLOB))
print(f"\nFound {len(nc_files)} ensemble member files\n")

# Pass 1: extract daily T_2DMMT and per-member 1981-2010 baselines
print("Pass 1: extracting daily T_2DMMT and computing per-member baselines...")
members_data = {}   # member -> (yr_daily, gcm_baseline, cal)

for nc_path in nc_files:
    ds_tmp = xr.open_dataset(nc_path, decode_times=CFTIME_CODER)
    member = int(ds_tmp.ensemble_member.values[0])
    cal    = ds_tmp.time.encoding.get("calendar", "?")
    ds_tmp.close()

    yr_daily = extract_season_daily(nc_path)

    baseline_vals = np.concatenate([
        v for yr, v in yr_daily.items() if BASELINE_START <= yr <= BASELINE_END
    ])
    gcm_baseline  = float(baseline_vals.mean())
    n_base_yrs    = sum(1 for yr in yr_daily if BASELINE_START <= yr <= BASELINE_END)

    print(f"  member {member:02d} [{cal:20s}]: "
          f"{SEASON_LABEL} baseline={gcm_baseline:.2f} degC  ({n_base_yrs} yrs)")

    members_data[member] = (yr_daily, gcm_baseline, cal)

# Cross-scale correction: align GCM t2c to the GEE model's centring.
# The GCM per-member baselines reflect the 1981-2010 climate; the GEE model
# was centred on the EFUS observed season mean.  The correction bridges this gap.
gcm_global_baseline = float(np.mean([b for _, b, _ in members_data.values()]))
correction          = T2DMMT_OBS_MEAN - gcm_global_baseline

print(f"\nGCM multi-model mean baseline (1981-2010, {SEASON_LABEL}): "
      f"{gcm_global_baseline:.2f} degC")
print(f"Observed {SEASON_LABEL} mean (T2DMMT_OBS_MEAN):   {T2DMMT_OBS_MEAN:.2f} degC")
print(f"Cross-correction applied to each member:          {correction:+.2f} degC\n")

# Pass 2: apply per-member + cross correction; predict seasonal exceedance %
# for both the modal profile (records → proj) and the EPC-band stock average
# (stock_records → proj_stock).
print("Pass 2: predicting seasonal exceedance % with cross-corrected t2c...")
records         = []
stock_records   = []
proj_future_t2c = []   # all future daily t2c values for extrapolation diagnostic

for member, (yr_daily, gcm_baseline, cal) in members_data.items():
    for yr, daily_t2 in yr_daily.items():
        if not (HIST_START <= yr <= PROJ_END):
            continue
        # Per-member debiasing + cross-scale correction
        t2c = daily_t2 - gcm_baseline - correction

        if yr >= PROJ_START:
            proj_future_t2c.extend(t2c.tolist())

        for thr in THRESHOLDS:
            model = f"gee_excprop_{thr}"
            # Stock-averaged: per-dwelling seasonal exceedance %, averaged per EPC band
            b0_arr, bc_arr = _dw_params_cache[model]
            dw_pct         = stock_exceed_pct(t2c, b0_arr, bc_arr)
            for epc in EPC_LEVELS:
                # Modal profile
                p = predict_profile(t2c, epc, model)
                records.append(dict(
                    member          = member,
                    year            = yr,
                    threshold       = thr,
                    epc             = epc,
                    mean_t2c        = float(t2c.mean()),
                    mean_exceed_pct = float(p.mean() * 100),
                    period          = "hist" if yr <= HIST_END else "future",
                ))
                # Stock average over the EFUS dwellings in this EPC band
                mask = _dw_epc == epc
                if mask.any():
                    stock_records.append(dict(
                        member          = member,
                        year            = yr,
                        threshold       = thr,
                        epc             = epc,
                        mean_exceed_pct = float(dw_pct[mask].mean()),
                        period          = "hist" if yr <= HIST_END else "future",
                    ))

proj       = pd.DataFrame(records)
proj_stock = pd.DataFrame(stock_records)

# ── Extrapolation diagnostic ──────────────────────────────────────────────────
proj_future_t2c = np.array(proj_future_t2c)
n_extrap = int(np.sum(
    (proj_future_t2c < train_t2c_min) | (proj_future_t2c > train_t2c_max)
))
print(f"\nExtrapolation diagnostic (centred T_2DMMT, t2c):")
print(f"  EFUS {SEASON_LABEL} training daily t2c range: "
      f"[{train_t2c_min:.2f}, {train_t2c_max:.2f}] degC")
print(f"  Future GCM (2020-2099) daily t2c range:      "
      f"[{proj_future_t2c.min():.2f}, {proj_future_t2c.max():.2f}] degC")
print(f"  Future daily values outside training range:  "
      f"{n_extrap:,} / {len(proj_future_t2c):,} "
      f"({100 * n_extrap / len(proj_future_t2c):.1f}%)")

# ── Model-predicted 2018 validation point ─────────────────────────────────────
# Passes observed EFUS T_2DMMT through predict_profile for the modal profile.
# This validates scale alignment, not model fit to actual overheating rates.
# Compare with obs_actual_df for model fit assessment.
t2dmmt_obs = daily_df["T_2DMMT"].dropna().values
obs_modelpred_rows = []
for thr in THRESHOLDS:
    model = f"gee_excprop_{thr}"
    for epc in EPC_LEVELS:
        t2c = t2dmmt_obs - T2DMMT_OBS_MEAN
        p   = predict_profile(t2c, epc, model)
        obs_modelpred_rows.append(dict(threshold=thr, epc=epc,
                                       mean_exceed_pct=float(p.mean() * 100),
                                       mean_t2c=float(t2c.mean())))
obs_modelpred_df = pd.DataFrame(obs_modelpred_rows)

out_csv = os.path.join(ANALYSIS, "climate_projection.csv")
proj.to_csv(out_csv, index=False)
proj_stock.to_csv(os.path.join(ANALYSIS, "climate_projection_stock.csv"), index=False)
print(f"\nSaved: {out_csv}  ({len(proj):,} rows, "
      f"{proj.member.nunique()} members)")

# ── Summary tables ────────────────────────────────────────────────────────────
for thr in THRESHOLDS:
    print(f"\nEnsemble-mean seasonal exceedance [%] -- {thr} degC  "
          f"({ROOM} overheating-criterion threshold = {CRIT_LINE:.0f}%):")
    sub = proj[proj.threshold == thr].copy()
    sub["decade"] = (sub.year // 10) * 10
    tbl = (sub.groupby(["decade", "epc"])["mean_exceed_pct"]
              .mean().unstack("epc"))
    tbl.columns = [EPC_LABELS[c] for c in tbl.columns]
    print(tbl.round(1).to_string())

print(f"\nObserved vs model-predicted {SEASON_LABEL} 2018 exceedance [%]:")
for thr in THRESHOLDS:
    print(f"\n  Threshold {thr} degC:")
    for epc in EPC_LEVELS:
        obs_row  = obs_actual_df[(obs_actual_df.threshold == thr) &
                                  (obs_actual_df.epc == epc)]
        pred_row = obs_modelpred_df[(obs_modelpred_df.threshold == thr) &
                                     (obs_modelpred_df.epc == epc)]
        obs_val  = float(obs_row.obs_exceed_pct) if not obs_row.empty else float("nan")
        pred_val = float(pred_row.mean_exceed_pct) if not pred_row.empty else float("nan")
        print(f"    EPC {EPC_LABELS[epc]:3s}  observed={obs_val:.2f}%  "
              f"model-predicted={pred_val:.2f}%  "
              f"diff={pred_val - obs_val:+.2f}%")

# ── Compute ensemble statistics per year ─────────────────────────────────────
def _ensemble_stats(df):
    return (
        df.groupby(["threshold", "epc", "year"])["mean_exceed_pct"]
        .agg(
            ens_mean  = "mean",
            ens_p10   = lambda x: np.percentile(x, 10),
            ens_p90   = lambda x: np.percentile(x, 90),
            n_members = "count",
        )
        .reset_index()
    )

ens       = _ensemble_stats(proj)        # modal profile
ens_stock = _ensemble_stats(proj_stock)  # stock-averaged over EFUS dwellings

# ── LaTeX projection table: ensemble-mean exceedance % by EPC band and year ───
TABLE_YEARS = [2018, 2035, 2050, 2080]
TABLE_THR   = THRESHOLDS[0]

def _ens_pct(ens_data, epc, year):
    sub = ens_data[(ens_data.threshold == TABLE_THR) &
                   (ens_data.epc == epc) & (ens_data.year == year)]
    return float(sub.ens_mean.iloc[0]) if not sub.empty else float("nan")

def emit_projection_table():
    ny = len(TABLE_YEARS)
    yhdr = " & ".join(str(y) for y in TABLE_YEARS)
    print("\n% --- projection table (auto-generated) ---")
    print(r"\begin{table}[ht]")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{l " + "r" * ny + " " + "r" * ny + "}")
    print(r"    \toprule")
    print(rf"    & \multicolumn{{{ny}}}{{c}}{{Modal dwelling}} &")
    print(rf"      \multicolumn{{{ny}}}{{c}}{{Stock average}} \\")
    print(rf"    \cmidrule(lr){{2-{ny + 1}}}\cmidrule(lr){{{ny + 2}-{2 * ny + 1}}}")
    print(rf"    EPC band & {yhdr} & {yhdr} \\")
    print(r"    \midrule")
    for epc in sorted(EPC_LABELS):
        modal = " & ".join(f"{_ens_pct(ens, epc, y):.1f}" for y in TABLE_YEARS)
        stock = " & ".join(f"{_ens_pct(ens_stock, epc, y):.1f}" for y in TABLE_YEARS)
        print(rf"    {EPC_LABELS[epc]:3s} & {modal} & {stock} \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(rf"  \caption{{Projected exceedance of occupied hours above "
          rf"{TABLE_THR}\,$^\circ$C (\%) for the modal dwelling and the "
          rf"stock average.}}")
    print(r"  \label{tab:projection}")
    print(r"\end{table}")

emit_projection_table()

# ── Plot ──────────────────────────────────────────────────────────────────────
# Single threshold (26 degC), two panels of the SAME exceedance-proportion metric:
#   left  = modal EFUS dwelling profile
#   right = stock average over all EFUS dwellings in each EPC band
THR = THRESHOLDS[0]

def draw_exceed_panel(ax, ens_data, title):
    """One exceedance-% panel: percentile band and ensemble mean."""
    for ci, epc in enumerate(EPC_LEVELS):
        col = COLORS[3 - ci]
        sub = ens_data[(ens_data.threshold == THR) &
                       (ens_data.epc == epc)].sort_values("year")
        yrs = sub.year.values

        # Shaded envelope (10th-90th percentile)
        ax.fill_between(yrs, smooth(sub.ens_p10.values), smooth(sub.ens_p90.values),
                        color=col, alpha=0.20, linewidth=0, zorder=2)

        # Ensemble mean
        ax.plot(yrs, smooth(sub.ens_mean.values), color=col, linewidth=1.8,
                zorder=4, label=f"EPC {EPC_LABELS[epc]}")

    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel(
        # rf"Occupied hours above {THR}$^\circ$C, {SEASON_LABEL} (\%)", fontsize=8
        rf"Occupied hours above {THR}$^\circ$C (\%)", fontsize=8

    )
    ax.set_title(title, fontsize=8)
    ax.set_xlim(HIST_START + SMOOTH_WINDOW // 2,
               PROJ_END  - SMOOTH_WINDOW // 2)

fig = plt.figure(figsize=(6.8, 2.6), constrained_layout=True)
gs  = fig.add_gridspec(1, 2)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

_dwtype_str = DWTYPE_LABELS.get(MODAL_DWTYPE, f"type {MODAL_DWTYPE}")
_dwage_str  = DWAGE_LABELS.get(MODAL_DWAGE,  f"age {MODAL_DWAGE}")
draw_exceed_panel(axes[0], ens,
                  rf"Modal dwelling ({_dwtype_str}, {_dwage_str})")
draw_exceed_panel(axes[1], ens_stock,
                  rf"Stock average ({N_DW} EFUS dwellings)")

# Panel labels (A)/(B) above the axes, matching the forest plots
for k, ax in enumerate(axes):
    ax.text(-0.02, 1.04, f"({chr(65 + k)})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right")

# Share a y-axis floor at/below 0 so the criterion line is always visible
ymax = 0.0
for ax in axes:
    ax.relim(); ax.autoscale_view()
    ymax = max(ymax, ax.get_ylim()[1])
for ax in axes:
    ax.set_ylim(0, ymax)

# Legend — one entry per EPC band: ensemble-mean line + central-80% band swatch
epc_handles = [
    (mlines.Line2D([], [], color=COLORS[3 - i], linewidth=1.8),
     plt.Rectangle((0, 0), 1, 1, fc=COLORS[3 - i], alpha=0.30))
    for i, _ in enumerate(EPC_LEVELS)
]
epc_labels = [f"EPC {EPC_LABELS[e]} mean" for e in EPC_LEVELS]
fig.legend(epc_handles, epc_labels, fontsize=6, frameon=True,
           loc="outside lower center", ncol=len(EPC_LEVELS),
           handler_map={tuple: HandlerTuple(ndivide=2)},
           title=r"Mean (line) and 10th--90th percentile (shaded)",
           title_fontsize=6)

# fig.suptitle(
#     rf"Expected \% of {SEASON_LABEL} occupied hours above {THR}$^\circ$C, by EPC band"
#     "\n"
#     rf"EFUS London {ROOM} $\cdot$ modal profile vs.\ stock average"
#     rf" $\cdot$ UKCP18 RCP8.5, {proj.member.nunique()} members"
#     rf" $\cdot$ Overheating criterion: $>{int(CRIT_LINE)}$\%",
#     fontsize=8,
# )

out_path = os.path.join(OUTDIR, "climate_projection.svg")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

# ── Stock-level P(seasonal overheating) ─────────────────────────────────────────
#
# For each dwelling j in the EFUS sample and each (member, year):
#
#   H_jd ~ Binomial(n_j, p_jd)          [occupied hours above threshold, day d]
#   seasonal sum H_j = Σ_d H_jd
#                    ≈ Normal(μ_j, σ²_j)   [Poisson-Binomial → normal approx]
#   μ_j  = n_j × Σ_d p_jd
#   σ²_j = n_j × Σ_d p_jd(1 − p_jd)
#   P(fail_j) = P(H_j / (n_j × n_days) > CRIT_LINE/100)
#             = 1 − Φ((c × n_j × n_days − μ_j) / σ_j)
#
# P(fail | EPC=e) is then averaged over all EFUS dwellings with EPC band e.

# dw_prof, _dw_params_cache, _dw_occ and _dw_epc are built once near the top of
# the script (they also drive the stock-averaged exceedance panel).
def _pfail(t2c_arr, b0_arr, bc_arr):
    """P(seasonal overheating) per dwelling via normal approx (vectorised)."""
    n_days = len(t2c_arr)
    eta    = b0_arr[:, None] + bc_arr[:, None] * t2c_arr[None, :]
    p      = 1.0 / (1.0 + np.exp(-eta))
    n_j    = np.round(_dw_occ).astype(int).clip(1)
    mu     = n_j * p.sum(axis=1)
    var    = n_j * (p * (1.0 - p)).sum(axis=1)
    sig    = np.sqrt(np.maximum(var, 1e-9))
    thresh = (CRIT_LINE / 100.0) * n_j * n_days
    z      = (thresh - mu) / sig
    return _scipy_stats.norm.sf(z)


# Pass 3 — reuse already-loaded members_data; no extra NetCDF reads
print("Pass 3: stock-level P(overheating criterion) per member × year...")
fail_recs = []
for member, (yr_daily, gcm_baseline, _) in members_data.items():
    for yr, daily_t2 in yr_daily.items():
        if not (HIST_START <= yr <= PROJ_END):
            continue
        t2c = daily_t2 - gcm_baseline - correction
        for thr in THRESHOLDS:
            b0_arr, bc_arr = _dw_params_cache[f"gee_excprop_{thr}"]
            pfail          = _pfail(t2c, b0_arr, bc_arr)
            for epc in EPC_LEVELS:
                mask = _dw_epc == epc
                if not mask.any():
                    continue
                fail_recs.append(dict(
                    member    = member,
                    year      = yr,
                    threshold = thr,
                    epc       = epc,
                    p_fail    = float(pfail[mask].mean()),
                ))

fail_proj = pd.DataFrame(fail_recs)

fail_ens = (
    fail_proj.groupby(["threshold", "epc", "year"])["p_fail"]
    .agg(
        ens_mean = "mean",
        ens_p10  = lambda x: np.percentile(x, 10),
        ens_p90  = lambda x: np.percentile(x, 90),
    )
    .reset_index()
)

# Observed 2018 EFUS failure rate: actual seasonal exceedance % per dwelling
obs_fail_rows = []
for thr in THRESHOLDS:
    ecol = f"daily_exceed_prop_{thr}"
    for epc in EPC_LEVELS:
        sub = daily_df[daily_df["EPceeb12e_efus"] == epc].copy()
        if sub.empty:
            continue
        sub["_eh"] = sub[ecol].fillna(0) * sub["occ_hours"]
        dw_exc = sub.groupby("CaseID")["_eh"].sum().rename("total_exceed")
        dw_occ = sub.groupby("CaseID")["occ_hours"].sum().rename("total_occ")
        dw_s   = pd.concat([dw_exc, dw_occ], axis=1).reset_index()
        dw_s["exc_pct"] = 100.0 * dw_s["total_exceed"] / dw_s["total_occ"].clip(1)
        dw_s["fail"]    = dw_s["exc_pct"] > CRIT_LINE
        obs_fail_rows.append(dict(threshold=thr, epc=epc,
                                  obs_pfail=float(dw_s["fail"].mean())))
obs_fail_df2 = pd.DataFrame(obs_fail_rows)

print(f"\nObserved EFUS 2018 stock failure rate:")
for thr in THRESHOLDS:
    print(f"  Threshold {thr} degC:")
    for epc in EPC_LEVELS:
        row = obs_fail_df2[(obs_fail_df2.threshold == thr) &
                            (obs_fail_df2.epc == epc)]
        if not row.empty:
            print(f"    EPC {EPC_LABELS[epc]:3s}  {float(row.obs_pfail):.3f}")

# ── Figure 2 ──────────────────────────────────────────────────────────────────
_ncol = len(THRESHOLDS)
fig2, axes2 = plt.subplots(1, _ncol, figsize=(5.5 * _ncol, 5.5),
                           constrained_layout=True)
axes2 = np.atleast_1d(axes2)

for ax, thr in zip(axes2, THRESHOLDS):
    for ci, epc in enumerate(EPC_LEVELS):
        col  = COLORS[3 - ci]
        sub  = (fail_ens[(fail_ens.threshold == thr) & (fail_ens.epc == epc)]
                .sort_values("year"))
        yrs  = sub.year.values
        s_m  = smooth(sub.ens_mean.values)
        s_lo = smooth(sub.ens_p10.values)
        s_hi = smooth(sub.ens_p90.values)
        ax.fill_between(yrs, s_lo, s_hi, color=col, alpha=0.20,
                        linewidth=0, zorder=1)
        ax.plot(yrs, s_m, color=col, linewidth=1.8, zorder=3,
                label=f"EPC {EPC_LABELS[epc]}")

    for ci, epc in enumerate(EPC_LEVELS):
        col = COLORS[3 - ci]
        row = obs_fail_df2[(obs_fail_df2.threshold == thr) &
                            (obs_fail_df2.epc == epc)]
        if row.empty:
            continue
        ax.scatter(2018, float(row.obs_pfail), marker="D", s=30, zorder=5,
                   color=col, edgecolors=col)

    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel(rf"P(seasonal overheating criterion met), {SEASON_LABEL}", fontsize=8)
    ax.set_title(f"Threshold {thr}$^\circ$C", fontsize=8)
    ax.set_xlim(HIST_START + SMOOTH_WINDOW // 2, PROJ_END - SMOOTH_WINDOW // 2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))

epc_handles2 = [
    mlines.Line2D([], [], color=COLORS[3 - i], linewidth=1.8,
                  label=f"EPC {EPC_LABELS[e]}")
    for i, e in enumerate(EPC_LEVELS)
]
style_handles2 = [
    mlines.Line2D([], [], color="gray", linewidth=1.8,
                  label=f"Ensemble mean ({fail_proj.member.nunique()} members)"),
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.20,
                  label="10th--90th percentile"),
    mlines.Line2D([], [], color="gray", marker="D", markersize=5,
                  linestyle="none", markerfacecolor="gray",
                  markeredgewidth=1.1, label="Observed 2018 (EFUS)"),
]
axes2[0].legend(handles=epc_handles2 + style_handles2,
                fontsize=5.5, frameon=True, loc="lower right", ncol=2)

fig2.suptitle(
    rf"Stock-level probability of seasonal overheating criterion met, {SEASON_LABEL}, by EPC band"
    "\n"
    rf"Normal approximation to Poisson-Binomial seasonal exceedance"
    rf" $\cdot$ {N_DW} EFUS London dwellings, equal-weighted"
    "\n"
    rf"UKCP18 RCP8.5, {fail_proj.member.nunique()} ensemble members"
    f" $\\cdot$ Overheating criterion: $>{int(CRIT_LINE)}$\\%"
    r" $\cdot$ Diamond $=$ observed 2018 EFUS failure rate",
    fontsize=7.5,
)

out_path2 = os.path.join(OUTDIR, "climate_projection_stock_fail.svg")
fig2.savefig(out_path2, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out_path2}")
