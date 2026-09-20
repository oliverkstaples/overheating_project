"""
Model-based impact profile plots from LME (nlme) and GEE (geepack) outputs.

Reads:
  analysis/lme_coefs.csv
  analysis/lme_varcomp.csv
  analysis/gee_coefs.csv
  analysis/gee_pred_grid.csv
  analysis/lme_fitted.csv

Outputs → plots/efus2017/{region}_{room}_{1month|4month}/impact_profiles/
  lme_forest.svg        LME building-effect forest plot (continuous outcomes)
  gee_forest.svg        GEE building-effect forest plot (log-odds)
  gee_prob_curves.svg   P(overheating criterion) vs Outdoor T_2DMMT by EPC band
  lme_diagnostic.svg    LME fitted vs observed
"""

import argparse
import glob
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import scienceplots  # noqa: F401
from scipy import stats as _sp_stats
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ── Room configuration ────────────────────────────────────────────────────────
ROOM_CONFIG = {
    "livingroom": {
        "label":     "Living room",
        "occ_label": "07:00--21:00",
        "crit_pct":  3,
    },
    "bedroom": {
        "label":     "Bedroom",
        "occ_label": "22:00--06:59",
        "crit_pct":  1,
    },
}

REGION_CONFIG = {
    "london":    {"gor": 7, "ukcp18_region": 4,  "label": "London"},
    "southwest": {"gor": 8, "ukcp18_region": 9,  "label": "South West"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--subdir", type=str, default="",
                    help="Time-based subdirectory suffix, e.g. 'august'")
parser.add_argument("--room", type=str, default="livingroom",
                    choices=list(ROOM_CONFIG.keys()),
                    help="Room type (default: livingroom)")
parser.add_argument("--region", type=str, default="london",
                    choices=list(REGION_CONFIG.keys()),
                    help="EFUS region (default: london)")
args = parser.parse_args()

ROOM       = args.room
RCFG       = ROOM_CONFIG[ROOM]
ROOM_LABEL = RCFG["label"]
OCC_LABEL  = RCFG["occ_label"]
CRIT_PCT   = RCFG["crit_pct"]
REGION     = args.region
RCFG_REG   = REGION_CONFIG[REGION]

_reg_pre   = "" if REGION == "london" else REGION
_room_sub  = f"{ROOM}/{args.subdir}" if args.subdir else ROOM
_ana_sub   = "/".join(filter(None, [_reg_pre, _room_sub]))
ANALYSIS_DIR = os.path.join(ROOT, "analysis", _ana_sub) if _ana_sub else os.path.join(ROOT, "analysis")

_month_label = "1month" if args.subdir == "august" else "4month"
OUTDIR = os.path.join(ROOT, "plots", "efus2017",
                      f"{REGION}_{ROOM}_{_month_label}", "impact_profiles")
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use(["science", "nature", "bright"])

# ── Load outputs ──────────────────────────────────────────────────────────────
lme_coefs  = pd.read_csv(os.path.join(ANALYSIS_DIR, "lme_coefs.csv"))
lme_vc     = pd.read_csv(os.path.join(ANALYSIS_DIR, "lme_varcomp.csv"))
gee_coefs  = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_coefs.csv"))
pred_grid     = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_pred_grid.csv"))
pred_grid_nin = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_pred_grid_nin.csv"))
lme_fitted = pd.read_csv(os.path.join(ANALYSIS_DIR, "lme_fitted.csv"))

# ── Load daily data early for metadata (n, season label) ─────────────────────
daily_df = pd.read_parquet(os.path.join(ANALYSIS_DIR, "daily_overheating.parquet"))
daily_df["date_key"] = pd.to_datetime(daily_df["date"]).dt.normalize()
_months = sorted(pd.to_datetime(daily_df["date"]).dt.month.unique())
_MNAMES = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
SEASON_LABEL = (_MNAMES.get(_months[0], str(_months[0])) if len(_months) == 1
                else f"{_MNAMES.get(_months[0], str(_months[0]))}--{_MNAMES.get(_months[-1], str(_months[-1]))}")
N_OBS = len(daily_df)

# Normalise interaction term names across models
# e.g. "T_2DMMT_c:EPC2", "minTout_c:EPC2", "meanTout_c:EPC2" → "clim:EPC2"
for df in [lme_coefs, gee_coefs]:
    df["term"] = df["term"].str.replace(r"^[^:]+:EPC", "clim:EPC", regex=True)

# ── Style helpers ─────────────────────────────────────────────────────────────
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def dark(c):
    return tuple(np.array(mcolors.to_rgb(c)) * 0.45)

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Term catalogue ────────────────────────────────────────────────────────────
# (term key in coef table, y-axis label, group header — None = continuation)
TERM_META = [
    ("EPC2",     "D vs C+",           r"EPC band" + "\n" + r"(ref = C+)"),
    ("EPC3",     "E vs C+",           None),
    ("EPC4",     "F/G vs C+",         None),
    ("dwtype1",  "Detached",          r"Dwelling type" + "\n" + r"(ref = Flat)"),
    ("dwtype2",  "Semi-det.",          None),
    ("dwtype3",  "End-terrace",        None),
    ("dwtype4",  "Mid-terrace",        None),
    ("dwtype5",  "Bungalow",           None),
    ("dwage1",   "Pre-1919",           r"Build era" + "\n" + r"(ref = Post-1990)"),
    ("dwage2",   "1919--44",      None),
    ("dwage3",   "1945--64",      None),
    ("dwage4",   "1965--74",      None),
    ("dwage5",   "1975--80",      None),
    ("dwage6",   "1981--90",      None),
    ("cooling1", "Cooling",            r"Cooling" + "\n" + r"(ref = No)"),
    ("clim:EPC2","Climate $\\times$ D",   r"Climate $\times$ EPC" + "\n" + r"(slope diff.)"),
    ("clim:EPC3","Climate $\\times$ E",   None),
    ("clim:EPC4","Climate $\\times$ F/G", None),
]
TERMS  = [t for t, _, _ in TERM_META]
LABELS = {t: l for t, l, _ in TERM_META}
GROUPS = {t: g for t, g, _ in TERM_META}
N      = len(TERMS)
YPOS   = list(range(N - 1, -1, -1))  # top → bottom

sig_handle   = mlines.Line2D([], [], color="gray", marker="D", markersize=5,
                              linestyle="-", label=r"$p < 0.05$")
insig_handle = mlines.Line2D([], [], color="gray", marker="o", markersize=4,
                              linestyle="-", label=r"$p \geq 0.05$")

# ── Forest panel helper ───────────────────────────────────────────────────────
def draw_forest_panel(ax, model_name, coefs_df, color, p_col,
                      ci_lo_col="ci_lo", ci_hi_col="ci_hi"):
    sub = coefs_df[coefs_df["model"] == model_name].set_index("term")

    ax.axvline(0, color="black", linewidth=0.8, zorder=1)

    prev_grp = None
    for yp, term in zip(YPOS, TERMS):
        grp = GROUPS[term]
        if grp and grp != prev_grp:
            ax.axhline(yp + 0.6, color="gray", linewidth=0.5,
                       linestyle="--", alpha=0.5)
        if grp:
            prev_grp = grp

        if term not in sub.index:
            continue
        row = sub.loc[term]
        est  = float(row["estimate"])
        lo   = float(row[ci_lo_col])
        hi   = float(row[ci_hi_col])
        pval = float(row[p_col])

        ax.plot([lo, hi], [yp, yp], color=color, linewidth=1.4,
                solid_capstyle="round", zorder=2)
        mk = "D" if pval < 0.05 else "o"
        ms = 5.0 if pval < 0.05 else 4.0
        ax.plot(est, yp, marker=mk, color=color, markersize=ms,
                markeredgecolor=dark(color), markeredgewidth=0.5,
                zorder=3, linestyle="none")
        ax.text(hi + abs(hi - lo) * 0.08 + 1e-6, yp, f"{est:+.2f}",
                fontsize=5, va="center", color=dark(color))

    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.tick_params(axis="y", which="both", left=False)

def set_forest_yticks(ax):
    ax.set_yticks(YPOS)
    ax.set_yticklabels([LABELS[t] for t in TERMS], fontsize=6.5)

def add_group_labels(ax):
    xlim = ax.get_xlim()
    x_left = xlim[0] - (xlim[1] - xlim[0]) * 0.85
    prev_grp = None
    for yp, term in zip(YPOS, TERMS):
        grp = GROUPS[term]
        if grp and grp != prev_grp:
            ax.text(x_left, yp + 0.75, grp, fontsize=5.5, va="bottom",
                    ha="left", color="gray", style="italic",
                    clip_on=False)
        if grp:
            prev_grp = grp

# =============================================================================
# Figure 1 — LME forest plot (continuous outcomes)
# =============================================================================
print("Figure 1: LME forest plot...")

LME_PANELS = [
    ("lme_max_tin",  r"$\Delta$ max $T_{in}$ ($^\circ$C)"),
    ("lme_min_tin",  r"$\Delta$ min $T_{in}$ ($^\circ$C)"),
    ("lme_logdh26",  r"$\Delta$ log(DH$_{26}$+1)"),
    ("lme_logdh27",  r"$\Delta$ log(DH$_{27}$+1)"),
]

fig, axes = plt.subplots(1, 4, figsize=(11.5, 7.5),
                         sharey=True, constrained_layout=True)

for ax_i, (ax, (mname, xlabel)) in enumerate(zip(axes, LME_PANELS)):
    col = COLORS[ax_i % len(COLORS)]
    draw_forest_panel(ax, mname, lme_coefs, col, p_col="p_cr2",
                      ci_lo_col="ci_lo_cr2", ci_hi_col="ci_hi_cr2")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(xlabel, fontsize=7)

set_forest_yticks(axes[0])
add_group_labels(axes[0])
fig.legend(handles=[sig_handle, insig_handle], loc="lower center",
           ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.03), frameon=True)
fig.suptitle(
    f"LME fixed-effect estimates: building characteristics on daily overheating --- {ROOM_LABEL}s"
    "\nRandom intercept + slope on climate covariate"
    rf" $\cdot$ corAR1 $\cdot$ 95\% CI $\cdot$ {RCFG_REG['label']}, EFUS 2017",
    fontsize=8,
)
save(fig, "lme_forest.svg")

# =============================================================================
# Figure 2 — GEE forest plot (binary / proportion outcomes, log-odds)
# =============================================================================
print("Figure 2: GEE forest plot...")

GEE_PANELS = [
    ("gee_overheat_26",    r"Log-odds ratio: daily overheating (26$^\circ$C)"),
    ("gee_overheat_27",    r"Log-odds ratio: daily overheating (27$^\circ$C)"),
    ("gee_excprop_26", r"Log-odds ratio: occ. hour exceedance (26$^\circ$C)"),
    ("gee_excprop_27", r"Log-odds ratio: occ. hour exceedance (27$^\circ$C)"),
]

fig, axes = plt.subplots(1, 4, figsize=(11.5, 7.5),
                         sharey=True, constrained_layout=True)

for ax_i, (ax, (mname, xlabel)) in enumerate(zip(axes, GEE_PANELS)):
    col = COLORS[ax_i % len(COLORS)]
    draw_forest_panel(ax, mname, gee_coefs, col, p_col="p_san",
                      ci_lo_col="ci_lo", ci_hi_col="ci_hi")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(xlabel, fontsize=7)

set_forest_yticks(axes[0])
fig.legend(handles=[sig_handle, insig_handle], loc="lower center",
           ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.03), frameon=True)
fig.suptitle(
    f"GEE estimates: building characteristics on overheating criterion and exceedance proportion --- {ROOM_LABEL}s"
    "\nAR(1) working correlation"
    rf" $\cdot$ sandwich SE $\cdot$ log-odds scale $\cdot$ {RCFG_REG['label']}, EFUS 2017",
    fontsize=8,
)
save(fig, "gee_forest.svg")

# =============================================================================
# Figure 3 — GEE probability curves: P(overheating criterion) vs Outdoor T_2DMMT by EPC band
# =============================================================================
print("Figure 3: GEE probability curves...")

EPC_LABELS = {1: "EPC C+", 2: "EPC D", 3: "EPC E", 4: "EPC F/G"}
GEE_MODELS = ["gee_overheat_26", "gee_overheat_27"]
THRESHOLDS  = ["26", "27"]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), constrained_layout=True)

for ax, mname, thr in zip(axes, GEE_MODELS, THRESHOLDS):
    sub = pred_grid[pred_grid["model"] == mname]
    for ci, (epc_lvl, grp) in enumerate(sub.groupby("EPC_level")):
        grp = grp.sort_values("T_2DMMT")
        col = COLORS[3 - ci]
        ax.plot(grp["T_2DMMT"], grp["prob"] * 100,
                color=col, linewidth=1.5,
                label=EPC_LABELS.get(int(epc_lvl), str(epc_lvl)))

    ax.set_xlabel(r"Outdoor $T_{2DMMT}$ ($^\circ$C)", fontsize=8)
    ax.set_ylabel(r"P(met overheating criterion) (\%)", fontsize=8)
    ax.set_title(f"Fixed threshold {thr}$^\circ$C", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

axes[1].legend(fontsize=6.5, frameon=True, loc="upper left")
fig.suptitle(
    f"Predicted P(met overheating criterion) vs Outdoor $T_{{2DMMT}}$ by EPC band --- {ROOM_LABEL}s"
    "\nGEE (AR1 working correlation)"
    rf" $\cdot$ ref: Flat, no cooling $\cdot$ {RCFG_REG['label']}, EFUS 2017",
    fontsize=8,
)
save(fig, "gee_prob_curves.svg")

# =============================================================================
# Figure 4 — LME fitted vs observed (daily level)
# =============================================================================
print("Figure 4: LME fitted vs observed...")

LME_DIAG = [
    ("lme_max_tin",  r"Daily max $T_{in}$ ($^\circ$C)"),
    ("lme_min_tin",  r"Daily min $T_{in}$ ($^\circ$C)"),
    ("lme_logdh26",  r"log(DH$_{26}$+1)"),
    ("lme_logdh27",  r"log(DH$_{27}$+1)"),
]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), constrained_layout=True)

for ax, (mname, mlabel), color in zip(axes, LME_DIAG, COLORS):
    sub = lme_fitted[lme_fitted["model"] == mname]
    fv  = sub["fitted"].values
    ov  = sub["observed"].values
    rmse = np.sqrt(np.mean((fv - ov) ** 2))
    r2   = np.corrcoef(fv, ov)[0, 1] ** 2

    lo = min(fv.min(), ov.min()) - 0.3
    hi = max(fv.max(), ov.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.7, zorder=1)
    ax.scatter(fv, ov, s=4, color=color, alpha=0.25,
               edgecolors=dark(color), linewidths=0.3, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Fitted", fontsize=7)
    ax.set_ylabel("Observed", fontsize=7)
    ax.set_title(mlabel, fontsize=7)
    ax.text(0.05, 0.95,
            f"$R^2$={r2:.2f}\nRMSE={rmse:.2f}",
            transform=ax.transAxes, fontsize=6, va="top")

fig.suptitle(
    f"LME fitted vs observed --- daily dwelling-day level (n={N_OBS:,})\n"
    f"{RCFG_REG['label']} {ROOM_LABEL.lower()}s, {SEASON_LABEL} 2017",
    fontsize=8,
)
save(fig, "lme_diagnostic.svg")

# =============================================================================
# Figure 5 — GEE: predicted P(overheating criterion) vs observed, by Outdoor T_2DMMT bin and EPC
# =============================================================================
print("Figure 5: GEE predicted vs observed comparison...")

gee_fitted = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_fitted.csv"))
gee_fitted["date_key"] = pd.to_datetime(gee_fitted["date"]).dt.normalize()

# Dwelling-level seasonal summary (actual overheating criterion, one row per dwelling)
summary_df = pd.read_parquet(os.path.join(ANALYSIS_DIR,
                                          "dwelling_overheating_summary.parquet"))
# Max T_2DMMT per dwelling over the season (better spread than mean; 2DMMT is
# already a running average so re-averaging loses peak heat information)
max_t2 = (daily_df.groupby("CaseID")["T_2DMMT"].max()
                  .rename("max_T_2DMMT").reset_index())
summary_df = summary_df.merge(max_t2, on="CaseID", how="left")

# Peak 2-day rolling mean of daily indoor min per dwelling: captures the worst
# sustained overnight warmth (a dwelling property, unlike outdoor T_2DMMT)
peak_2d_min = (
    daily_df.sort_values(["CaseID", "date"])
    .groupby("CaseID")["daily_min_Tin"]
    .apply(lambda x: x.rolling(2, min_periods=1).mean().max())
    .rename("peak_2d_min_Tin")
    .reset_index()
)
summary_df = summary_df.merge(peak_2d_min, on="CaseID", how="left")

# Daily indoor 2DMMnT: 2-day running mean of daily_min_Tin per dwelling
daily_df = daily_df.sort_values(["CaseID", "date_key"])
daily_df["T_2DMMnT"] = (
    daily_df.groupby("CaseID")["daily_min_Tin"]
    .transform(lambda x: x.rolling(2, min_periods=2).mean())
)

EPC_LABELS_NUM = {1: "C+", 2: "D", 3: "E", 4: "F/G"}
N_BINS = 6

fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.5), constrained_layout=True)

# Pre-load T_2DMMT centre for stock-averaged panel E
_centres_val = pd.read_csv(os.path.join(ANALYSIS_DIR, "climate_centres.csv"))
_T2DMMT_CENTRE = float(_centres_val.loc[_centres_val.variable == "T_2DMMT", "centre"].iloc[0])

# Per-dwelling profiles for stock averaging (one row per dwelling)
_dw_val = (daily_df[["CaseID", "dwtype_efus", "dwage_efus", "EPceeb12e_efus", "AnyCooling"]]
           .drop_duplicates("CaseID").reset_index(drop=True))
_epc_val   = _dw_val["EPceeb12e_efus"].values.astype(int)
_dwtype_val = _dw_val["dwtype_efus"].values.astype(int)
_dwage_val  = _dw_val["dwage_efus"].values.astype(int)
_cool_val   = _dw_val["AnyCooling"].values.astype(int)

for row_i, (mname, thr) in enumerate([("gee_overheat_26", "26"), ("gee_overheat_27", "27")]):

    # ── merge fitted with daily covariates ────────────────────────────────────
    sub = gee_fitted[gee_fitted["model"] == mname].copy()
    merged = sub.merge(
        daily_df[["CaseID", "date_key", "T_2DMMT", "EPceeb12e_efus"]],
        on=["CaseID", "date_key"], how="inner"
    )
    merged["EPC_label"] = merged["EPceeb12e_efus"].map(EPC_LABELS_NUM)

    # Equal-width T_2DMMT bins across the full range
    t_min = merged["T_2DMMT"].min()
    t_max = merged["T_2DMMT"].max()
    bins  = np.linspace(t_min, t_max, N_BINS + 1)
    merged["bin"] = pd.cut(merged["T_2DMMT"], bins=bins, include_lowest=True)
    merged["bin_mid"] = merged["bin"].apply(lambda b: (b.left + b.right) / 2)

    # ── Panel A: predicted curves + empirical binned proportions by EPC ───────
    ax_a = axes[row_i, 0]
    pg = pred_grid[pred_grid["model"] == mname]
    for ci, (epc_lvl, grp) in enumerate(pg.groupby("EPC_level")):
        grp = grp.sort_values("T_2DMMT")
        col = COLORS[3 - ci]
        ax_a.plot(grp["T_2DMMT"], grp["prob"] * 100,
                  color=col, linewidth=1.5, label=f"EPC {EPC_LABELS_NUM.get(int(epc_lvl), str(epc_lvl))}")

    for ci, (epc_code, epc_grp) in enumerate(merged.groupby("EPceeb12e_efus")):
        col = COLORS[3 - ci]
        binned = (
            epc_grp.groupby("bin_mid", observed=True)
            .agg(obs_prop=("observed", "mean"), n=("observed", "count"))
            .reset_index()
        )
        binned = binned[binned["n"] >= 5]
        ax_a.scatter(binned["bin_mid"], binned["obs_prop"] * 100,
                     color=col, s=12, zorder=4,
                     edgecolors=dark(col), linewidths=0.5)

    ax_a.set_xlabel(r"Outdoor $T_{2DMMT}$ ($^\circ$C)", fontsize=7)
    ax_a.set_ylabel(r"P(met overheating criterion) (\%)" + "\n" + r"(dots: all dwelling types; lines: post-1990 flat, no cooling)",
                    fontsize=6)
    ax_a.set_title(f"Predicted (lines: ref. dwelling) vs observed (dots: all types)\n"
                   f"by EPC band --- {thr}$^\circ$C", fontsize=7)
    if row_i == 0:
        ax_a.legend(fontsize=5.5, frameon=True, loc="upper left")

    # ── Panel B: EPC-stratified mean fitted vs mean observed ──────────────────
    ax_b = axes[row_i, 1]
    epc_summary = (
        merged.groupby("EPceeb12e_efus")
        .agg(obs_rate=("observed",  "mean"),
             pred_mean=("fitted",   "mean"),
             n=("observed", "count"))
        .reset_index()
    )
    epc_labels_str = [EPC_LABELS_NUM[int(e)] for e in epc_summary["EPceeb12e_efus"]]
    x = np.arange(len(epc_summary))
    w = 0.32
    epc_colors_b = [COLORS[3 - i] for i in range(len(epc_summary))]
    for i, (row, col, lab) in enumerate(zip(
            epc_summary.itertuples(), epc_colors_b,
            epc_labels_str)):
        ax_b.bar(i - w/2, row.obs_rate  * 100, w, color=col, alpha=0.85,
                 edgecolor=dark(col), linewidth=0.5)
        ax_b.bar(i + w/2, row.pred_mean * 100, w, color=col, alpha=0.4,
                 edgecolor=dark(col), linewidth=0.5, hatch="//")
        ax_b.text(i, max(row.obs_rate, row.pred_mean) * 100 + 0.5,
                  f"n={row.n}", fontsize=4.5, ha="center", color="gray")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(epc_labels_str, fontsize=6.5)
    ax_b.set_xlabel("EPC band", fontsize=7)
    ax_b.set_ylabel(r"Mean P(met overheating criterion) (\%)", fontsize=7)
    ax_b.set_title(f"Mean predicted vs observed by EPC\n{thr}$^\circ$C", fontsize=7)
    if row_i == 0:
        from matplotlib.patches import Patch
        ax_b.legend(
            handles=[Patch(facecolor="gray", alpha=0.85, label="Observed"),
                     Patch(facecolor="gray", alpha=0.4,  hatch="//", label="Predicted")],
            fontsize=5.5, frameon=True
        )

    # ── Panel C: calibration — predicted decile bins vs observed proportion ───
    ax_c = axes[row_i, 2]
    n_cal_bins = 8
    merged["pred_decile"] = pd.qcut(merged["fitted"], q=n_cal_bins, duplicates="drop")
    cal = (
        merged.groupby("pred_decile", observed=True)
        .agg(mean_pred=("fitted",   "mean"),
             mean_obs =("observed", "mean"),
             n        =("observed", "count"))
        .reset_index()
    )
    ax_c.plot([0, 1], [0, 1], "k--", lw=0.7, zorder=1)
    ax_c.scatter(cal["mean_pred"] * 100, cal["mean_obs"] * 100,
                 s=cal["n"] / 8, color=COLORS[row_i],
                 edgecolors=dark(COLORS[row_i]), linewidths=0.5, zorder=3)
    for _, cr in cal.iterrows():
        ax_c.annotate(f"{int(cr['n'])}",
                      (cr["mean_pred"] * 100, cr["mean_obs"] * 100),
                      fontsize=4, xytext=(2, 2), textcoords="offset points",
                      color="gray")
    ax_c.set_xlabel(r"Mean predicted (\%)", fontsize=7)
    ax_c.set_ylabel(r"Observed proportion (\%)", fontsize=7)
    ax_c.set_title(f"Calibration (8 quantile bins)\n{thr}$^\circ$C", fontsize=7)

    # ── Panel D: P(overheating criterion) vs indoor 2DMMnT — GEE model lines + empirical dots
    ax_d = axes[row_i, 3]

    nin_mname = f"gee_nin_overheat_{thr}"
    pg_nin = pred_grid_nin[pred_grid_nin["model"] == nin_mname]
    for ci, (epc_lvl, grp) in enumerate(pg_nin.groupby("EPC_level")):
        grp = grp.sort_values("T_2DMMnT")
        col = COLORS[3 - ci]
        ax_d.plot(grp["T_2DMMnT"], grp["prob"] * 100,
                  color=col, linewidth=1.5,
                  label=f"EPC {EPC_LABELS_NUM.get(int(epc_lvl), str(epc_lvl))}")

    nin_fitted = gee_fitted[gee_fitted["model"] == nin_mname].copy()
    nin_fitted["date_key"] = pd.to_datetime(nin_fitted["date"]).dt.normalize()
    merged_d = nin_fitted.merge(
        daily_df[["CaseID", "date_key", "T_2DMMnT", "EPceeb12e_efus"]],
        on=["CaseID", "date_key"], how="inner"
    ).dropna(subset=["T_2DMMnT"])

    t2n_min = merged_d["T_2DMMnT"].min()
    t2n_max = merged_d["T_2DMMnT"].max()
    bins_d   = np.linspace(t2n_min, t2n_max, N_BINS + 1)
    merged_d["bin"]     = pd.cut(merged_d["T_2DMMnT"], bins=bins_d, include_lowest=True)
    merged_d["bin_mid"] = merged_d["bin"].apply(lambda b: (b.left + b.right) / 2)

    for ci, (epc_code, epc_grp) in enumerate(merged_d.groupby("EPceeb12e_efus")):
        col = COLORS[3 - ci]
        binned_d = (
            epc_grp.groupby("bin_mid", observed=True)
            .agg(obs_prop=("observed", "mean"), n=("observed", "count"))
            .reset_index()
        )
        binned_d = binned_d[binned_d["n"] >= 5]
        ax_d.scatter(binned_d["bin_mid"], binned_d["obs_prop"] * 100,
                     color=col, s=12, zorder=4,
                     edgecolors=dark(col), linewidths=0.5)

    ax_d.set_xlabel(r"Indoor $T_{2DMMnT}$ ($^\circ$C)", fontsize=7)
    ax_d.set_ylabel(r"P(met overheating criterion) (\%)" + "\n"
                    + r"(dots: all dwelling types; lines: ref. dwelling)",
                    fontsize=6)
    ax_d.set_title(f"Predicted (lines: ref. dwelling) vs observed (dots: all types)\n"
                   f"by EPC band, indoor $T_{{2DMMnT}}$ --- {thr}$^\\circ$C", fontsize=7)
    if row_i == 0:
        ax_d.legend(fontsize=5.5, frameon=True, loc="upper left")

    # ── Panel E: stock-averaged P(overheating criterion) by EPC, marginalised over dwelling types
    ax_e = axes[row_i, 4]

    def _gce(term):
        row = gee_coefs[(gee_coefs["model"] == mname) & (gee_coefs["term"] == term)]
        return float(row["estimate"].iloc[0]) if len(row) else 0.0

    _n_val = len(_dw_val)
    _ints_e = np.empty(_n_val)
    _slps_e = np.empty(_n_val)
    _b0 = _gce("(Intercept)"); _bc = _gce("T_2DMMT_c")
    for _i in range(_n_val):
        _epc = _epc_val[_i]; _dwt = _dwtype_val[_i]
        _dwa = _dwage_val[_i]; _coo = _cool_val[_i]
        _b_epc = 0.0 if _epc == 1 else _gce(f"EPC{_epc}")
        _b_int = 0.0 if _epc == 1 else _gce(f"T_2DMMT_c:EPC{_epc}")
        _b_dwt = 0.0 if _dwt == 6 else _gce(f"dwtype{_dwt}")
        _b_dwa = 0.0 if _dwa == 7 else _gce(f"dwage{_dwa}")
        _b_coo = 0.0 if _coo == 0 else _gce("cooling1")
        _ints_e[_i] = _b0 + _b_epc + _b_dwt + _b_dwa + _b_coo
        _slps_e[_i] = _bc + _b_int

    _t2_grid = np.linspace(merged["T_2DMMT"].min(), merged["T_2DMMT"].max(), 120)
    _t2c_grid = _t2_grid - _T2DMMT_CENTRE

    for ci, epc_code in enumerate(sorted(_dw_val["EPceeb12e_efus"].unique())):
        col  = COLORS[3 - ci]
        mask = _epc_val == epc_code
        eta  = _ints_e[mask, np.newaxis] + _slps_e[mask, np.newaxis] * _t2c_grid[np.newaxis, :]
        prob_mean = (1.0 / (1.0 + np.exp(-eta))).mean(axis=0) * 100.0
        ax_e.plot(_t2_grid, prob_mean, color=col, linewidth=1.5,
                  label=f"EPC {EPC_LABELS_NUM[epc_code]}")

        epc_grp = merged[merged["EPceeb12e_efus"] == epc_code]
        binned_e = (
            epc_grp.groupby("bin_mid", observed=True)
            .agg(obs_prop=("observed", "mean"), n=("observed", "count"))
            .reset_index()
        )
        binned_e = binned_e[binned_e["n"] >= 5]
        ax_e.scatter(binned_e["bin_mid"], binned_e["obs_prop"] * 100,
                     color=col, s=12, zorder=4,
                     edgecolors=dark(col), linewidths=0.5)

    ax_e.axhline(CRIT_PCT, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_e.set_xlabel(r"Outdoor $T_{2DMMT}$ ($^\circ$C)", fontsize=7)
    ax_e.set_ylabel(r"P(met overheating criterion) (\%)", fontsize=7)
    ax_e.set_title(f"Stock-averaged: P(met overheating criterion) by EPC\n"
                   f"(marginalised over dwtype, dwage, cooling) --- {thr}$^\\circ$C",
                   fontsize=7)
    if row_i == 0:
        ax_e.legend(fontsize=5.5, frameon=True, loc="upper left")

fig.suptitle(
    f"GEE model validation: predicted P(met overheating criterion) vs observed --- {ROOM_LABEL}s"
    "\nA/D: ref. dwelling (Flat, post-1990, no cooling)"
    r" $\cdot$ E: stock-averaged over all dwelling types within EPC band"
    r" $\cdot$ Dots = empirical rates $\cdot$ EFUS 2017",
    fontsize=8,
)
save(fig, "gee_validation.svg")

# =============================================================================
print("Figure 6: Cumulative overheating exceedance events vs Outdoor T_2DMMT...")

THRESHOLDS = [26, 27, 28]

# --- Statistical helpers (EPC C+ vs others) -----------------------------------
def _pstars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

# Dwelling-level EPC lookup (one row per dwelling)
_dw_epc = daily_df.groupby("CaseID")["EPceeb12e_efus"].first()

# Per-dwelling seasonal means — proper unit of replication for MW tests
_dw_prop_exceed = {
    thr: daily_df.groupby("CaseID").apply(lambda g, t=thr: (g["daily_mean_Tin"] > t).mean())
    for thr in THRESHOLDS
}
_dw_dh_mean = {
    thr: daily_df.groupby("CaseID")[f"daily_dh_{thr}"].mean()
    for thr in THRESHOLDS
}

def _mw_annot(series):
    """Mann-Whitney U: C+ vs D/E/F/G on dwelling-level series. Returns annotation lines."""
    ref = series[_dw_epc == 1].values
    lines = []
    for epc in [2, 3, 4]:
        cmp = series[_dw_epc == epc].values
        if len(cmp) >= 3:
            _, p = _sp_stats.mannwhitneyu(ref, cmp, alternative="two-sided")
            lines.append(f"{EPC_LABELS_NUM[epc]}: {_pstars(p)}")
        else:
            lines.append(f"{EPC_LABELS_NUM[epc]}: n/a")
    return "\n".join(lines)

def _ks_annot(flag_col):
    """KS test: T_2DMMT distribution on overheating days, C+ vs D/E/F/G.
    D is the effect size; p-values are approximate (pseudo-replicated days)."""
    t2_ref = daily_df.loc[
        (daily_df["EPceeb12e_efus"] == 1) & (daily_df[flag_col] == 1), "T_2DMMT"
    ].dropna().values
    lines = []
    for epc in [2, 3, 4]:
        t2_cmp = daily_df.loc[
            (daily_df["EPceeb12e_efus"] == epc) & (daily_df[flag_col] == 1), "T_2DMMT"
        ].dropna().values
        if len(t2_cmp) >= 5 and len(t2_ref) >= 5:
            D, p = _sp_stats.ks_2samp(t2_ref, t2_cmp)
            lines.append(f"{EPC_LABELS_NUM[epc]}: D={D:.2f} {_pstars(p)}")
        else:
            lines.append(f"{EPC_LABELS_NUM[epc]}: n/a")
    return "\n".join(lines)

def _add_annot(ax, text, loc="tl"):
    """Add a small annotation box; loc='tl'|'tr'|'bl'|'br'."""
    x, y = (0.03, 0.97) if loc == "tl" else (0.97, 0.97) if loc == "tr" else \
           (0.03, 0.03) if loc == "bl" else (0.97, 0.03)
    ha = "left" if loc in ("tl", "bl") else "right"
    va = "top"  if loc in ("tl", "tr") else "bottom"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=5, va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, linewidth=0.4))

# ------------------------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(6.8, 7.5),
                         sharex=True, constrained_layout=True)

# ── Row 0: cumulative fraction of overheating exceedance events ──────────────────────
for ax, threshold in zip(axes[0], THRESHOLDS):
    thr = str(threshold)
    flag_col = f"daily_overheat_{thr}"

    for ci, (epc_code, grp) in enumerate(daily_df.groupby("EPceeb12e_efus")):
        col   = COLORS[3 - ci]
        label = EPC_LABELS_NUM.get(int(epc_code), str(epc_code))

        grp_s   = grp.sort_values("T_2DMMT")
        # Normalise by total dwelling-days so curves reach the band's overall
        # exceedance rate rather than 100% — preserves between-EPC comparison
        cum_pct = 100 * grp_s[flag_col].cumsum() / len(grp_s)

        ax.plot(grp_s["T_2DMMT"], cum_pct,
                linewidth=1.4, color=col, label=f"EPC {label}")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f"{threshold}$^\\circ$C", fontsize=8)
    ax.grid(axis="y")
    _add_annot(ax, f"KS vs C+:\n{_ks_annot(f'daily_overheat_{threshold}')}", loc="tl")

axes[0, 0].set_ylabel("Cumulative overheating rate\n(fraction of dwelling-days)", fontsize=7)

# ── Row 1: proportion of living rooms with mean T_in > threshold ───────────────
# Bin T_2DMMT to nearest integer, then compute fraction of dwelling-days in each
# bin where the daily mean indoor temperature exceeds the threshold.
daily_df["T_2DMMT_bin"] = daily_df["T_2DMMT"].round()

for ax, threshold in zip(axes[1], THRESHOLDS):
    thr = str(threshold)

    for ci, (epc_code, grp) in enumerate(daily_df.groupby("EPceeb12e_efus")):
        col   = COLORS[3 - ci]
        label = EPC_LABELS_NUM.get(int(epc_code), str(epc_code))

        prop = (
            grp.groupby("T_2DMMT_bin")
            .apply(lambda g: (g["daily_mean_Tin"] > threshold).mean() * 100)
            .reset_index(name="pct_exceed")
        )

        ax.plot(prop["T_2DMMT_bin"], prop["pct_exceed"],
                marker="o", markersize=2.5, linewidth=1.4,
                color=col, label=f"EPC {label}")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y")
    ax.set_ylim(0, 100)
    _add_annot(ax, f"MW vs C+:\n{_mw_annot(_dw_prop_exceed[threshold])}", loc="tl")

axes[1, 0].set_ylabel(f"{ROOM_LABEL}s with mean $T_{{in}}$ $>$ threshold", fontsize=7)

# ── Row 2: mean degree-hours above threshold per T_2DMMT bin ──────────────────
for ax, threshold in zip(axes[2], THRESHOLDS):
    thr = str(threshold)
    dh_col = f"daily_dh_{thr}"

    for ci, (epc_code, grp) in enumerate(daily_df.groupby("EPceeb12e_efus")):
        col   = COLORS[3 - ci]
        label = EPC_LABELS_NUM.get(int(epc_code), str(epc_code))

        dh = (
            grp.groupby("T_2DMMT_bin")[dh_col]
            .mean()
            .reset_index(name="mean_dh")
        )

        ax.plot(dh["T_2DMMT_bin"], dh["mean_dh"],
                marker="o", markersize=2.5, linewidth=1.4,
                color=col, label=f"EPC {label}")

    ax.set_xlabel(r"Outdoor $T_{2DMMT}$ ($^\circ$C)", fontsize=7)
    ax.grid(axis="y")
    _add_annot(ax, f"MW vs C+:\n{_mw_annot(_dw_dh_mean[threshold])}", loc="tl")

axes[2, 0].set_ylabel(r"Mean degree-hours above threshold" + "\n" + r"(°C$\cdot$h per dwelling-day)", fontsize=7)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=6, title="EPC band", frameon=True,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    f"Fixed threshold exceedance vs Outdoor $T_{{2DMMT}}$ by EPC band --- {ROOM_LABEL}s"
    f"\n{RCFG_REG['label']}, EFUS 2017, {SEASON_LABEL} (occupied hours {OCC_LABEL})",
    fontsize=8,
)
save(fig, "fixed_cumulative_exceedance_vs_2dmmt.svg")

# =============================================================================
# Figure 7 — Climate projection: E[seasonal exceedance %] vs year by EPC band
# =============================================================================
print("Figure 7: Climate projection...")

_PROJ_THRESHOLDS = [26, 27]
_EPC_LEVELS      = [1, 2, 3, 4]
_EPC_LABELS_PROJ = {1: "C+", 2: "D", 3: "E", 4: "F/G"}
_LONDON_REGION   = RCFG_REG["ukcp18_region"]
_BASELINE_START  = 1981
_BASELINE_END    = 2010
_HIST_START      = 1981
_HIST_END        = 2019
_PROJ_END        = 2099
_SMOOTH_WINDOW   = 6
_GEE_REF_EPC     = 1
_GEE_REF_DWTYPE  = 6
_GEE_REF_DWAGE   = 7
_GEE_REF_COOL    = 0
_MIN_EPC_BAND    = 3   # minimum dwellings per EPC band for a reliable projection
_MAX_COEF        = 100 # maximum tolerated |coefficient| before treating model as unreliable

# NB: this Figure 7 block uses the 28 UKCP18 *GCM* members (test_data/GCM, formerly
# "UKCP tasmax"); Part 3 (efus_climate_projection.py) uses the 16 RCM members instead.
_NC_GLOB = os.path.join(ROOT, "test_data", "GCM",
                        "tasmax_rcp85_land-gcm_uk_region_*_day_*.nc")

# Fresh read of gee_coefs with original term names (not clim: normalised)
_gee_coefs_proj  = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_coefs.csv"))
_centres         = pd.read_csv(os.path.join(ANALYSIS_DIR, "climate_centres.csv"))
_T2DMMT_OBS_MEAN = float(_centres.loc[_centres.variable == "T_2DMMT", "centre"].iloc[0])

# --- Stock profiles: all observed dwellings -----------------------------------
_dwelling_profiles = (
    daily_df[["CaseID", "dwtype_efus", "dwage_efus", "EPceeb12e_efus", "AnyCooling"]]
    .drop_duplicates("CaseID")
    .reset_index(drop=True)
)
_epc_arr    = _dwelling_profiles["EPceeb12e_efus"].values.astype(int)
_dwtype_arr = _dwelling_profiles["dwtype_efus"].values.astype(int)
_dwage_arr  = _dwelling_profiles["dwage_efus"].values.astype(int)
_cool_arr   = _dwelling_profiles["AnyCooling"].values.astype(int)

# --- Reliability gate: check each threshold before projecting -----------------
def _proj_reliable(thr):
    """Return (True, '') or (False, reason) for a given threshold."""
    mname = f"gee_excprop_{thr}"
    if mname not in _gee_coefs_proj.model.values:
        return False, f"{mname} absent (GEE did not converge)"
    maxc = _gee_coefs_proj[_gee_coefs_proj.model == mname]["estimate"].abs().max()
    if maxc >= _MAX_COEF:
        return False, f"{mname} diverged (max|coef|={maxc:.1e} ≥ {_MAX_COEF})"
    epc_counts = _dwelling_profiles.groupby("EPceeb12e_efus").size()
    thin = epc_counts[epc_counts < _MIN_EPC_BAND]
    if len(thin):
        bands = ", ".join(f"EPC{k}(n={v})" for k, v in thin.items())
        return False, f"too few dwellings in {bands} (min={_MIN_EPC_BAND})"
    return True, ""

_valid_thresholds = []
for _thr in _PROJ_THRESHOLDS:
    _ok, _reason = _proj_reliable(_thr)
    if _ok:
        _valid_thresholds.append(_thr)
    else:
        print(f"  Skipping {_thr}°C projection: {_reason}")

if not _valid_thresholds:
    print("  No thresholds reliable — skipping Figure 7.")
else:

    def _gee_coef(model, term):
        row = _gee_coefs_proj[(_gee_coefs_proj.model == model) &
                              (_gee_coefs_proj.term  == term)]
        return float(row.estimate.iloc[0]) if len(row) else 0.0

    def _precompute_stock(model):
        n      = len(_dwelling_profiles)
        b0     = _gee_coef(model, "(Intercept)")
        b_clim = _gee_coef(model, "T_2DMMT_c")
        intercepts = np.empty(n)
        slopes     = np.empty(n)
        for i in range(n):
            epc    = _epc_arr[i];  dwtype = _dwtype_arr[i]
            dwage  = _dwage_arr[i]; cool  = _cool_arr[i]
            b_epc = 0.0 if epc    == _GEE_REF_EPC    else _gee_coef(model, f"EPC{epc}")
            b_int = 0.0 if epc    == _GEE_REF_EPC    else _gee_coef(model, f"T_2DMMT_c:EPC{epc}")
            b_dwt = 0.0 if dwtype == _GEE_REF_DWTYPE else _gee_coef(model, f"dwtype{dwtype}")
            b_dwa = 0.0 if dwage  == _GEE_REF_DWAGE  else _gee_coef(model, f"dwage{dwage}")
            b_coo = 0.0 if cool   == _GEE_REF_COOL   else _gee_coef(model, "cooling1")
            intercepts[i] = b0 + b_epc + b_dwt + b_dwa + b_coo
            slopes[i]     = b_clim + b_int
        return intercepts, slopes

    def _predict_stock_seasonal(t2c_arr, intercepts, slopes):
        eta = intercepts[:, np.newaxis] + slopes[:, np.newaxis] * t2c_arr[np.newaxis, :]
        p   = 1.0 / (1.0 + np.exp(-eta))
        return p.mean(axis=1) * 100.0

    # Pre-compute stock intercepts/slopes for each reliable threshold model
    _stock_cache = {
        f"gee_excprop_{thr}": _precompute_stock(f"gee_excprop_{thr}")
        for thr in _valid_thresholds
    }

    _CFTIME = xr.coders.CFDatetimeCoder(use_cftime=True)

    def _extract_maysep_t2dmmt(nc_path, region=_LONDON_REGION):
        ds     = xr.open_dataset(nc_path, decode_times=_CFTIME)
        tasmax = ds.sel(region=region).isel(ensemble_member=0)["tasmax"].values
        times  = ds.time.values
        ds.close()
        months = np.array([t.month for t in times])
        years  = np.array([t.year  for t in times])
        result = {}
        for yr in np.unique(years):
            ms_idx  = np.where((months >= 5) & (months <= 9) & (years == yr))[0]
            apr_idx = np.where((months == 4) & (years == yr))[0]
            if len(ms_idx) < 100:
                continue
            prev   = tasmax[apr_idx[-1]] if len(apr_idx) > 0 else tasmax[ms_idx[0]]
            vals   = np.concatenate([[prev], tasmax[ms_idx]])
            result[yr] = 0.5 * (vals[:-1] + vals[1:])
        return result

    # Pass 1: extract daily T_2DMMT and per-member 1981-2010 baselines
    _members = {}
    for _nc in sorted(glob.glob(_NC_GLOB)):
        _ds = xr.open_dataset(_nc, decode_times=_CFTIME)
        _m  = int(_ds.ensemble_member.values[0])
        _ds.close()
        _yrd  = _extract_maysep_t2dmmt(_nc)
        _base = float(np.concatenate([v for yr, v in _yrd.items()
                                      if _BASELINE_START <= yr <= _BASELINE_END]).mean())
        _members[_m] = (_yrd, _base)

    _gcm_mean_base = float(np.mean([b for _, b in _members.values()]))
    _correction    = _T2DMMT_OBS_MEAN - _gcm_mean_base

    # Pass 2: predict seasonal exceedance % for each member × year × threshold × EPC
    _records = []
    for _m, (_yrd, _base) in _members.items():
        for _yr, _t2 in _yrd.items():
            if not (_HIST_START <= _yr <= _PROJ_END):
                continue
            _t2c = _t2 - _base - _correction
            for _thr in _valid_thresholds:
                _mod  = f"gee_excprop_{_thr}"
                _ints, _slps = _stock_cache[_mod]
                _seasonal_dw = _predict_stock_seasonal(_t2c, _ints, _slps)
                for _epc in _EPC_LEVELS:
                    _mask = _epc_arr == _epc
                    if _mask.sum() == 0:
                        continue
                    _records.append(dict(member=_m, year=_yr, threshold=_thr, epc=_epc,
                                         mean_exceed_pct=float(_seasonal_dw[_mask].mean())))

    _proj = pd.DataFrame(_records)
    # Written under a distinct name: climate_projection.csv is the authoritative Part 3
    # (RCM, 16-member) output of efus_climate_projection.py and must not be overwritten here.
    _proj.to_csv(os.path.join(ANALYSIS_DIR, "climate_projection_gcm.csv"), index=False)

    # Observed 2018 validation markers (stock-aggregated)
    _t2_obs  = daily_df.groupby("date_key")["T_2DMMT"].mean().values
    _t2c_obs = _t2_obs - _T2DMMT_OBS_MEAN
    _obs_rows = []
    for _thr in _valid_thresholds:
        _mod  = f"gee_excprop_{_thr}"
        _ints, _slps = _stock_cache[_mod]
        _seasonal_dw = _predict_stock_seasonal(_t2c_obs, _ints, _slps)
        for _epc in _EPC_LEVELS:
            _mask = _epc_arr == _epc
            if _mask.sum() == 0:
                continue
            _obs_rows.append(dict(threshold=_thr, epc=_epc,
                                  mean_exceed_pct=float(_seasonal_dw[_mask].mean())))
    _obs_df = pd.DataFrame(_obs_rows)

    _ens = (
        _proj.groupby(["threshold", "epc", "year"])["mean_exceed_pct"]
        .agg(ens_mean = "mean",
             ens_p05  = lambda x: np.percentile(x,  5),
             ens_p10  = lambda x: np.percentile(x, 10),
             ens_p90  = lambda x: np.percentile(x, 90),
             ens_p95  = lambda x: np.percentile(x, 95))
        .reset_index()
    )

    _n_panels = len(_valid_thresholds)
    _fig7_w   = 5.8 * _n_panels + 0.4
    fig7, axes7 = plt.subplots(1, _n_panels, figsize=(_fig7_w, 5.0),
                                constrained_layout=True, squeeze=False)
    axes7 = axes7[0]

    for ax7, thr in zip(axes7, _valid_thresholds):
        _epc_present = sorted(_proj[_proj.threshold == thr].epc.unique())
        for epc in _epc_present:
            col = COLORS[3 - (_EPC_LEVELS.index(epc))]
            sub = _ens[(_ens.threshold == thr) & (_ens.epc == epc)].sort_values("year")
            yrs = sub.year.values

            def _smooth(arr):
                return (pd.Series(arr)
                        .rolling(_SMOOTH_WINDOW, center=True, min_periods=5)
                        .mean().values)

            ax7.fill_between(yrs, _smooth(sub.ens_p05.values), _smooth(sub.ens_p95.values),
                             color=col, alpha=0.10, linewidth=0, zorder=1)
            ax7.fill_between(yrs, _smooth(sub.ens_p10.values), _smooth(sub.ens_p90.values),
                             color=col, alpha=0.20, linewidth=0, zorder=2)
            ax7.plot(yrs, _smooth(sub.ens_mean.values), color=col, linewidth=1.8, zorder=4,
                     label=f"EPC {_EPC_LABELS_PROJ[epc]}")

            for _, _mg in _proj[(_proj.threshold == thr) & (_proj.epc == epc)].groupby("member"):
                _ms = _mg.sort_values("year")
                ax7.plot(_ms.year, _ms.mean_exceed_pct,
                         color=col, linewidth=0.2, alpha=0.15, zorder=0)

        for epc in _epc_present:
            col  = COLORS[3 - (_EPC_LEVELS.index(epc))]
            _row = _obs_df[(_obs_df.threshold == thr) & (_obs_df.epc == epc)]
            if len(_row):
                ax7.scatter(2018, float(_row.mean_exceed_pct.iloc[0]), marker="D", s=30, zorder=5,
                            color="none", edgecolors=col, linewidths=1.3)

        ax7.axhline(CRIT_PCT, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=3)
        ax7.set_xlabel("Year", fontsize=8)
        ax7.set_ylabel(r"Occupied hours above threshold, May--Sep (\%)", fontsize=8)
        ax7.set_title(f"Threshold {thr}$^\circ$C --- {ROOM_LABEL}s, {SEASON_LABEL}", fontsize=8)
        ax7.set_xlim(_HIST_START + _SMOOTH_WINDOW // 2, _PROJ_END - _SMOOTH_WINDOW // 2)

    for ax7 in axes7:
        ax7.relim(); ax7.autoscale_view()
        _ylo, _yhi = ax7.get_ylim()
        ax7.set_ylim(min(_ylo, 0), _yhi)

    _epc_present_all = sorted(_proj.epc.unique())
    _epc_handles = [mlines.Line2D([], [], color=COLORS[3 - (_EPC_LEVELS.index(e))], linewidth=1.8,
                                   label=f"EPC {_EPC_LABELS_PROJ[e]}")
                    for e in _epc_present_all]
    _style_handles = [
        mlines.Line2D([], [], color="gray", linewidth=1.8,
                      label="Ensemble mean (28 members)"),
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.30,
                      label="10th--90th percentile"),
        plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.15,
                      label="5th--95th percentile"),
        mlines.Line2D([], [], color="gray", linewidth=0.4, alpha=0.5,
                      label="Individual members"),
        mlines.Line2D([], [], color="gray", marker="D", markersize=5, linestyle="none",
                      markerfacecolor="none", markeredgewidth=1.1,
                      label="Obs. May--Sep 2018 (validation)"),
    ]
    axes7[0].legend(handles=_epc_handles + _style_handles,
                    fontsize=5.5, frameon=True, loc="upper left", ncol=2)

    _n_dw = len(_dwelling_profiles)
    fig7.suptitle(
        r"Projected \% of occupied hours exceeding threshold by EPC band"
        r" --- UKCP18 RCP8.5, 28 members"
        "\n"
        rf"Overheating criterion: $>${CRIT_PCT}\%"
        rf" $\cdot$ Stock-averaged over all {_n_dw} dwellings per EPC band"
        r" $\cdot$ Diamonds = observed 2018 (validation)",
        fontsize=7.5,
    )
    save(fig7, "climate_projection.svg")

print("\nAll figures saved.")
