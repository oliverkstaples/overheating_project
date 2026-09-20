#!/usr/bin/env python
"""
Out-of-sample validation of the exceedance-proportion GEE (gee_excprop_t),
London livingroom.  The model was fitted on AUGUST; here we apply its
coefficients to the rest of the season (May, June, July, September) of the same
dwellings and compare predicted vs observed exceedance proportions.

Outputs:
  plots/efus2017/london_livingroom_1month/impact_profiles/gee_excprop_validation_oos.svg
  prints in-sample vs out-of-sample Brier scores and calibration-in-the-large.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "nature", "bright"])

ANALYSIS = "analysis/livingroom"             # 4-month May-Sep dataset
AUG      = "analysis/livingroom/august"      # fitted-model coefficients/centring
OUT      = "plots/efus2017/london_livingroom_1month/impact_profiles"
THRESHOLDS = [26, 27, 28]

coefs    = pd.read_csv(f"{AUG}/gee_coefs.csv")
T_CENTRE = float(pd.read_csv(f"{AUG}/climate_centres.csv")
                 .query("variable=='T_2DMMT'")["centre"].iloc[0])

df = pd.read_parquet(f"{ANALYSIS}/daily_overheating.parquet")
df["month"] = pd.to_datetime(df["date"]).dt.month
df["t2c"]   = df["T_2DMMT"] - T_CENTRE
oos = df[df.month != 8].copy()   # out-of-sample: May, Jun, Jul, Sep
ins = df[df.month == 8].copy()   # in-sample reference: August
print(f"in-sample (Aug): {len(ins)} dwelling-days; "
      f"out-of-sample (May-Jul,Sep): {len(oos)} dwelling-days, "
      f"{oos.CaseID.nunique()} dwellings\n")

EPC_LABELS  = {1: "C+", 2: "D", 3: "E", 4: "F/G"}
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def predicted_prop(data, thr):
    """Predicted exceedance proportion from the August gee_excprop_{thr} coefs."""
    d = coefs[coefs.model == f"gee_excprop_{thr}"].set_index("term")["estimate"]
    g = lambda t: float(d[t]) if t in d.index else 0.0
    t2c = data["t2c"].to_numpy()
    eta = g("(Intercept)") + g("T_2DMMT_c") * t2c
    eta += data["EPceeb12e_efus"].map(lambda e: g(f"EPC{e}")).to_numpy()
    eta += t2c * data["EPceeb12e_efus"].map(lambda e: g(f"T_2DMMT_c:EPC{e}")).to_numpy()
    eta += data["dwtype_efus"].map(lambda v: g(f"dwtype{v}")).to_numpy()
    eta += data["dwage_efus"].map(lambda v: g(f"dwage{v}")).to_numpy()
    eta += np.where(data["AnyCooling"].to_numpy() == 1, g("cooling1"), 0.0)
    return 1.0 / (1.0 + np.exp(-eta))


def brier(pred, obs, w=None):
    if w is None:
        return float(np.mean((pred - obs) ** 2))
    return float(np.average((pred - obs) ** 2, weights=w))


# ── Metrics ───────────────────────────────────────────────────────────────────
print(f"{'thr':>4} {'set':>10} {'mean_obs':>9} {'mean_pred':>10} {'Brier':>9} {'Brier_w':>9}")
for thr in THRESHOLDS:
    for name, data in (("in (Aug)", ins), ("out", oos)):
        obs  = data[f"daily_exceed_prop_{thr}"].to_numpy()
        pred = predicted_prop(data, thr)
        w    = data["occ_hours"].to_numpy()
        print(f"{thr:>4} {name:>10} {obs.mean():>9.4f} {pred.mean():>10.4f} "
              f"{brier(pred,obs):>9.4f} {brier(pred,obs,w):>9.4f}")
print()

# ── 90% CI of a mean, cluster-robust by dwelling (CaseID) ──────────────────────
def mean_ci(y, cl, z=1.645):
    y = np.asarray(y, float)
    n = len(y); m = y.mean()
    s = pd.Series(y - m).groupby(np.asarray(cl)).sum().to_numpy()
    se = np.sqrt((s ** 2).sum()) / n           # cluster-robust SE of the mean
    return m, max(m - z * se, 0.0), m + z * se

def grouped_ci(g, valcol="obs", clcol="CaseID"):
    out = []
    for key, grp in g:
        m, lo, hi = mean_ci(grp[valcol], grp[clcol])
        out.append((key, m, lo, hi, grp["pred"].mean(),
                    grp["T_2DMMT"].mean() if "T_2DMMT" in grp else np.nan, len(grp)))
    return pd.DataFrame(out, columns=["key", "mo", "lo", "hi", "mp", "t", "n"])

# ── Figure: rows = thresholds, cols = calibration | by-EPC | dose-response ─────
fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5), constrained_layout=True)
for ri, thr in enumerate(THRESHOLDS):
    obs  = oos[f"daily_exceed_prop_{thr}"].to_numpy()
    pred = predicted_prop(oos, thr)
    sub  = oos.assign(obs=obs, pred=pred)

    # (col 0) calibration: binned predicted vs observed (90% CI on observed)
    axc = axes[ri, 0]
    sub["pb"] = pd.qcut(sub["pred"], q=10, duplicates="drop")
    cal = grouped_ci(sub.groupby("pb", observed=True))
    axc.plot([0, 100], [0, 100], "k--", lw=0.8)
    axc.errorbar(cal.mp * 100, cal.mo * 100,
                 yerr=[(cal.mo - cal.lo) * 100, (cal.hi - cal.mo) * 100],
                 fmt="o", ms=3.5, color=COLORS[0], ecolor=COLORS[0],
                 elinewidth=0.8, capsize=2, zorder=3)
    axc.set_xlabel(r"Predicted exceedance (\%)", fontsize=8)
    axc.set_ylabel(r"Observed exceedance (\%, 90\% CI)", fontsize=8)
    axc.set_title(f"Calibration --- {thr}$^\\circ$C", fontsize=9)

    # (col 1) mean predicted vs observed by EPC band (90% CI on observed)
    axb = axes[ri, 1]
    eg = grouped_ci(sub.groupby("EPceeb12e_efus")).sort_values("key")
    x = np.arange(len(eg)); w = 0.38
    axb.bar(x - w/2, eg.mo * 100, w, color=COLORS[0], alpha=0.85, label="Observed",
            yerr=[(eg.mo - eg.lo) * 100, (eg.hi - eg.mo) * 100],
            capsize=2, error_kw=dict(elinewidth=0.8))
    axb.bar(x + w/2, eg.mp * 100, w, color=COLORS[0], alpha=0.40, hatch="//",
            label="Predicted")
    axb.set_xticks(x)
    axb.set_xticklabels([EPC_LABELS.get(int(e), str(e)) for e in eg.key], fontsize=7)
    axb.set_xlabel("EPC band", fontsize=8)
    axb.set_ylabel(r"Mean exceedance (\%, 90\% CI)", fontsize=8)
    axb.set_title(f"By EPC --- {thr}$^\\circ$C", fontsize=9)
    if ri == 0:
        axb.legend(fontsize=7, frameon=True)

    # (col 2) dose-response: observed (binned, 90% CI) vs predicted across 2DMMT
    axd = axes[ri, 2]
    sub["tb"] = pd.cut(sub["T_2DMMT"], bins=np.arange(10, 35, 2))
    dr = grouped_ci(sub.groupby("tb", observed=True))
    axd.plot(dr.t, dr.mp * 100, "-", color=COLORS[1], lw=1.5, label="Predicted")
    axd.errorbar(dr.t, dr.mo * 100,
                 yerr=[(dr.mo - dr.lo) * 100, (dr.hi - dr.mo) * 100],
                 fmt="o", ms=3.5, color=COLORS[0], ecolor=COLORS[0],
                 elinewidth=0.8, capsize=2, zorder=3, label="Observed")
    axd.set_xlabel(r"Outdoor 2DMMT ($^\circ$C)", fontsize=8)
    axd.set_ylabel(r"Mean exceedance (\%, 90\% CI)", fontsize=8)
    axd.set_title(f"Dose-response --- {thr}$^\\circ$C", fontsize=9)
    if ri == 0:
        axd.legend(fontsize=7, frameon=True, loc="upper left")

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/gee_excprop_validation_oos.svg", bbox_inches="tight")
print(f"wrote {OUT}/gee_excprop_validation_oos.svg")
