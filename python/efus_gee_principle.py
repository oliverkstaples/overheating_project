#!/usr/bin/env python
"""
"GEE fitted vs observed" figure, all three overheating thresholds (26/27/28 C)
on one axes in different colours.

For each threshold the GEE's mean fitted probability per 2DMMT bin (line) is
shown against the empirical overheating proportion in that bin (dots).  Uses the
London living-room May--Sep arm (analysis/livingroom).  Style matches the LME
diagnostics notebooks (scienceplots, usetex).

Output: plots/efus2017/london_livingroom_4month/impact_profiles/gee_fitted_vs_observed.svg
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

plt.style.use(["science", "nature", "bright"])
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"figure.dpi": 120})

DAILY  = "analysis/livingroom/daily_overheating.parquet"
FITTED = "analysis/livingroom/gee_fitted.csv"
OUT    = "plots/efus2017/london_livingroom_4month/impact_profiles"
os.makedirs(OUT, exist_ok=True)

THRESHOLDS = [26, 27, 28]
COLOURS    = {26: "#2E67D0", 27: "#E08214", 28: "#C0392B"}   # cool -> warm
NBIN = 12

# ── data: join the GEE's fitted prob / observed flag to each day's 2DMMT ───────
daily = pd.read_parquet(DAILY)[["CaseID", "date", "T_2DMMT"]]
daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
fit = pd.read_csv(FITTED)
fit["date"] = pd.to_datetime(fit["date"]).dt.normalize()


def binned(thr):
    d = fit[fit.model == f"gee_overheat_{thr}"].merge(
        daily, on=["CaseID", "date"], how="inner").dropna(
        subset=["T_2DMMT", "fitted", "observed"])
    edges = np.quantile(d["T_2DMMT"], np.linspace(0, 1, NBIN + 1))
    edges[-1] += 1e-6
    d = d.assign(bin=np.digitize(d["T_2DMMT"], edges) - 1)
    return d.groupby("bin").agg(
        x=("T_2DMMT", "mean"),
        emp=("observed", "mean"),
        fit=("fitted", "mean"),
    ).reset_index()


fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)

for thr in THRESHOLDS:
    g = binned(thr)
    c = COLOURS[thr]
    ax.plot(g["x"], g["fit"], color=c, linewidth=1.7, zorder=3)
    ax.scatter(g["x"], g["emp"], s=16, color=c, zorder=4,
               edgecolors="white", linewidths=0.4)

ax.set_xlabel(r"$\widetilde{\mathrm{2DMMT}}$ ($^\circ$C)", fontsize=8)
ax.set_ylabel(r"$P(\mathrm{overheat})$", fontsize=8)
ax.set_title(r"Fitted GEE vs.\ observed (London LR, May--Sep)", fontsize=8)
ax.set_ylim(-0.04, 1.04)
ax.tick_params(labelsize=7)

# legend: colour = threshold, plus a line/marker key for fitted vs empirical
thr_handles = [Line2D([], [], color=COLOURS[t], linewidth=1.7,
                      label=rf"${t}\,^\circ$C") for t in THRESHOLDS]
key_handles = [Line2D([], [], color="black", linewidth=1.7, label="GEE fitted"),
               Line2D([], [], color="black", marker="o", linestyle="none",
                      markersize=4, markeredgecolor="white", markeredgewidth=0.4,
                      label="Empirical (binned)")]
leg1 = ax.legend(handles=thr_handles, fontsize=6.5, frameon=False,
                 loc="upper left", title=r"Threshold", title_fontsize=6.5)
ax.add_artist(leg1)
ax.legend(handles=key_handles, fontsize=6.5, frameon=False, loc="lower right")

fig.savefig(f"{OUT}/gee_fitted_vs_observed.svg", bbox_inches="tight")
fig.savefig(f"{OUT}/gee_fitted_vs_observed.png", dpi=200, bbox_inches="tight")
print("wrote", f"{OUT}/gee_fitted_vs_observed.svg")
