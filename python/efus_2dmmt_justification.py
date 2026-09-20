#!/usr/bin/env python
"""
Justification plot for using 2DMMT (rather than the instantaneous outdoor
temperature) as the climate predictor in the overheating GEE.

Overlays, on a single axes, hourly indoor temperature T_in against:
  - the concurrent hourly outdoor temperature T_out (grey), and
  - that day's 2-day running mean maximum outdoor temperature, 2DMMT (blue),
each with an OLS fit line and its R^2.  The tighter 2DMMT cloud / higher R^2
reflects the building thermal lag (and the diurnal noise in raw T_out) that
motivates 2DMMT in the GEE.

Style matches the LME diagnostics notebooks (scienceplots science/nature/bright,
usetex, small fonts, rasterised scatter).

Output: plots/efus2017/london_livingroom_4month/impact_profiles/gee_2dmmt_justification.svg
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "nature", "bright"])
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"figure.dpi": 120})

HOURLY = "efus_indoor_outdoor_livingroom.parquet"
DAILY  = "analysis/livingroom/daily_overheating.parquet"   # defines population + T_2DMMT
OUT    = "plots/efus2017/london_livingroom_4month/impact_profiles"
os.makedirs(OUT, exist_ok=True)

BLUE, GREY = "#2E67D0", "#9A9A9A"
Y = "T_in"   # hourly indoor temperature (deg C)

# Join each hour's T_in / T_out to that dwelling-day's 2DMMT.  The inner join also
# restricts the hourly data to the analysis population (London living room, May-Sep).
hourly = pd.read_parquet(HOURLY)
hourly["date"] = hourly["hour"].dt.normalize()
daily = pd.read_parquet(DAILY)[["CaseID", "date", "T_2DMMT"]]
daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
df = hourly.merge(daily, on=["CaseID", "date"], how="inner").dropna(
    subset=["T_in", "T_out", "T_2DMMT"])


def ols_fit(x, y):
    """Return (slope, intercept, r2) for a simple OLS of y on x."""
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return b, a, r ** 2


fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)

series = [
    ("T_out",   GREY, r"Hourly $T_{\mathrm{out}}$"),
    ("T_2DMMT", BLUE, r"2DMMT ($\widetilde{\mathrm{2DMMT}}$)"),
]

xline = np.linspace(
    df[["T_out", "T_2DMMT"]].min().min(),
    df[["T_out", "T_2DMMT"]].max().max(),
    200,
)

for col, c, lab in series:
    d = df[[col, Y]].dropna()
    ax.scatter(d[col], d[Y], s=2, alpha=0.05, color=c, rasterized=True,
               edgecolors="none", zorder=1)
    b, a, r2 = ols_fit(d[col].values, d[Y].values)
    ax.plot(xline, a + b * xline, color=c, linewidth=1.6, zorder=3,
            label=rf"{lab}: $R^2={r2:.3f}$")

ax.set_xlabel(r"Outdoor temperature ($^\circ$C)", fontsize=8)
ax.set_ylabel(r"Hourly indoor $T_{\mathrm{in}}$ ($^\circ$C)", fontsize=8)
ax.set_title(r"London living room (May--Sep): hourly $T_{\mathrm{in}}$ vs.\ outdoor measure",
             fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, loc="upper left")

fig.savefig(f"{OUT}/gee_2dmmt_justification.svg", bbox_inches="tight")
fig.savefig(f"{OUT}/gee_2dmmt_justification.png", dpi=200, bbox_inches="tight")
print("wrote", f"{OUT}/gee_2dmmt_justification.svg")
for col, _, lab in series:
    d = df[[col, Y]].dropna()
    b, a, r2 = ols_fit(d[col].values, d[Y].values)
    print(f"  {lab:24s} slope={b:.3f} R2={r2:.3f}")
