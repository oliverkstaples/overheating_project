#!/usr/bin/env python
"""TITLE-PAGE plot for the climate projection.

Simplified companion to the description title plot: one clear message --
more energy-efficient homes carry higher overheating risk, and that risk
rises over time under UKCP18 RCP8.5.

Reads the already-computed climate_projection.csv (no NetCDF needed) and
reduces the two-panel ensemble figure to a single clean schematic:
efficiency as a continuous red->blue colour gradient (no discrete EPC
legend), arrow-style axis labels, top/right spines removed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

plt.style.use(["science", "nature", "bright"])

ROOM      = "livingroom"
THRESHOLD = 26
SMOOTH    = 11          # years; clean schematic curves

# Presentation sizing
LABEL_FS = 13
TICK_FS  = 11
CBAR_FS  = 12

CSV  = f"analysis/{ROOM}/august/climate_projection.csv"
OUTDIR = "plots/efus2017/london_livingroom_1month/description"

df = pd.read_csv(CSV)
df = df[df["threshold"] == THRESHOLD]

# Ensemble mean across climate members, then smooth over years
ens = (
    df.groupby(["epc", "year"])["mean_exceed_pct"]
    .mean()
    .reset_index()
)

cmap = LinearSegmentedColormap.from_list("efficiency", ["#b40426", "#3b4cc0"])
norm = Normalize(0, 1)

fig, ax = plt.subplots(figsize=(2.6, 2.5))

# 1 = C+ (most efficient) ... 4 = F/G (least efficient)
for epc in [4, 3, 2, 1]:

    efficiency = (4 - epc) / 3            # 0 = least efficient, 1 = most efficient

    s = ens[ens["epc"] == epc].sort_values("year")
    y = s["mean_exceed_pct"].rolling(SMOOTH, center=True, min_periods=1).mean()

    # trim smoothing edge effects
    keep = SMOOTH // 2
    ax.plot(
        s["year"].iloc[keep:-keep],
        y.iloc[keep:-keep],
        linewidth=2.2,
        color=cmap(norm(efficiency)),
    )

# Same format as the description title plot: drop top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(which="both", top=False, right=False)
ax.tick_params(labelsize=TICK_FS, length=4, width=1.0)

ax.set_xlabel(r"Time $\rightarrow$", fontsize=LABEL_FS)
ax.set_ylabel(r"Overheating risk $\rightarrow$", fontsize=LABEL_FS)

# Continuous efficiency colour bar on top, in place of the EPC legend
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", location="top",
    fraction=0.06, pad=0.18,
)
cbar.set_label(r"More efficient $\rightarrow$", fontsize=CBAR_FS)
cbar.set_ticks([])
cbar.outline.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/title_climate_projection_vs_efficiency_{ROOM}.svg",
            bbox_inches="tight")
plt.savefig(f"{OUTDIR}/title_climate_projection_vs_efficiency_{ROOM}.png",
            dpi=200, bbox_inches="tight")
print("wrote", f"{OUTDIR}/title_climate_projection_vs_efficiency_{ROOM}.svg")
