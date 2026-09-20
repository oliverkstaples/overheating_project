"""
Impact profiles for the mean-temperature night criterion (bedroom only).

Plots:
  impact_epc.svg          violin: mean night T_in by EPC band
  impact_dwtype.svg       violin: mean night T_in by dwelling type
  impact_binary.svg       violin: mean night T_in by binary chars
  impact_overheat_rate.svg bar: seasonal failure rate by building char
  gee_forest.svg          GEE log-odds forest plot (2 panels)
  gee_prob_curves.svg     P(mean night >= thr) vs T_2DMMT by EPC

Usage:
  python python/efus_overheating_impact_profiles_meannight.py --region london
  python python/efus_overheating_impact_profiles_meannight.py --region southwest
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import scienceplots  # noqa: F401

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--region", type=str, default="london",
                    choices=["london", "southwest"])
args = parser.parse_args()

REGION  = args.region
RLAB    = {"london": "London", "southwest": "South West"}[REGION]
_reg_pre = "" if REGION == "london" else REGION
path_parts = list(filter(None, [_reg_pre, "bedroom/mean_criterion"]))
ANALYSIS_DIR = os.path.join(ROOT, "analysis", *path_parts)

OUTDIR = os.path.join(ROOT, "plots", "efus2017",
                      f"{REGION}_bedroom_4month", "impact_profiles_meannight")
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use(["science", "nature", "bright"])
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ── Load data ─────────────────────────────────────────────────────────────────
daily   = pd.read_parquet(os.path.join(ANALYSIS_DIR, "daily_overheating.parquet"))
summary = pd.read_parquet(os.path.join(ANALYSIS_DIR, "dwelling_overheating_summary.parquet"))
gee_coefs = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_coefs.csv"))
pred_grid = pd.read_csv(os.path.join(ANALYSIS_DIR, "gee_pred_grid.csv"))
cc        = pd.read_csv(os.path.join(ANALYSIS_DIR, "climate_centres.csv"))
T2DMMT_CENTRE = float(cc.loc[cc["variable"] == "T_2DMMT", "centre"].values[0])

THRESHOLDS = [26, 27]
OVERHEAT_NIGHTS_THRESHOLD = 7  # >7 nights = seasonal fail

def dark(c):
    return tuple(np.array(mcolors.to_rgb(c)) * 0.45)

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Group definitions ─────────────────────────────────────────────────────────
EPC_ORDER  = [4, 3, 2, 1]
EPC_LABELS = {1: "C+\n", 2: "D\n", 3: "E\n", 4: "F/G\n"}
DW_ORDER   = [1, 2, 3, 4, 5, 6]
DW_LABELS  = {1: "Detached", 2: "Semi-det.", 3: "End-terr.",
              4: "Mid-terr.", 5: "Bungalow",  6: "Flat"}
BINARY_CHARS = [
    ("CavityWall",          {0: "Solid wall",   1: "Cavity wall"}),
    ("InsulatedWalls_efus", {0: "No insul.",     1: "Insulated"}),
    ("FullyDblGlz_efus",    {0: "Part. glazing", 1: "Full dbl. glaz."}),
    ("AnyCooling",          {0: "No cooling",   1: "Cooling"}),
]


# ── Violin helper ─────────────────────────────────────────────────────────────
def draw_violin(ax, data_list, xlabels, ylabel, is_pct=False):
    valid = [(lab, arr) for lab, arr in zip(xlabels, data_list) if len(arr) >= 2]
    if not valid:
        return
    xlabels_filt, data_list = zip(*valid)
    xlabels   = list(xlabels_filt)
    data_list = list(data_list)
    n = len(data_list)
    positions = list(range(1, n + 1))

    parts = ax.violinplot(data_list, positions=positions,
                          showmeans=False, showmedians=False, showextrema=True)

    dark_colors = []
    for i, pc in enumerate(parts["bodies"]):
        color      = COLORS[i % len(COLORS)]
        dark_color = dark(color)
        dark_colors.append(dark_color)
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.45)

    parts["cmins"].set_visible(False)
    parts["cmaxes"].set_visible(False)
    parts["cbars"].set_linewidth(1.2)
    parts["cbars"].set_color(dark_colors)

    for i, data in enumerate(data_list, start=1):
        if len(data) < 2:
            continue
        color      = COLORS[i - 1]
        dark_color = dark_colors[i - 1]
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.add_patch(plt.Rectangle(
            (i - 0.06, q1), 0.12, q3 - q1,
            facecolor=color, edgecolor=dark_color,
            linewidth=1.2, zorder=4,
        ))
        ax.plot([i - 0.049, i + 0.049], [med, med],
                color="white", linewidth=1.2, solid_capstyle="butt", zorder=5)
        unit = "%" if is_pct else "°C"
        ax.text(i + 0.4, med, f"{med:.1f}{unit}",
                fontsize=5, va="center", ha="center", color=dark_color)
        ax.text(i, np.max(data) + 0.3,
                f"IQR\n[{q1:.1f}, {q3:.1f}]",
                fontsize=5, va="bottom", ha="center", color=dark_color)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ymin = min(np.min(d) for d in data_list if len(d) > 0) - 1
    ymax = max(np.max(d) for d in data_list if len(d) > 0) + 5
    ax.set_ylim(ymin, ymax)
    if is_pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: EPC band — mean night T_in violin
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: EPC band violin...")

fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.8), constrained_layout=True)
for ax, thr in zip(axes, THRESHOLDS):
    groups = [daily.loc[daily["EPceeb12e_efus"] == b, "mean_tin_night"].dropna().values
              for b in EPC_ORDER]
    draw_violin(ax, groups, [EPC_LABELS[b] for b in EPC_ORDER],
                r"Mean night $T_{in}$ (°C)")
    ax.axhline(thr, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
               label=f"Threshold {thr}°C")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlabel("EPC score band")
    ax.set_title(f"Threshold {thr}°C", fontsize=8)

fig.suptitle(
    f"Mean night indoor temperature (22:00--06:59) by EPC band\n"
    f"{RLAB} bedrooms, May--Sep · EFUS 2017",
    fontsize=9,
)
save(fig, "impact_epc.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Dwelling type violin
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Dwelling type violin...")

fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8), constrained_layout=True)
for ax, thr in zip(axes, THRESHOLDS):
    groups = [daily.loc[daily["dwtype_efus"] == d, "mean_tin_night"].dropna().values
              for d in DW_ORDER]
    draw_violin(ax, groups, [DW_LABELS[d] for d in DW_ORDER],
                r"Mean night $T_{in}$ (°C)")
    ax.axhline(thr, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
               label=f"Threshold {thr}°C")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlabel("Dwelling type")
    ax.set_title(f"Threshold {thr}°C", fontsize=8)

fig.suptitle(
    f"Mean night indoor temperature (22:00--06:59) by dwelling type\n"
    f"{RLAB} bedrooms, May--Sep · EFUS 2017",
    fontsize=9,
)
save(fig, "impact_dwtype.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Binary characteristics violin
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Binary characteristics...")

fig, axes = plt.subplots(4, 2, figsize=(6.0, 9.5), constrained_layout=True)

for row, (char, labels_dict) in enumerate(BINARY_CHARS):
    codes   = [0, 1]
    xlabels = [labels_dict[c] for c in codes]
    for col_idx, thr in enumerate(THRESHOLDS):
        ax = axes[row, col_idx]
        groups = [daily.loc[daily[char] == c, "mean_tin_night"].dropna().values
                  for c in codes]
        draw_violin(ax, groups, xlabels, r"Mean night $T_{in}$ (°C)")
        ax.axhline(thr, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        if all(len(g) > 0 for g in groups):
            diff = np.mean(groups[1]) - np.mean(groups[0])
            ax.text(0.97, 0.97, f"$\\Delta$ = {diff:+.2f}°C",
                    transform=ax.transAxes, fontsize=6, ha="right", va="top")
        if row == 0:
            ax.set_title(f"Threshold {thr}°C", fontsize=8)

fig.suptitle(
    f"Mean night indoor temperature by building characteristics\n"
    f"{RLAB} bedrooms, May--Sep · EFUS 2017",
    fontsize=9,
)
save(fig, "impact_binary.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Seasonal failure rate bar charts (>7 nights above threshold)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Seasonal failure rates...")

RATE_CONFIGS = [
    ("EPceeb12e_efus", EPC_ORDER,  [EPC_LABELS[b] for b in EPC_ORDER],  "EPC band"),
    ("dwtype_efus",    DW_ORDER,   [DW_LABELS[d] for d in DW_ORDER],    "Dwelling type"),
    ("AnyCooling",     [0, 1],     ["No cooling", "Cooling"],            "Cooling"),
    ("CavityWall",     [0, 1],     ["Solid wall", "Cavity wall"],        "Wall construction"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), constrained_layout=True)

for ax, (char, codes, xlabels, char_title) in zip(axes.flat, RATE_CONFIGS):
    x     = np.arange(len(codes))
    width = 0.22
    offs  = np.array([-0.5, 0.5]) * width

    for j, (thr, offset) in enumerate(zip(THRESHOLDS, offs)):
        fail_col = f"season_overheat_{thr}"
        probs = []
        for code in codes:
            grp = summary[summary[char] == code][fail_col]
            probs.append(100 * grp.mean() if len(grp) > 0 else np.nan)

        col = COLORS[j % len(COLORS)]
        dk  = dark(col)
        bars = ax.bar(x + offset, probs, width=width * 0.9,
                      color=col, alpha=0.6, edgecolor=dk,
                      linewidth=0.8, label=f"$\\geq${thr}°C")
        for bar, p in zip(bars, probs):
            if not np.isnan(p):
                ax.text(bar.get_x() + bar.get_width() / 2, p + 1.5,
                        f"{p:.0f}%", fontsize=5, ha="center",
                        va="bottom", color=dk)

    for xi, code in enumerate(codes):
        n = int((summary[char] == code).sum())
        ax.text(xi, -7, f"n={n}", fontsize=5, ha="center",
                va="top", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(-10, 108)
    ax.set_ylabel("Dwellings meeting overheating criterion (%)")
    ax.set_title(char_title, fontsize=8)
    ax.grid(axis="y")
    ax.legend(fontsize=6, frameon=True, loc="upper right")

fig.suptitle(
    f"Mean-temperature seasonal overheating rate by building characteristic --- {RLAB} bedrooms\n"
    rf"Criterion: $>$7 nights with mean 22:00--06:59 temperature above threshold · EFUS 2017",
    fontsize=8.5,
)
save(fig, "impact_overheat_rate.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: GEE forest plot (log-odds, 2 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: GEE forest plot...")

# Normalise interaction term names
gee_coefs["term"] = gee_coefs["term"].str.replace(
    r"^[^:]+:EPC", "clim:EPC", regex=True)

TERM_META = [
    ("EPC2",      "D vs C+",             r"EPC band" + "\n" + r"(ref = C+)"),
    ("EPC3",      "E vs C+",             None),
    ("EPC4",      "F/G vs C+",           None),
    ("dwtype1",   "Detached",            r"Dwelling type" + "\n" + r"(ref = Flat)"),
    ("dwtype2",   "Semi-det.",           None),
    ("dwtype3",   "End-terrace",         None),
    ("dwtype4",   "Mid-terrace",         None),
    ("dwtype5",   "Bungalow",            None),
    ("dwage1",    "Pre-1919",            r"Build era" + "\n" + r"(ref = Post-1990)"),
    ("dwage2",    "1919--44",            None),
    ("dwage3",    "1945--64",            None),
    ("dwage4",    "1965--74",            None),
    ("dwage5",    "1975--80",            None),
    ("dwage6",    "1981--90",            None),
    ("cooling1",  "Cooling",             r"Cooling" + "\n" + r"(ref = No)"),
    ("clim:EPC2", r"$T_{2DMMT} \times$ D",   r"$T_{2DMMT} \times$ EPC" + "\n" + r"(slope diff.)"),
    ("clim:EPC3", r"$T_{2DMMT} \times$ E",   None),
    ("clim:EPC4", r"$T_{2DMMT} \times$ F/G", None),
]
TERMS  = [t for t, _, _ in TERM_META]
LABELS = {t: l for t, l, _ in TERM_META}
GROUPS = {t: g for t, g, _ in TERM_META}
N    = len(TERMS)
YPOS = list(range(N - 1, -1, -1))

sig_handle   = mlines.Line2D([], [], color="gray", marker="D", markersize=5,
                              linestyle="-", label=r"$p < 0.05$")
insig_handle = mlines.Line2D([], [], color="gray", marker="o", markersize=4,
                              linestyle="-", label=r"$p \geq 0.05$")

def draw_forest_panel(ax, model_name, coefs_df, color):
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
        row  = sub.loc[term]
        est  = float(row["estimate"])
        lo   = float(row["ci_lo"])
        hi   = float(row["ci_hi"])
        pval = float(row["p_san"])
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
    xlim   = ax.get_xlim()
    x_left = xlim[0] - (xlim[1] - xlim[0]) * 0.85
    prev_grp = None
    for yp, term in zip(YPOS, TERMS):
        grp = GROUPS[term]
        if grp and grp != prev_grp:
            ax.text(x_left, yp + 0.75, grp, fontsize=5.5, va="bottom",
                    ha="left", color="gray", style="italic", clip_on=False)
        if grp:
            prev_grp = grp

GEE_PANELS = [
    ("gee_mean_26", r"Log-odds: mean night $T_{in} \geq$ 26°C"),
    ("gee_mean_27", r"Log-odds: mean night $T_{in} \geq$ 27°C"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 7.5),
                         sharey=True, constrained_layout=True)

for ax_i, (ax, (mname, xlabel)) in enumerate(zip(axes, GEE_PANELS)):
    col = COLORS[ax_i % len(COLORS)]
    draw_forest_panel(ax, mname, gee_coefs, col)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(xlabel, fontsize=7)

set_forest_yticks(axes[0])
add_group_labels(axes[0])
fig.legend(handles=[sig_handle, insig_handle], loc="lower center",
           ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.03), frameon=True)
fig.suptitle(
    f"GEE estimates: building characteristics on night mean temperature exceedance --- {RLAB} bedrooms\n"
    r"AR(1) working correlation $\cdot$ sandwich SE $\cdot$ log-odds scale $\cdot$ EFUS 2017",
    fontsize=8,
)
save(fig, "gee_forest.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: GEE probability curves
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: GEE probability curves...")

EPC_LABEL_MAP = {1: "EPC C+", 2: "EPC D", 3: "EPC E", 4: "EPC F/G"}

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), constrained_layout=True)

for ax, thr in zip(axes, THRESHOLDS):
    mname = f"gee_mean_{thr}"
    sub   = pred_grid[pred_grid["model"] == mname]
    epc_levels = sorted(sub["EPC_level"].unique())
    for ci, epc_lvl in enumerate(reversed(epc_levels)):
        grp = sub[sub["EPC_level"] == epc_lvl].sort_values("T_2DMMT")
        col = COLORS[ci % len(COLORS)]
        ax.plot(grp["T_2DMMT"], grp["prob"] * 100,
                color=col, linewidth=1.5,
                label=EPC_LABEL_MAP.get(int(epc_lvl), str(epc_lvl)))

    ax.axvline(T2DMMT_CENTRE, color="gray", linewidth=0.7,
               linestyle=":", alpha=0.7, label=f"Observed mean ({T2DMMT_CENTRE:.1f}°C)")
    ax.set_xlabel(r"$T_{2DMMT}$ (°C)", fontsize=8)
    ax.set_ylabel(r"P(mean night $T_{in} \geq$ threshold) (%)", fontsize=7)
    ax.set_title(f"Threshold {thr}°C", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

fig.suptitle(
    f"P(mean night indoor temperature $\\geq$ threshold) vs $T_{{2DMMT}}$ by EPC band\n"
    f"{RLAB} bedrooms, May--Sep · GEE AR(1) · EFUS 2017",
    fontsize=8.5,
)
save(fig, "gee_prob_curves.svg")

print("\nAll figures saved to:", OUTDIR)
