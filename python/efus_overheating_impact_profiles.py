"""
Impact profiles: how building characteristics affect overheating risk.

Violin style matches efus_mm_london_1month.ipynb exactly:
  plt.style.use(['science', 'nature', 'bright'])
  colors from prop_cycle; dark_color = color * 0.45
  showextrema=True; cmins/cmaxes hidden; cbars linewidth 1.2
  IQR box (i-0.06, q1) width 0.12; median white line i±0.049

Violin data: daily_overheating.parquet (one row per dwelling × day)
OLS data:    dwelling_overheating_summary.parquet (one row per dwelling)

Outputs → plots/efus2017/{region}_{room}_{1month|4month}/impact_profiles/
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
import scienceplots                        # noqa: F401
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ROOM_CONFIG = {
    "livingroom": {"label": "Living room", "occ_label": "07:00--21:00", "crit_pct": 3},
    "bedroom":    {"label": "Bedroom",      "occ_label": "22:00--06:59", "crit_pct": 1},
}

parser = argparse.ArgumentParser()
parser.add_argument("--subdir", type=str, default="")
parser.add_argument("--room",   type=str, default="livingroom",
                    choices=list(ROOM_CONFIG.keys()))
parser.add_argument("--region", type=str, default="london",
                    choices=["london", "southwest"])
args = parser.parse_args()

ROOM       = args.room
RCFG       = ROOM_CONFIG[ROOM]
ROOM_LABEL = RCFG["label"]
CRIT_PCT   = RCFG["crit_pct"]
REGION     = args.region
REGION_LABEL = {"london": "London", "southwest": "South West"}[REGION]

_reg_pre   = "" if REGION == "london" else REGION
_room_sub  = f"{ROOM}/{args.subdir}" if args.subdir else ROOM
_ana_sub   = "/".join(filter(None, [_reg_pre, _room_sub]))
ANALYSIS_DIR = os.path.join(ROOT, "analysis", _ana_sub)

_month_label = "1month" if args.subdir == "august" else "4month"
OUTDIR = os.path.join(ROOT, "plots", "efus2017",
                      f"{REGION}_{ROOM}_{_month_label}", "impact_profiles")
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use(["science", "nature", "bright"])

# ── Data ──────────────────────────────────────────────────────────────────────
daily   = pd.read_parquet(os.path.join(ANALYSIS_DIR, "daily_overheating.parquet"))
summary = pd.read_parquet(os.path.join(ANALYSIS_DIR, "dwelling_overheating_summary.parquet"))

_months = sorted(pd.to_datetime(daily["date"]).dt.month.unique())
_MNAMES = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
SEASON_LABEL = (_MNAMES.get(_months[0], str(_months[0])) if len(_months) == 1
                else f"{_MNAMES.get(_months[0], str(_months[0]))}--{_MNAMES.get(_months[-1], str(_months[-1]))}")

# ── Group definitions ─────────────────────────────────────────────────────────
# EPC: worst → best (matches notebook display_bands = [4,3,2,1])
EPC_ORDER  = [4, 3, 2, 1]
EPC_LABELS = {1: "C+\n", 2: "D\n",
              3: "E\n",  4: "F/G\n"}

# Dwelling type: natural order
DW_ORDER  = [1, 2, 3, 4, 5, 6]
DW_LABELS = {1: "Detached", 2: "Semi-det.", 3: "End-terr.",
             4: "Mid-terr.", 5: "Bungalow",  6: "Flat"}

# Dwelling age: oldest → newest (labels match model_plots.py / gee_excprop_plots.py)
DWAGE_ORDER  = [1, 2, 3, 4, 5, 6, 7]
DWAGE_LABELS = {1: "Pre-1919", 2: "1919--44", 3: "1945--64", 4: "1965--74",
                5: "1975--80", 6: "1981--90", 7: "Post-1990"}

# Binary characteristics: (column, {0: label, 1: label})
BINARY_CHARS = [
    ("CavityWall",          {0: "Solid wall",    1: "Cavity wall"}),
    ("InsulatedWalls_efus", {0: "No insul.",      1: "Insulated"}),
    ("FullyDblGlz_efus",    {0: "Part. glazing",  1: "Full dbl. glaz."}),
    ("AnyCooling",          {0: "No cooling",    1: "Cooling"}),
]

# ── Outcome definitions ───────────────────────────────────────────────────────
# (column in daily dataset, y-label, unit string for median/IQR labels, y-lim)
# Outcome tuples: (dataset_key, column, ylabel, unit, ylim)
# "daily"   → daily_overheating.parquet
# "summary" → dwelling_overheating_summary.parquet (exceedance already in %)
TEMP_OUTCOMES = [
    ("daily",   "daily_max_Tin",  r"Daily max $T_{in}$ ($^\circ$C)", "°C", (19, 33)),
    ("daily",   "daily_min_Tin",  r"Daily min $T_{in}$ ($^\circ$C)", "°C", (16, 31)),
]
PCT_OUTCOMES = [
    ("summary", "exceed_pct_26",  f"{SEASON_LABEL}" + r" occ. hours $\geq$26°C (%)", "%", (0, 45)),
    ("summary", "exceed_pct_27",  f"{SEASON_LABEL}" + r" occ. hours $\geq$27°C (%)", "%", (0, 30)),
]
ALL_OUTCOMES = TEMP_OUTCOMES + PCT_OUTCOMES
DS = {"daily": daily, "summary": summary}


# ── Violin helper (exact notebook style) ─────────────────────────────────────
def draw_violin(ax, data_list, xlabels, ylabel, ylim=None,
                title=None, is_pct=False):
    """
    data_list : list of 1-D arrays, one per group.
    Matches notebook style from efus_mm_london_1month.ipynb exactly.
    """
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n = len(data_list)
    positions = list(range(1, n + 1))

    # Note: pct data must already be in 0–100 scale (not 0–1) before calling

    # Drop groups with fewer than 2 observations (e.g. absent EPC bands)
    valid   = [(lab, arr) for lab, arr in zip(xlabels, data_list) if len(arr) >= 2]
    if not valid:
        return
    xlabels_filt, data_list = zip(*valid)
    xlabels    = list(xlabels_filt)
    data_list  = list(data_list)
    n          = len(data_list)
    positions  = list(range(1, n + 1))

    parts = ax.violinplot(
        data_list,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=True,
    )

    dark_colors = []
    for i, pc in enumerate(parts["bodies"]):
        color      = colors_cycle[i % len(colors_cycle)]
        dark_color = tuple(np.array(mcolors.to_rgb(color)) * 0.45)
        dark_colors.append(dark_color)
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.45)

    parts["cmins"].set_visible(False)
    parts["cmaxes"].set_visible(False)
    parts["cbars"].set_linewidth(1.2)
    parts["cbars"].set_color(dark_colors)

    for i, data in enumerate(data_list, start=1):
        color      = colors_cycle[i - 1]
        dark_color = dark_colors[i - 1]

        if len(data) < 2:
            continue

        q1, med, q3 = np.percentile(data, [25, 50, 75])

        # IQR box
        ax.add_patch(plt.Rectangle(
            (i - 0.06, q1), 0.12, q3 - q1,
            facecolor=color, edgecolor=dark_color,
            linewidth=1.2, zorder=4,
        ))

        # Median line
        ax.plot(
            [i - 0.049, i + 0.049], [med, med],
            color="white", linewidth=1.2,
            solid_capstyle="butt", zorder=5,
        )

        # Median label
        unit = "%" if is_pct else "°C"
        ax.text(
            i + 0.4, med, f"{med:.1f}{unit}",
            fontsize=5, va="center", ha="center", color=dark_color,
        )

        # IQR label
        y_offset = np.max(data) + 0.3

        ax.text(
            i,
            y_offset,
            f"IQR\n[{q1:.1f}, {q3:.1f}]",
            fontsize=5,
            va="bottom",
            ha="center",
            color=dark_color
        )

        # # IQR label — temperature: below violin / right (last: above / left)
        # #             percentage:  above violin spine (matches cells 19/21)
        # if is_pct:
        #     y_iqr = np.max(data) + 1.5
        #     ax.text(
        #         i, y_iqr,
        #         f"IQR\n[{q1:.1f}, {q3:.1f}]",
        #         fontsize=5, va="bottom", ha="center", color=dark_color,
        #     )
        # else:
        #     x_off = 0.4 if i < n else -0.4
        #     y_off = q1 - 0.5 if i < n else q3 + 0.5
        #     ax.text(
        #         i + x_off, y_off,
        #         f"IQR\n[{q1:.1f}, {q3:.1f}] °C",
        #         fontsize=5, va="top", ha="center", color=dark_color,
        #     )

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")

    # if ylim and ylim[0] is not None:
    #     ax.set_ylim(ylim)
    # elif ylim is None:
    #     ymin = min(np.min(d) for d in data_list if len(d) > 0) - 1
    #     ymax = max(np.max(d) for d in data_list if len(d) > 0) + 5
    #     ax.set_ylim(ymin, ymax)

    ymin = min(np.min(d) for d in data_list if len(d) > 0) - 1
    ymax = max(np.max(d) for d in data_list if len(d) > 0) + 5
    ax.set_ylim(ymin, ymax)


    if is_pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    if title:
        ax.set_title(title, fontsize=8)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: EPC band
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: EPC band impact...")

fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.6), constrained_layout=True)

xlabels_epc = [EPC_LABELS[b] for b in EPC_ORDER]

for ax, (ds_key, col, ylabel, unit, ylim) in zip(axes, ALL_OUTCOMES):
    is_pct = (unit == "%")
    df_use = DS[ds_key]
    groups = [df_use.loc[df_use["EPceeb12e_efus"] == b, col].dropna().values
              for b in EPC_ORDER]
    draw_violin(ax, groups, xlabels_epc, ylabel, ylim=ylim, is_pct=is_pct)

axes[0].set_title(r"Daily max $T_{in}$", fontsize=8)
axes[1].set_title(r"Daily min $T_{in}$", fontsize=8)
axes[2].set_title(f"Exceedance " + r"$\geq$26°C" + f" ({SEASON_LABEL})", fontsize=8)
axes[3].set_title(f"Exceedance " + r"$\geq$27°C" + f" ({SEASON_LABEL})", fontsize=8)

for ax in axes:
    ax.set_xlabel("EPC score band")

fig.suptitle(
    f"Impact of EPC band on overheating metrics --- {REGION_LABEL} {ROOM_LABEL.lower()}s, {SEASON_LABEL}\n"
    f"EFUS 2017 {REGION_LABEL} subset",
    fontsize=9,
)
save(fig, "impact_epc.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Dwelling type
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Dwelling type impact...")

fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.8), constrained_layout=True)

xlabels_dw = [DW_LABELS[d] for d in DW_ORDER]

for ax, (ds_key, col, ylabel, unit, ylim) in zip(axes, ALL_OUTCOMES):
    is_pct = (unit == "%")
    df_use = DS[ds_key]
    groups = [df_use.loc[df_use["dwtype_efus"] == d, col].dropna().values
              for d in DW_ORDER]
    draw_violin(ax, groups, xlabels_dw, ylabel, ylim=ylim, is_pct=is_pct)

axes[0].set_title(r"Daily max $T_{in}$", fontsize=8)
axes[1].set_title(r"Daily min $T_{in}$", fontsize=8)
axes[2].set_title(f"Exceedance " + r"$\geq$26°C" + f" ({SEASON_LABEL})", fontsize=8)
axes[3].set_title(f"Exceedance " + r"$\geq$27°C" + f" ({SEASON_LABEL})", fontsize=8)

for ax in axes:
    ax.set_xlabel("Dwelling type")

fig.suptitle(
    f"Impact of dwelling type on overheating metrics --- {REGION_LABEL} {ROOM_LABEL.lower()}s, {SEASON_LABEL}\n"
    f"EFUS 2017 {REGION_LABEL} subset",
    fontsize=9,
)
save(fig, "impact_dwtype.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2b: Dwelling age (build era)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2b: Dwelling age impact...")

fig, axes = plt.subplots(1, 4, figsize=(13.0, 2.8), constrained_layout=True)

xlabels_dwage = [DWAGE_LABELS[a] for a in DWAGE_ORDER]

for ax, (ds_key, col, ylabel, unit, ylim) in zip(axes, ALL_OUTCOMES):
    is_pct = (unit == "%")
    df_use = DS[ds_key]
    groups = [df_use.loc[df_use["dwage_efus"] == a, col].dropna().values
              for a in DWAGE_ORDER]
    draw_violin(ax, groups, xlabels_dwage, ylabel, ylim=ylim, is_pct=is_pct)

axes[0].set_title(r"Daily max $T_{in}$", fontsize=8)
axes[1].set_title(r"Daily min $T_{in}$", fontsize=8)
axes[2].set_title(f"Exceedance " + r"$\geq$26°C" + f" ({SEASON_LABEL})", fontsize=8)
axes[3].set_title(f"Exceedance " + r"$\geq$27°C" + f" ({SEASON_LABEL})", fontsize=8)

for ax in axes:
    ax.set_xlabel("Dwelling age (build era)")
    ax.tick_params(axis="x", labelsize=6)

fig.suptitle(
    f"Impact of dwelling age on overheating metrics --- {REGION_LABEL} {ROOM_LABEL.lower()}s, {SEASON_LABEL}\n"
    f"EFUS 2017 {REGION_LABEL} subset",
    fontsize=9,
)
save(fig, "impact_dwage.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Binary building characteristics
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Binary characteristics...")

BIN_OUTCOMES = [
    ("daily",   "daily_max_Tin",  r"Daily max $T_{in}$ ($^\circ$C)",                          "°C", (19, 33)),
    ("summary", "exceed_pct_26",  f"{SEASON_LABEL}" + r" occ. hours $\geq$26°C (%)",          "%",  (0, 45)),
    ("summary", "exceed_pct_27",  f"{SEASON_LABEL}" + r" occ. hours $\geq$27°C (%)",          "%",  (0, 30)),
]

fig, axes = plt.subplots(4, 3, figsize=(7.5, 9.5), constrained_layout=True)

for row, (char, labels_dict) in enumerate(BINARY_CHARS):
    codes    = [0, 1]
    xlabels  = [labels_dict[c] for c in codes]

    for col_idx, (ds_key, col, ylabel, unit, ylim) in enumerate(BIN_OUTCOMES):
        ax     = axes[row, col_idx]
        is_pct = (unit == "%")
        df_use = DS[ds_key]
        groups = [df_use.loc[df_use[char] == c, col].dropna().values for c in codes]

        draw_violin(ax, groups, xlabels, ylabel, ylim=ylim, is_pct=is_pct)

        # Mean difference annotation
        if all(len(g) > 0 for g in groups):
            diff     = np.mean(groups[1]) - np.mean(groups[0])
            unit_str = "%" if is_pct else "°C"
            ax.text(0.97, 0.97, f"$\\Delta$ = {diff:+.2f}{unit_str}",
                    transform=ax.transAxes, fontsize=6,
                    ha="right", va="top")

        if col_idx == 0:
            ax.set_ylabel(ylabel)
        if row == 0:
            titles = [r"Daily max $T_{in}$", r"Exceedance $\geq$26°C" + f" ({SEASON_LABEL})",
                      r"Exceedance $\geq$27°C" + f" ({SEASON_LABEL})"]
            ax.set_title(titles[col_idx], fontsize=8)

fig.suptitle(
    f"Impact of physical building characteristics on overheating metrics\n"
    f"{REGION_LABEL} {ROOM_LABEL.lower()}s, {SEASON_LABEL} · EFUS 2017",
    fontsize=9,
)
save(fig, "impact_binary.svg")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Overheating criterion rates
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: overheating criterion rates...")

CRIT_THRESHOLDS = [26, 27, 28]
colors_cycle    = plt.rcParams["axes.prop_cycle"].by_key()["color"]

RATE_CONFIGS = [
    ("EPceeb12e_efus", EPC_ORDER,  [EPC_LABELS[b] for b in EPC_ORDER],  "EPC band"),
    ("dwtype_efus",    DW_ORDER,   [DW_LABELS[d] for d in DW_ORDER],    "Dwelling type"),
    ("AnyCooling",     [0, 1],     ["No cooling", "Cooling"],            "Cooling"),
    ("CavityWall",     [0, 1],     ["Solid wall", "Cavity wall"],        "Wall construction"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), constrained_layout=True)

for ax, (char, codes, xlabels, char_title) in zip(axes.flat, RATE_CONFIGS):
    x     = np.arange(len(codes))
    n_t   = len(CRIT_THRESHOLDS)
    width = 0.22
    offs  = np.linspace(-(n_t - 1) / 2, (n_t - 1) / 2, n_t) * width

    for j, (thr, offset) in enumerate(zip(CRIT_THRESHOLDS, offs)):
        fail_col = f"season_overheat_{thr}"
        probs    = []
        counts   = []
        for code in codes:
            grp = summary[summary[char] == code][fail_col]
            probs.append(100 * grp.mean() if len(grp) > 0 else np.nan)
            counts.append((int(grp.sum()), len(grp)))

        col = colors_cycle[j % len(colors_cycle)]
        dk  = tuple(np.array(mcolors.to_rgb(col)) * 0.45)

        bars = ax.bar(x + offset, probs, width=width * 0.9,
                      color=col, alpha=0.6, edgecolor=dk,
                      linewidth=0.8, label=f"$\geq${thr}°C")

        for bar, p, (n_fail, n_tot) in zip(bars, probs, counts):
            if not np.isnan(p):
                ax.text(bar.get_x() + bar.get_width() / 2, p + 1.5,
                        f"{p:.0f}%", fontsize=5, ha="center",
                        va="bottom", color=dk)

    # Sample size labels below x-axis
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
    f"Overheating criterion rate by building characteristic --- {REGION_LABEL} {ROOM_LABEL.lower()}s, {SEASON_LABEL}\n"
    rf"Criterion: $>${CRIT_PCT}\% occupied hours above threshold  ·  EFUS 2017",
    fontsize=9,
)
save(fig, "impact_overheating_rate.svg")

print("\nAll figures saved to:", OUTDIR)
