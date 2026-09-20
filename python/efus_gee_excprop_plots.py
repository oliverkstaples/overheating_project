#!/usr/bin/env python
"""
Focused plots/tables for the exceedance-proportion GEE (gee_excprop_t), London
livingroom August, thresholds 26/27/28 C.

Outputs (plots/efus2017/london_livingroom_1month/impact_profiles/):
  gee_excprop_prob_curves.svg       expected % occupied hours > t vs 2DMMT, by EPC
  gee_excprop_forest_full.svg       all terms; significant (p<0.05) blue, rest grey
  gee_excprop_forest_significant.svg  only terms significant in >=1 threshold

Also prints LaTeX-ready odds-ratio tables for 26/27/28.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "nature", "bright"])

ANALYSIS = "analysis/livingroom/august"
OUT      = "plots/efus2017/london_livingroom_1month/impact_profiles"
os.makedirs(OUT, exist_ok=True)

THRESHOLDS = [26, 27, 28]
SIG        = 0.05
BLUE, GREY = "#2E67D0", "#9A9A9A"

coefs    = pd.read_csv(f"{ANALYSIS}/gee_coefs.csv")
centres  = pd.read_csv(f"{ANALYSIS}/climate_centres.csv")
T_CENTRE = float(centres.loc[centres.variable == "T_2DMMT", "centre"].iloc[0])

# term key -> y-axis label (each contrast names its reference level explicitly)
TERM_META = [
    ("T_2DMMT_c",      "2DMMT (per $^\\circ$C)"),
    ("EPC2",           "D vs C+"),
    ("EPC3",           "E vs C+"),
    ("EPC4",           "F/G vs C+"),
    ("dwtype1",        "Detached vs Flat"),
    ("dwtype2",        "Semi-det. vs Flat"),
    ("dwtype3",        "End-terrace vs Flat"),
    ("dwtype4",        "Mid-terrace vs Flat"),
    ("dwtype5",        "Bungalow vs Flat"),
    ("dwage1",         "Pre-1919 vs Post-1990"),
    ("dwage2",         "1919--44 vs Post-1990"),
    ("dwage3",         "1945--64 vs Post-1990"),
    ("dwage4",         "1965--74 vs Post-1990"),
    ("dwage5",         "1975--80 vs Post-1990"),
    ("dwage6",         "1981--90 vs Post-1990"),
    ("cooling1",       "Cooling vs None"),
]
LABELS = {k: l for k, l in TERM_META}

EPC_LABELS  = {1: "EPC C+", 2: "EPC D", 3: "EPC E", 4: "EPC F/G"}
# Match the original model_plots.py prob-curve scheme: default matplotlib cycle,
# EPC level e -> COLORS[4 - e]  (C+ -> [3], D -> [2], E -> [1], F/G -> [0]).
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

import matplotlib.colors as mcolors
def dark(c):                       # darker edge shade, matching the description plots
    return tuple(np.array(mcolors.to_rgb(c)) * 0.45)


def cdf(model):
    return coefs[coefs.model == model].set_index("term")


# ── LaTeX odds-ratio tables ───────────────────────────────────────────────────
def latex_table(thr):
    d = cdf(f"gee_excprop_{thr}")
    rows = []
    for k, lab in TERM_META:
        if k not in d.index:
            continue
        r = d.loc[k]
        b, lo, hi, p = r.estimate, r.ci_lo, r.ci_hi, r.p_san
        star = "$^{*}$" if p < 0.05 else ""
        rows.append(f"  {lab} & {np.exp(b):.2f} & ({np.exp(lo):.2f}, {np.exp(hi):.2f}) "
                    f"& {p:.3f}{star} \\\\")
    print(f"\n% ── Odds ratios, exceedance-proportion GEE, threshold {thr} C ──")
    print("\\begin{tabular}{l r r r}")
    print("\\toprule\nTerm & OR & 95\\% CI & $p$ \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}")


for t in THRESHOLDS:
    latex_table(t)


# ── Figure 1: probability (expected exceedance %) curves ──────────────────────
def predict_curve(d, epc, t2c):
    b0 = d.loc["(Intercept)", "estimate"]
    bc = d.loc["T_2DMMT_c", "estimate"]
    be = 0.0 if epc == 1 else d.loc[f"EPC{epc}", "estimate"]
    ik = f"T_2DMMT_c:EPC{epc}"
    bi = d.loc[ik, "estimate"] if ik in d.index else 0.0
    eta = b0 + (bc + bi) * t2c + be
    return 1.0 / (1.0 + np.exp(-eta))


T_real = np.linspace(15.0, 32.0, 200)
t2c    = T_real - T_CENTRE

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), constrained_layout=True, sharey=True)
for ax, thr in zip(axes, THRESHOLDS):
    d = cdf(f"gee_excprop_{thr}")
    for epc in (1, 2, 3, 4):
        if epc != 1 and f"EPC{epc}" not in d.index:
            continue
        ax.plot(T_real, 100 * predict_curve(d, epc, t2c),
                color=COLORS[(4 - epc) % len(COLORS)], linewidth=1.6, label=EPC_LABELS[epc])
    ax.set_title(f"Threshold {thr}$^\\circ$C", fontsize=9)
    ax.set_xlabel(r"Outdoor 2DMMT ($^\circ$C)", fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.4)
axes[0].set_ylabel("Expected occupied hours\nabove threshold (\\%)", fontsize=8)
# axes[2].legend(fontsize=7, frameon=True, loc="upper left")


# Shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=True,
    fontsize=6,
    title="EPC band",
    bbox_to_anchor=(0.5, -0.15)
)

# fig.suptitle("Predicted exceedance of occupied hours vs outdoor 2DMMT by EPC band "
#              "--- London livingroom, August\nExceedance-proportion GEE (AR(1), sandwich SE); "
#              "reference dwelling: Flat, no cooling", fontsize=9)
fig.savefig(f"{OUT}/gee_excprop_prob_curves.svg", bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {OUT}/gee_excprop_prob_curves.svg")


# ── Forest helper (odds-ratio scale) ──────────────────────────────────────────
def forest(terms, fname, title):
    ypos = list(range(len(terms) - 1, -1, -1))
    # fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.34 * len(terms) + 1.6),
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6),
                             sharey=True, constrained_layout=True)
    for ax, thr in zip(axes, THRESHOLDS):
        d = cdf(f"gee_excprop_{thr}")
        ax.axvline(1.0, color="black", linewidth=0.8)
        for yp, term in zip(ypos, terms):
            if term not in d.index:
                continue
            r = d.loc[term]
            orr, lo, hi, p = np.exp(r.estimate), np.exp(r.ci_lo), np.exp(r.ci_hi), r.p_san
            c = BLUE if p < SIG else GREY
            ax.plot([lo, hi], [yp, yp], color=c, linewidth=1.4, zorder=2)
            ax.plot(orr, yp, "o", color=c, markersize=4.5,
                    markeredgecolor=dark(c), markeredgewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"{thr}$^\\circ$C threshold", fontsize=9)
        ax.set_xlabel("Odds ratio (log scale)", fontsize=8)
        ax.grid(axis="x", alpha=0.3, linewidth=0.4)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([LABELS[t] for t in terms], fontsize=7)
    # Panel labels (A)/(B)/(C), matching the description plots
    # for k, ax in enumerate(axes):
    #     ax.text(-0.02, 1.04, f"({chr(65 + k)})", transform=ax.transAxes,
    #             fontsize=11, fontweight="bold", va="bottom", ha="right")
    import matplotlib.lines as mlines
    h = [mlines.Line2D([], [], color=BLUE, marker="o", label="$p<0.05$"),
         mlines.Line2D([], [], color=GREY, marker="o", label="$p\\geq0.05$")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.1), frameon=True)
    # fig.suptitle(title, fontsize=9)
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{fname}")


ALL_TERMS = [k for k, _ in TERM_META]
forest(ALL_TERMS, "gee_excprop_forest_full.svg",
       "Exceedance-proportion GEE: odds ratios (all terms) --- London livingroom, August\n"
       "Significant (blue) vs non-significant (grey); AR(1), sandwich SE")

# significant in >=1 threshold
sig_terms = [k for k in ALL_TERMS
             if any((cdf(f"gee_excprop_{t}").reindex([k]).p_san < SIG).fillna(False).iloc[0]
                    for t in THRESHOLDS)]
forest(sig_terms, "gee_excprop_forest_significant.svg",
       "Exceedance-proportion GEE: significant odds ratios --- London livingroom, August\n"
       "(terms with $p<0.05$ at $\\geq$1 threshold); AR(1), sandwich SE")
print("\nDone.")
