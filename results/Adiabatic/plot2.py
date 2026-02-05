from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# SETTINGS
# =============================================================================
DATA_DIR = Path(r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Adiabatic")
FILE_GLOB = "sim_results_Tin=*C*.csv"

# Choose which fraction to plot for species:
#   "X" -> mole fraction (vol% ~ X*100)
#   "Y" -> mass fraction (wt% ~ Y*100)
FRACTION_KIND = "X"   # <-- change to "Y" if you want mass fraction

# Time window
T_MAX_S = 100.0
T_MIN_POS = 1e-9  # avoid log(0)

# Columns
TIME_COL = "t[s]"
TEMP_COL = "T[K]"
SPECIES = ["CH4", "C2H6", "C2H4", "C2H2", "H2"]
CARBON_COL = "carbon_yield[-]"

# Axis limits (in percent)
YLIMS = {
    "CH4": (0, 100),
    "C2H6": (0, 5),
    "C2H4": (0, 10),
    "C2H2": (0, 15),
    "H2": (0, 100),
    "Carbon yield": (0, 100),
}

# Output folders
OUT_SINGLE = DATA_DIR / "_plots_by_Tin"
OUT_OVERLAY = DATA_DIR / "_plots_all_cases_3x2"
OUT_SINGLE.mkdir(parents=True, exist_ok=True)
OUT_OVERLAY.mkdir(parents=True, exist_ok=True)

# Color map
BASE_CMAP = plt.get_cmap("autumn")

# =============================================================================
# HELPERS
# =============================================================================
def truncate_colormap(cmap, minval=0.25, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}",
        cmap(np.linspace(minval, maxval, n))
    )

CMAP = truncate_colormap(BASE_CMAP, 0.25, 1.0)


def parse_tin_c(fname: str) -> float | None:
    m = re.search(r"Tin\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*C", fname)
    return float(m.group(1)) if m else None


def safe_time_for_log(t):
    t = np.asarray(t, dtype=float)
    t = np.where(t <= 0, np.nan, t)
    t = np.nan_to_num(t, nan=T_MIN_POS)
    return np.maximum(t, T_MIN_POS)


def add_temperature_top_axis_single(ax, t, T_K, n_ticks=5):
    if len(t) < 2:
        return

    idx = np.unique(np.logspace(0, np.log10(len(t) - 1), n_ticks, dtype=int))
    idx = np.clip(idx, 0, len(t) - 1)

    t_ticks = t[idx]
    T_C_ticks = T_K[idx] - 273.15

    secax = ax.secondary_xaxis("top")
    secax.set_xscale("log")
    secax.set_xticks(t_ticks)
    secax.set_xticklabels([f"{Tc:.0f}" for Tc in T_C_ticks])
    secax.set_xlabel("Temperature [°C]")


def get_species_col(df: pd.DataFrame, sp: str, kind: str) -> str | None:
    """
    Return the appropriate column name for species sp:
      prefer '{sp}_{kind}' if present (new format),
      else fall back to '{sp}' (legacy).
    """
    cand = f"{sp}_{kind}"
    if cand in df.columns:
        return cand
    if sp in df.columns:
        return sp
    return None


def species_ylabel_and_scale(kind: str):
    if kind.upper() == "X":
        return "Vol %", 100.0
    if kind.upper() == "Y":
        return "Mass %", 100.0
    raise ValueError("FRACTION_KIND must be 'X' or 'Y'.")


# =============================================================================
# LOAD FILES
# =============================================================================
files = sorted(DATA_DIR.glob(FILE_GLOB))
if not files:
    raise FileNotFoundError("No simulation files found.")

y_label, y_scale = species_ylabel_and_scale(FRACTION_KIND)

# =============================================================================
# PART A — SINGLE-CASE FIGURES (3×2) WITH TOP TEMP AXIS
# =============================================================================
for fp in files:
    tin_c = parse_tin_c(fp.name)
    df = pd.read_csv(fp)

    # Required columns check (species columns can be _X/_Y or legacy)
    req_base = [TIME_COL, TEMP_COL, CARBON_COL]
    if any(c not in df.columns for c in req_base):
        continue
    if any(get_species_col(df, sp, FRACTION_KIND) is None for sp in SPECIES):
        continue

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TEMP_COL]).sort_values(TIME_COL)
    df = df[df[TIME_COL] <= T_MAX_S]

    t = safe_time_for_log(df[TIME_COL].to_numpy())
    T_K = df[TEMP_COL].to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(16, 8), dpi=150)
    axes = axes.ravel()

    fig.suptitle(
        f"Species ({y_label}) — Tin = {tin_c:.0f}°C (0–{T_MAX_S:.0f}s, log time) | using {FRACTION_KIND}",
        fontsize=14,
    )

    for i, sp in enumerate(SPECIES):
        ax = axes[i]
        col = get_species_col(df, sp, FRACTION_KIND)
        y = pd.to_numeric(df[col], errors="coerce").to_numpy() * y_scale
        ax.plot(t, y, label=f"{sp} ({col})")
        ax.set_title(sp)
        ax.set_xscale("log")
        ax.set_xlim(T_MIN_POS, T_MAX_S)
        ax.set_ylim(*YLIMS[sp])
        ax.set_xlabel("Time [s] (log scale)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        add_temperature_top_axis_single(ax, t, T_K)

    ax = axes[5]
    y_c = pd.to_numeric(df[CARBON_COL], errors="coerce").to_numpy() * 100.0
    ax.plot(t, y_c, label="Carbon yield")
    ax.set_title("Carbon yield")
    ax.set_xscale("log")
    ax.set_xlim(T_MIN_POS, T_MAX_S)
    ax.set_ylim(*YLIMS["Carbon yield"])
    ax.set_xlabel("Time [s] (log scale)")
    ax.set_ylabel("Carbon yield [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    add_temperature_top_axis_single(ax, t, T_K)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_SINGLE / f"{fp.stem}_species_3x2_{FRACTION_KIND}.png", bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# PART B — ALL CASES IN ONE 3×2 FIGURE (SINGLE SHARED LEGEND)
# =============================================================================
cases = []
for fp in files:
    tin_c = parse_tin_c(fp.name)
    if tin_c is None:
        continue

    df = pd.read_csv(fp)

    # Required columns check
    if TIME_COL not in df.columns or CARBON_COL not in df.columns:
        continue
    if any(get_species_col(df, sp, FRACTION_KIND) is None for sp in SPECIES):
        continue

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    df = df[df[TIME_COL] <= T_MAX_S]

    t = safe_time_for_log(df[TIME_COL].to_numpy())

    data = {}
    for sp in SPECIES:
        col = get_species_col(df, sp, FRACTION_KIND)
        data[sp] = pd.to_numeric(df[col], errors="coerce").to_numpy() * y_scale

    data["Carbon yield"] = pd.to_numeric(df[CARBON_COL], errors="coerce").to_numpy() * 100.0

    cases.append({"Tin": tin_c, "t": t, **data})

if not cases:
    raise RuntimeError("No valid cases found after filtering for required columns.")

cases = sorted(cases, key=lambda d: d["Tin"])
norm = mcolors.Normalize(vmin=min(c["Tin"] for c in cases), vmax=max(c["Tin"] for c in cases))

fig, axes = plt.subplots(3, 2, figsize=(16, 8), dpi=150)
axes = axes.ravel()

fig.suptitle(
    f"All inlet-temperature cases — Species ({y_label}) (0–{T_MAX_S:.0f}s, log time) | using {FRACTION_KIND}",
    fontsize=14
)

for i, sp in enumerate(SPECIES):
    ax = axes[i]
    for c in cases:
        # Label only on first subplot to avoid duplicates in legend
        lbl = f"Tin={c['Tin']:.0f}°C" if i == 0 else "_nolegend_"
        ax.plot(
            c["t"],
            c[sp],
            color=CMAP(norm(c["Tin"])),
            linewidth=2,
            label=lbl,
        )
    ax.set_title(sp)
    ax.set_xscale("log")
    ax.set_xlim(T_MIN_POS, T_MAX_S)
    ax.set_ylim(*YLIMS[sp])
    ax.set_xlabel("Time [s] (log scale)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

ax = axes[5]
for c in cases:
    ax.plot(
        c["t"],
        c["Carbon yield"],
        color=CMAP(norm(c["Tin"])),
        linewidth=2,
        label="_nolegend_",
    )

ax.set_title("Carbon yield")
ax.set_xscale("log")
ax.set_xlim(T_MIN_POS, T_MAX_S)
ax.set_ylim(*YLIMS["Carbon yield"])
ax.set_xlabel("Time [s] (log scale)")
ax.set_ylabel("Carbon yield [%]")
ax.grid(True, alpha=0.3)

# Single shared legend from the first subplot
handles, labels = axes[0].get_legend_handles_labels()
ncol = min(len(labels), 6) if labels else 1
fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.savefig(OUT_OVERLAY / f"ALLCASES_species_3x2_overlay_{FRACTION_KIND}.png", bbox_inches="tight")
plt.close(fig)

print("Done.")
