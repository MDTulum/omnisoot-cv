# OmniSoot vs OpenSmoke — species evolution (vol%) with LOG-x
# Fixed y-axis ranges per your specification

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------
# PATHS / SETTINGS
# ----------------------------
ABF_DIR = Path(r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\10sec\Monodisperse\ReactiveDimerization\ABF")
OPENSMOKE_DIR = ABF_DIR.parent / "opensmoke"

OUT_DIR = ABF_DIR.parent / "_figures_OmniSoot_vs_OpenSmoke_like_example"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPS_C = [1000, 1500, 2000, 2500]
TMAX_S = 10.0
T_EPS = 1e-6  # avoid log(0)

SPECIES = ["CH4", "C2H6", "C2H4", "C2H2", "H2"]
CY_COL = "carbon_yield[-]"

FIGSIZE = (14, 8)
DPI = 250

# Fixed y-axis limits
YLIMS = {
    "CH4": (0, 100),
    "C2H2": (0, 25),
    "C2H4": (0, 10),
    "C2H6": (0, 1),
    "H2":  (0, 100),
    "CarbonYield": (0, 100),
}

# ----------------------------
# HELPERS
# ----------------------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t[s]" not in df.columns:
        raise KeyError(f"'t[s]' not found in {path.name}")
    df = df[df["t[s]"] <= TMAX_S].copy()
    df.loc[df["t[s]"] <= 0.0, "t[s]"] = T_EPS
    return df.sort_values("t[s]").reset_index(drop=True)

def set_log_time_axis(ax):
    ax.set_xscale("log")
    ax.set_xlabel("Time [s] (log scale)")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    ax.grid(True, which="both", alpha=0.25)

def to_volpct(series):
    return series * 100.0

def plot_solid_and_dashed_same_color(ax, x1, y1, x2, y2, label_solid, label_dashed):
    line1, = ax.plot(x1, y1, linestyle="-", label=label_solid)
    color = line1.get_color()
    ax.plot(x2, y2, linestyle="--", color=color, label=label_dashed)

# ----------------------------
# MAIN PLOTTING
# ----------------------------
for T in TEMPS_C:
    abf = load_csv(ABF_DIR / f"sim_results_T={T}C.csv")
    osd = load_csv(OPENSMOKE_DIR / f"sim_results_T={T}C.csv")

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE)
    axes = axes.flatten()

    # Species plots
    for i, sp in enumerate(SPECIES):
        ax = axes[i]
        set_log_time_axis(ax)
        ax.set_ylabel("Vol %")
        ax.set_title(sp)
        ax.set_ylim(*YLIMS[sp])

        plot_solid_and_dashed_same_color(
            ax,
            abf["t[s]"], to_volpct(abf[sp]),
            osd["t[s]"], to_volpct(osd[sp]),
            label_solid=f"{sp} (OmniSoot)",
            label_dashed=f"{sp} (OpenSmoke)",
        )
        ax.legend(frameon=False)

    # Carbon yield plot
    ax = axes[5]
    set_log_time_axis(ax)
    ax.set_ylabel("Carbon yield [%]")
    ax.set_title("Carbon yield")
    ax.set_ylim(*YLIMS["CarbonYield"])

    plot_solid_and_dashed_same_color(
        ax,
        abf["t[s]"], abf[CY_COL] * 100,
        osd["t[s]"], osd[CY_COL] * 100,
        label_solid="Carbon yield (OmniSoot)",
        label_dashed="Carbon yield (OpenSmoke)",
    )
    ax.legend(frameon=False)

    fig.suptitle(
        f"OmniSoot vs OpenSmoke — Species (vol%) — T = {T}°C  (0–{TMAX_S:g}s, log time)",
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / f"OmniSoot_vs_OpenSmoke_species_volpct_T={T}C.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out}")

print(f"\nDone. Figures saved in:\n{OUT_DIR}\n")
