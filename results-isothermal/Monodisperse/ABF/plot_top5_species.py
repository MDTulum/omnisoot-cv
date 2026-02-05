import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.collections import LineCollection


# =========================
# User settings
# =========================
RESULTS_DIR = r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Monodisperse\ReactiveDimerization"
PATTERN = "sim_results_T=*C.csv"

# Species to plot (mole fractions in CSV → converted to vol %)
PLOT_SPECIES = ["CH4", "C2H6", "C2H4", "C2H2", "H2"]
INERT_SPECIES = ["N2", "CO2"]

OTHERS_COL = "Others"
CARBON_YIELD_COL = "carbon_yield[-]"

# Labels
OTHERS_LABEL = "Others [vol %]"
CARBON_YIELD_LABEL = "Carbon yield [%]"

# Log-axis safety
EPS = 1e-12

# =========================
# PPT-friendly styling
# =========================
FIG_W = 13.33   # ~16:9 slide width in inches
FIG_H = 7.5     # slide height in inches
LINE_W = 1.8
ALPHA = 0.95

TITLE_FS = 14
AX_TITLE_FS = 11
LABEL_FS = 9
TICK_FS = 8
CBAR_FS = 9

# Better y-limits per subplot (improves readability a lot)
YLIMS = {
    "CH4": (0, 100),
    "H2": (0, 100),
    "C2H6": (0, 5),
    "C2H4": (0, 10),
    "C2H2": (0, 30),
    "Others": (0, 20),
    CARBON_YIELD_COL: (0, 100),
}

CBAR_TICKS = [1000, 1250, 1500, 1750, 2000, 2250, 2500]


def _find_csv_files(folder):
    return sorted(glob.glob(os.path.join(folder, PATTERN)))


def _parse_temp_c(path):
    m = re.search(r"T=([0-9]+(?:\.[0-9]+)?)C", os.path.basename(path))
    return float(m.group(1)) if m else None


def _ensure_columns(df, cols, path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing} in file: {path}")


def _setup_log_x(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)


def _setup_linear_xy(ax):
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)


def load_runs():
    files = _find_csv_files(RESULTS_DIR)
    if not files:
        raise SystemExit(f"No files matching {PATTERN} found in: {RESULTS_DIR}")

    required = ["t[s]", "z[m]"] + PLOT_SPECIES + INERT_SPECIES + [CARBON_YIELD_COL]

    runs = []
    for path in files:
        T_c = _parse_temp_c(path)
        if T_c is None:
            raise SystemExit(f"Could not parse temperature from filename: {path}")

        df = pd.read_csv(path)
        _ensure_columns(df, required, path)

        # log-safe time and z
        df["t[s]"] = np.maximum(df["t[s]"].to_numpy(dtype=float), EPS)
        df["z[m]"] = np.maximum(df["z[m]"].to_numpy(dtype=float), EPS)

        # Compute Others from original mole fractions
        x_sum = np.zeros(len(df), dtype=float)
        for sp in PLOT_SPECIES + INERT_SPECIES:
            x_sum += df[sp].to_numpy(dtype=float)

        # Convert selected species + inerts to vol%
        for sp in PLOT_SPECIES + INERT_SPECIES:
            df[sp] = 100.0 * df[sp].to_numpy(dtype=float)

        # Others in vol%
        df[OTHERS_COL] = np.clip(100.0 * (1.0 - x_sum), 0.0, 100.0)

        # Carbon yield in %
        df[CARBON_YIELD_COL] = 100.0 * df[CARBON_YIELD_COL].to_numpy(dtype=float)

        runs.append((T_c, df))

    runs.sort(key=lambda x: x[0])
    return runs


def make_color_mapping(runs):
    temps = np.array([T for T, _ in runs], dtype=float)
    norm = plt.Normalize(vmin=float(temps.min()), vmax=float(temps.max()))
    cmap = plt.cm.plasma
    return temps, norm, cmap


def add_temperature_colorbar(fig, norm, cmap, label="Temperature [°C]", cax_rect=None):
    """Manually positioned colorbar so subplots don't get squeezed."""
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if cax_rect is None:
        cax_rect = [0.94, 0.15, 0.015, 0.70]

    cax = fig.add_axes(cax_rect)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label, fontsize=CBAR_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_ticks(CBAR_TICKS)
    return cbar


# -------------------------
# Gradient line helper
# -------------------------
def plot_gradient_line(ax, x, y, cmap, norm, linewidth=2.2, alpha=0.95, zorder=3):
    """
    Draw a line where each segment is colored by the temperature along x (T on x-axis).
    x should be Temperature in °C.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by x to ensure monotonic segments
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Build line segments
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color segments by midpoint temperature
    x_mid = 0.5 * (x[:-1] + x[1:])

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    lc.set_array(x_mid)
    ax.add_collection(lc)

    # Also add points
    ax.scatter(x, y, c=cmap(norm(x)), s=18, alpha=alpha, zorder=zorder + 1, edgecolors="none")
    ax.autoscale_view()
    return lc


# =========================
# 1) Existing time-profile figures
# =========================
def plot_species_grid(runs):
    grid_names = PLOT_SPECIES + [OTHERS_COL]
    grid_titles = [f"{sp} [vol %]" for sp in PLOT_SPECIES] + [OTHERS_LABEL]

    _, norm, cmap = make_color_mapping(runs)

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H), sharex=True)
    axes = axes.flatten()

    for ax, name, title in zip(axes, grid_names, grid_titles):
        for T_c, df in runs:
            ax.plot(
                df["t[s]"],
                df[name],
                color=cmap(norm(T_c)),
                linewidth=LINE_W,
                alpha=ALPHA,
            )
        ax.set_title(title, fontsize=AX_TITLE_FS, pad=6)
        ax.set_ylim(*YLIMS.get(name, (0, 100)))
        _setup_log_x(ax)

    for ax in axes[3:]:
        ax.set_xlabel("Time [s] (log)", fontsize=LABEL_FS)

    fig.suptitle("Constant Temperature Performance — Species (vol %)", fontsize=TITLE_FS, y=0.96)

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.94, 0.16, 0.015, 0.68])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10, wspace=0.22, hspace=0.28)

    out_path = os.path.join(RESULTS_DIR, "constant_temperature_species_grid_slide.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def plot_carbon_yield_time(runs):
    _, norm, cmap = make_color_mapping(runs)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, 4.2))

    for T_c, df in runs:
        ax.plot(
            df["t[s]"],
            df[CARBON_YIELD_COL],
            color=cmap(norm(T_c)),
            linewidth=2.2,
            alpha=ALPHA,
        )

    ax.set_title("Constant Temperature Performance — Carbon Yield [%]", fontsize=TITLE_FS, pad=10)
    ax.set_ylabel(CARBON_YIELD_LABEL, fontsize=LABEL_FS)
    ax.set_xlabel("Time [s] (log)", fontsize=LABEL_FS)
    ax.set_ylim(0, 100)
    _setup_log_x(ax)

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.93, 0.18, 0.02, 0.68])
    fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.16)

    out_path = os.path.join(RESULTS_DIR, "constant_temperature_carbon_yield_slide.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


# =========================
# 2) Summary vs Temperature (gradient-colored)
# =========================
def _summarize_runs_vs_temperature(runs, columns):
    Ts = np.array([T for T, _ in runs], dtype=float)
    max_dict = {c: np.zeros(len(runs), dtype=float) for c in columns}
    out_dict = {c: np.zeros(len(runs), dtype=float) for c in columns}

    for i, (_, df) in enumerate(runs):
        for c in columns:
            arr = df[c].to_numpy(dtype=float)
            max_dict[c][i] = np.nanmax(arr)
            out_dict[c][i] = float(df[c].iloc[-1])  # end value

    return Ts, max_dict, out_dict


def plot_species_vs_temperature(runs, mode="max"):
    grid_names = PLOT_SPECIES + [OTHERS_COL]
    grid_titles = [f"{sp} [vol %]" for sp in PLOT_SPECIES] + [OTHERS_LABEL]

    Ts, max_dict, out_dict = _summarize_runs_vs_temperature(runs, grid_names)
    ydict = max_dict if mode.lower() == "max" else out_dict

    # Color mapping based on temperature (x-axis)
    temps, norm, cmap = make_color_mapping(runs)

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H))
    axes = axes.flatten()

    for ax, name, title in zip(axes, grid_names, grid_titles):
        y = ydict[name]
        plot_gradient_line(ax, Ts, y, cmap=cmap, norm=norm, linewidth=2.2, alpha=0.95)
        ax.set_title(title, fontsize=AX_TITLE_FS, pad=6)
        ax.set_xlabel("Temperature [°C]", fontsize=LABEL_FS)
        ax.set_ylim(*YLIMS.get(name, (0, 100)))
        _setup_linear_xy(ax)

    mode_txt = "Max" if mode.lower() == "max" else "Outlet"
    fig.suptitle(f"Constant Temperature Performance — Species {mode_txt} vs Temperature", fontsize=TITLE_FS, y=0.96)

    # Colorbar (matches the gradient)
    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.94, 0.16, 0.015, 0.68])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10, wspace=0.22, hspace=0.28)

    out_name = f"species_{mode.lower()}_vs_temperature.png"
    out_path = os.path.join(RESULTS_DIR, out_name)
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def plot_carbon_yield_vs_temperature(runs, mode="max"):
    Ts, max_dict, out_dict = _summarize_runs_vs_temperature(runs, [CARBON_YIELD_COL])
    y = max_dict[CARBON_YIELD_COL] if mode.lower() == "max" else out_dict[CARBON_YIELD_COL]

    _, norm, cmap = make_color_mapping(runs)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, 4.2))
    plot_gradient_line(ax, Ts, y, cmap=cmap, norm=norm, linewidth=2.6, alpha=0.95)

    mode_txt = "Max" if mode.lower() == "max" else "Outlet"
    ax.set_title(f"Constant Temperature Performance — Carbon Yield {mode_txt} vs Temperature", fontsize=TITLE_FS, pad=10)
    ax.set_xlabel("Temperature [°C]", fontsize=LABEL_FS)
    ax.set_ylabel("Carbon yield [%]", fontsize=LABEL_FS)
    ax.set_ylim(0, 100)
    _setup_linear_xy(ax)

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.93, 0.18, 0.02, 0.68])
    fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.16)

    out_name = f"carbon_yield_{mode.lower()}_vs_temperature.png"
    out_path = os.path.join(RESULTS_DIR, out_name)
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def main():
    runs = load_runs()

    # Time-profile plots
    plot_species_grid(runs)
    plot_carbon_yield_time(runs)

    # Temperature-summary plots (gradient colored + colorbar)
    plot_species_vs_temperature(runs, mode="max")
    plot_species_vs_temperature(runs, mode="outlet")
    plot_carbon_yield_vs_temperature(runs, mode="max")
    plot_carbon_yield_vs_temperature(runs, mode="outlet")


if __name__ == "__main__":
    main()
