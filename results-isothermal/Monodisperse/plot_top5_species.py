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
ABF_DIR = r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Monodisperse\ReactiveDimerization\ABF"
CRECK_DIR = r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Monodisperse\ReactiveDimerization\creck_soot_mechanism"
OPENSMOKE_DIR = r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Monodisperse\openSmoke"

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


def load_runs(results_dir, *, others_strategy="residual", needs_z=True):
    """
    others_strategy:
      - "residual": compute Others = 100*(1 - sum(PLOT_SPECIES+INERT_SPECIES))
      - "column":   use df["Others"] (and just convert to vol%)
    needs_z:
      - True: require z[m] column (Omnisoot folders)
      - False: don't require z[m] (OpenSMOKE folder)
    """
    files = sorted(glob.glob(os.path.join(results_dir, PATTERN)))
    if not files:
        raise SystemExit(f"No files matching {PATTERN} found in: {results_dir}")

    required = ["t[s]"] + PLOT_SPECIES + [CARBON_YIELD_COL]
    if needs_z:
        required += ["z[m]"]
        required += INERT_SPECIES  # only required when we do "residual" using inerts
    else:
        if others_strategy == "residual":
            raise SystemExit("OpenSMOKE loader must use others_strategy='column' (per your request).")

    runs = []
    for path in files:
        T_c = _parse_temp_c(path)
        if T_c is None:
            raise SystemExit(f"Could not parse temperature from filename: {path}")

        df = pd.read_csv(path)
        _ensure_columns(df, required, path)

        # log-x safety for time (and z when present)
        df["t[s]"] = np.maximum(df["t[s]"].to_numpy(dtype=float), EPS)
        if needs_z:
            df["z[m]"] = np.maximum(df["z[m]"].to_numpy(dtype=float), EPS)

        # Convert to vol % (assumes inputs are mole fractions)
        for sp in PLOT_SPECIES:
            df[sp] = 100.0 * df[sp].to_numpy(dtype=float)

        if needs_z:
            for sp in INERT_SPECIES:
                df[sp] = 100.0 * df[sp].to_numpy(dtype=float)

        # Others handling
        if others_strategy == "residual":
            x_sum_pct = np.zeros(len(df), dtype=float)
            for sp in PLOT_SPECIES + INERT_SPECIES:
                x_sum_pct += df[sp].to_numpy(dtype=float)
            df[OTHERS_COL] = np.clip(100.0 - x_sum_pct, 0.0, 100.0)

        elif others_strategy == "column":
            if OTHERS_COL not in df.columns:
                raise SystemExit(f"Expected '{OTHERS_COL}' column in: {path}")
            df[OTHERS_COL] = 100.0 * df[OTHERS_COL].to_numpy(dtype=float)
        else:
            raise ValueError("others_strategy must be 'residual' or 'column'")

        # Carbon yield to %
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
    Draw a line where each segment is colored by temperature along x.
    x should be Temperature in °C.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
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

    ax.scatter(x, y, c=cmap(norm(x)), s=18, alpha=alpha, zorder=zorder + 1, edgecolors="none")
    ax.autoscale_view()
    return lc


# =========================
# Time-limited summary helpers (UP TO 1s)
# =========================
def _value_at_time(df, col, t_target=1.0, tcol="t[s]", method="interp"):
    """
    Return df[col] at t_target.
    method:
      - "interp": linear interpolation (recommended)
      - "last":   last available sample at or before t_target (no interpolation)
    """
    t = df[tcol].to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    if method == "last":
        mask = t <= t_target
        if not np.any(mask):
            return float(y[0])
        return float(y[np.where(mask)[0][-1]])

    if t_target <= t[0]:
        return float(y[0])
    if t_target >= t[-1]:
        return float(y[-1])

    return float(np.interp(t_target, t, y))


def _summarize_runs_vs_temperature_upto_time(
    runs,
    columns,
    *,
    t_cut=1.0,
    tcol="t[s]",
    outlet_method="interp",
):
    """
    For each temperature run:
      - max_dict[c][i] = max of c over t <= t_cut
      - out_dict[c][i] = value of c at t = t_cut (interp or last<=t_cut)
    """
    Ts = np.array([T for T, _ in runs], dtype=float)
    max_dict = {c: np.zeros(len(runs), dtype=float) for c in columns}
    out_dict = {c: np.zeros(len(runs), dtype=float) for c in columns}

    for i, (_, df) in enumerate(runs):
        t = df[tcol].to_numpy(dtype=float)
        mask = t <= t_cut

        if not np.any(mask):
            mask = np.zeros_like(t, dtype=bool)
            mask[0] = True

        df_cut = df.loc[mask]

        for c in columns:
            arr = df_cut[c].to_numpy(dtype=float)
            max_dict[c][i] = np.nanmax(arr)
            out_dict[c][i] = _value_at_time(df, c, t_target=t_cut, tcol=tcol, method=outlet_method)

    return Ts, max_dict, out_dict


# =========================
# 1) Existing time-profile figures
# =========================
def plot_species_grid(runs, out_dir, xcol="t[s]", x_label="Time [s] (log)"):
    grid_names = PLOT_SPECIES + [OTHERS_COL]
    grid_titles = [f"{sp} [vol %]" for sp in PLOT_SPECIES] + [OTHERS_LABEL]

    _, norm, cmap = make_color_mapping(runs)

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H), sharex=True)
    axes = axes.flatten()

    for ax, name, title in zip(axes, grid_names, grid_titles):
        for T_c, df in runs:
            ax.plot(df[xcol], df[name], color=cmap(norm(T_c)), linewidth=LINE_W, alpha=ALPHA)
        ax.set_title(title, fontsize=AX_TITLE_FS, pad=6)
        ax.set_ylim(*YLIMS.get(name, (0, 100)))
        _setup_log_x(ax)

    for ax in axes[3:]:
        ax.set_xlabel(x_label, fontsize=LABEL_FS)

    fig.suptitle("Constant Temperature Performance — Species (vol %)", fontsize=TITLE_FS, y=0.96)

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.94, 0.16, 0.015, 0.68])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10, wspace=0.22, hspace=0.28)

    out_path = os.path.join(out_dir, f"constant_temperature_species_grid_{xcol}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def plot_carbon_yield_profile(runs, out_dir, xcol="t[s]", x_label="Time [s] (log)"):
    _, norm, cmap = make_color_mapping(runs)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, 4.2))

    for T_c, df in runs:
        ax.plot(df[xcol], df[CARBON_YIELD_COL], color=cmap(norm(T_c)), linewidth=2.2, alpha=ALPHA)

    ax.set_title("Constant Temperature Performance — Carbon Yield [%]", fontsize=TITLE_FS, pad=10)
    ax.set_ylabel(CARBON_YIELD_LABEL, fontsize=LABEL_FS)
    ax.set_xlabel(x_label, fontsize=LABEL_FS)
    ax.set_ylim(0, 100)
    _setup_log_x(ax)

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.93, 0.18, 0.02, 0.68])
    fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.16)

    out_path = os.path.join(out_dir, f"constant_temperature_carbon_yield_{xcol}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


# =========================
# 2) Summary vs Temperature (gradient-colored), using t<=1s + outlet@1s
# =========================
def plot_species_vs_temperature_compare_3(
    runs_a, runs_b, runs_c,
    label_a="ABF", label_b="CRECK", label_c="openSmoke_Creck",
    mode="max", out_dir=".",
    t_cut=1.0,
    outlet_method="interp",
):
    grid_names = PLOT_SPECIES + [OTHERS_COL]
    grid_titles = [f"{sp} [vol %]" for sp in PLOT_SPECIES] + [OTHERS_LABEL]

    Ts_a, max_a, out_a = _summarize_runs_vs_temperature_upto_time(
        runs_a, grid_names, t_cut=t_cut, outlet_method=outlet_method
    )
    Ts_b, max_b, out_b = _summarize_runs_vs_temperature_upto_time(
        runs_b, grid_names, t_cut=t_cut, outlet_method=outlet_method
    )
    Ts_c, max_c, out_c = _summarize_runs_vs_temperature_upto_time(
        runs_c, grid_names, t_cut=t_cut, outlet_method=outlet_method
    )

    y_a = max_a if mode.lower() == "max" else out_a
    y_b = max_b if mode.lower() == "max" else out_b
    y_c = max_c if mode.lower() == "max" else out_c

    temps_all = np.array(
        [T for T, _ in runs_a] + [T for T, _ in runs_b] + [T for T, _ in runs_c],
        dtype=float
    )
    norm = plt.Normalize(vmin=float(temps_all.min()), vmax=float(temps_all.max()))
    cmap = plt.cm.plasma

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H))
    axes = axes.flatten()

    for ax, name, title in zip(axes, grid_names, grid_titles):
        # ABF: solid
        lc1 = plot_gradient_line(ax, Ts_a, y_a[name], cmap=cmap, norm=norm, linewidth=2.2, alpha=0.95)
        # CRECK (Omnisoot): dashed
        lc2 = plot_gradient_line(ax, Ts_b, y_b[name], cmap=cmap, norm=norm, linewidth=2.2, alpha=0.95)
        lc2.set_linestyle("--")
        # openSmoke_Creck: dotted
        lc3 = plot_gradient_line(ax, Ts_c, y_c[name], cmap=cmap, norm=norm, linewidth=2.2, alpha=0.95)
        lc3.set_linestyle(":")

        ax.set_title(title, fontsize=AX_TITLE_FS, pad=6)
        ax.set_xlabel("Temperature [°C]", fontsize=LABEL_FS)
        ax.set_ylim(*YLIMS.get(name, (0, 100)))
        _setup_linear_xy(ax)

        ax.plot([], [], "k-", label=label_a, linewidth=2.2)
        ax.plot([], [], "k--", label=label_b, linewidth=2.2)
        ax.plot([], [], "k:", label=label_c, linewidth=2.2)
        ax.legend(fontsize=8, loc="best", frameon=True)

    mode_txt = "Max (t ≤ 1s)" if mode.lower() == "max" else f"Outlet @ {t_cut:g}s"
    fig.suptitle(
        f"Species {mode_txt} vs Temperature — {label_a} vs {label_b} vs {label_c}",
        fontsize=TITLE_FS, y=0.96
    )

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.94, 0.16, 0.015, 0.68])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10, wspace=0.22, hspace=0.28)

    out_path = os.path.join(
        out_dir,
        f"species_{mode.lower()}_vs_temperature_upto_{t_cut:g}s_{label_a}_vs_{label_b}_vs_{label_c}.png"
    )
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def plot_carbon_yield_vs_temperature_compare_3(
    runs_a, runs_b, runs_c,
    label_a="ABF", label_b="CRECK", label_c="openSmoke_Creck",
    mode="max", out_dir=".",
    t_cut=1.0,
    outlet_method="interp",
):
    Ts_a, max_a, out_a = _summarize_runs_vs_temperature_upto_time(
        runs_a, [CARBON_YIELD_COL], t_cut=t_cut, outlet_method=outlet_method
    )
    Ts_b, max_b, out_b = _summarize_runs_vs_temperature_upto_time(
        runs_b, [CARBON_YIELD_COL], t_cut=t_cut, outlet_method=outlet_method
    )
    Ts_c, max_c, out_c = _summarize_runs_vs_temperature_upto_time(
        runs_c, [CARBON_YIELD_COL], t_cut=t_cut, outlet_method=outlet_method
    )

    y_a = max_a[CARBON_YIELD_COL] if mode.lower() == "max" else out_a[CARBON_YIELD_COL]
    y_b = max_b[CARBON_YIELD_COL] if mode.lower() == "max" else out_b[CARBON_YIELD_COL]
    y_c = max_c[CARBON_YIELD_COL] if mode.lower() == "max" else out_c[CARBON_YIELD_COL]

    temps_all = np.array(
        [T for T, _ in runs_a] + [T for T, _ in runs_b] + [T for T, _ in runs_c],
        dtype=float
    )
    norm = plt.Normalize(vmin=float(temps_all.min()), vmax=float(temps_all.max()))
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, 4.2))

    lc1 = plot_gradient_line(ax, Ts_a, y_a, cmap=cmap, norm=norm, linewidth=2.6, alpha=0.95)
    lc2 = plot_gradient_line(ax, Ts_b, y_b, cmap=cmap, norm=norm, linewidth=2.6, alpha=0.95)
    lc2.set_linestyle("--")
    lc3 = plot_gradient_line(ax, Ts_c, y_c, cmap=cmap, norm=norm, linewidth=2.6, alpha=0.95)
    lc3.set_linestyle(":")

    mode_txt = "Max (t ≤ 1s)" if mode.lower() == "max" else f"Outlet @ {t_cut:g}s"
    ax.set_title(
        f"Carbon Yield {mode_txt} vs Temperature — {label_a} vs {label_b} vs {label_c}",
        fontsize=TITLE_FS, pad=10
    )
    ax.set_xlabel("Temperature [°C]", fontsize=LABEL_FS)
    ax.set_ylabel("Carbon yield [%]", fontsize=LABEL_FS)
    ax.set_ylim(0, 100)
    _setup_linear_xy(ax)

    ax.plot([], [], "k-", label=label_a, linewidth=2.6)
    ax.plot([], [], "k--", label=label_b, linewidth=2.6)
    ax.plot([], [], "k:", label=label_c, linewidth=2.6)
    ax.legend(fontsize=8, loc="best", frameon=True)

    add_temperature_colorbar(fig, norm=norm, cmap=cmap, cax_rect=[0.93, 0.18, 0.02, 0.68])
    fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.16)

    out_path = os.path.join(
        out_dir,
        f"carbon_yield_{mode.lower()}_vs_temperature_upto_{t_cut:g}s_{label_a}_vs_{label_b}_vs_{label_c}.png"
    )
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


def main():
    # Omnisoot folders (have z[m], and we compute Others as residual)
    runs_abf = load_runs(ABF_DIR, others_strategy="residual", needs_z=True)
    runs_creck = load_runs(CRECK_DIR, others_strategy="residual", needs_z=True)

    # OpenSMOKE folder (NO z[m], and use provided "Others" column)
    runs_opensmoke = load_runs(OPENSMOKE_DIR, others_strategy="column", needs_z=False)

    # ---- Profile plots (separate for each mechanism; OpenSMOKE gets only time profiles) ----
    for label, runs, out_dir, has_z in [
        ("ABF", runs_abf, ABF_DIR, True),
        ("CRECK", runs_creck, CRECK_DIR, True),
        ("openSmoke_Creck", runs_opensmoke, OPENSMOKE_DIR, False),
    ]:
        # time profiles
        plot_species_grid(runs, out_dir=out_dir, xcol="t[s]", x_label="Time [s] (log)")
        plot_carbon_yield_profile(runs, out_dir=out_dir, xcol="t[s]", x_label="Time [s] (log)")

        # length profiles only if z exists
        if has_z:
            plot_species_grid(runs, out_dir=out_dir, xcol="z[m]", x_label="Axial position z [m] (log)")
            plot_carbon_yield_profile(runs, out_dir=out_dir, xcol="z[m]", x_label="Axial position z [m] (log)")

    # ---- Summary vs temperature (compare on same graph), using t_cut=1s ----
    OUT_COMPARE = os.path.join(os.path.dirname(ABF_DIR), "ABF_vs_CRECK_vs_openSmoke_Creck")
    os.makedirs(OUT_COMPARE, exist_ok=True)

    t_cut = 1.0
    outlet_method = "interp"  # or "last"

    plot_species_vs_temperature_compare_3(
        runs_abf, runs_creck, runs_opensmoke,
        label_a="ABF", label_b="CRECK", label_c="openSmoke_Creck",
        mode="max", out_dir=OUT_COMPARE,
        t_cut=t_cut, outlet_method=outlet_method
    )
    plot_species_vs_temperature_compare_3(
        runs_abf, runs_creck, runs_opensmoke,
        label_a="ABF", label_b="CRECK", label_c="openSmoke_Creck",
        mode="outlet", out_dir=OUT_COMPARE,
        t_cut=t_cut, outlet_method=outlet_method
    )

    plot_carbon_yield_vs_temperature_compare_3(
        runs_abf, runs_creck, runs_opensmoke,
        label_a="ABF", label_b="CRECK", label_c="openSmoke_Creck",
        mode="max", out_dir=OUT_COMPARE,
        t_cut=t_cut, outlet_method=outlet_method
    )
    plot_carbon_yield_vs_temperature_compare_3(
        runs_abf, runs_creck, runs_opensmoke,
        label_a="ABF", label_b="CRECK", label_c="openSmoke_Creck",
        mode="outlet", out_dir=OUT_COMPARE,
        t_cut=t_cut, outlet_method=outlet_method
    )


if __name__ == "__main__":
    main()
