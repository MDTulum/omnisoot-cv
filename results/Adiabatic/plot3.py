from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
DATA_DIR = Path(r"C:\_\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Adiabatic")

# Cases / files
TIN_LIST_C = [1400, 1600, 1800, 2000]  # expects sim_results_Tin=1400C.csv, ...

# Column names (edit here if your CSV headers differ)
COL_TIME = "t[s]"
COL_Z    = "z[m]"
COL_TK   = "T[K]"

# Plot limits
T_MAX_S = 10.0            # seconds
T_Y_MIN = 1000.0          # °C
T_Y_MAX = 2000.0          # °C

# Output
OUT_FIG = DATA_DIR / "T_vs_time_logx_2x2_1000to2000C_up_to_10s.png"

# -----------------------------
# Helper
# -----------------------------
def load_case(tin_c: int) -> pd.DataFrame:
    fp = DATA_DIR / f"sim_results_Tin={tin_c}C.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    df = pd.read_csv(fp)

    missing = [c for c in [COL_TIME, COL_Z, COL_TK] if c not in df.columns]
    if missing:
        raise KeyError(
            f"{fp.name}: missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[COL_TIME, COL_Z, COL_TK]].dropna()

    # Log-x safe + time limit
    df = df[(df[COL_TIME] > 0) & (df[COL_TIME] <= T_MAX_S)]

    # Convert K → C
    df["T[C]"] = df[COL_TK] - 273.15

    df = df.sort_values(COL_TIME).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{fp.name}: no rows left after filtering.")

    return df

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
axes = axes.ravel()

for i, tin_c in enumerate(TIN_LIST_C):
    ax = axes[i]
    df = load_case(tin_c)

    ax.plot(df[COL_TIME], df["T[C]"], linewidth=2)
    ax.set_xscale("log")

    # x-limits: start close to zero
    t_min = float(df[COL_TIME].min())
    ax.set_xlim(t_min, T_MAX_S)

    # y-limits: identical for all panels
    ax.set_ylim(T_Y_MIN, T_Y_MAX)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title(f"Tin = {tin_c} °C")
    ax.set_xlabel("Residence time, t [s] (log scale)")
    ax.set_ylabel("Temperature, T [°C]")
    ax.tick_params(axis="y", labelleft=True)

fig.suptitle(
    "Temperature vs Residence Time (0 < t ≤ 10 s, log-x)\n"
    "Uniform Temperature Scale: 1000–2000 °C (Adiabatic)",
    y=0.98
)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"Saved figure to: {OUT_FIG}")
