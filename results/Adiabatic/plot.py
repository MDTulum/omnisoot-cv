import re
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# User settings
# -----------------------------
DATA_DIR = Path(r"C:\Users\MehranDadsetan\OneDrive - Tulum Energy S.A\Documents\Github_Codes\omnisoot-cv\results\Adiabatic")
FILE_GLOB = "sim_results_Tin=*C*.csv"

# Targets in Celsius
TARGETS_C = list(range(1000, 1751, 50))

# Export only selected columns (optional). None => export ALL columns
EXPORT_COLUMNS = None

# Output
OUT_DIR = DATA_DIR / "_postprocess_targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROFILES = OUT_DIR / "interpolated_profiles_at_targets"
OUT_PROFILES.mkdir(parents=True, exist_ok=True)

TIME_COL = "t[s]"
TEMP_COL = "T[K]"
Z_COL = "z[m]"  # optional

# -----------------------------
# Helpers
# -----------------------------
def parse_tin_c_from_filename(fname):
    m = re.search(r"Tin\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*C", fname)
    return float(m.group(1)) if m else None


def apply_export_filter(df):
    if EXPORT_COLUMNS is None:
        return df

    keep = [c for c in EXPORT_COLUMNS if c in df.columns]
    for c in (TIME_COL, TEMP_COL, Z_COL):
        if c in df.columns and c not in keep:
            keep.insert(0, c)

    # de-dupe preserve order
    seen = set()
    keep2 = []
    for c in keep:
        if c not in seen:
            keep2.append(c)
            seen.add(c)
    return df[keep2]


def _clean_df_for_interp(df):
    if TIME_COL not in df.columns or TEMP_COL not in df.columns:
        raise ValueError(f"CSV must contain '{TIME_COL}' and '{TEMP_COL}'.")

    work = df.copy()
    work[TIME_COL] = pd.to_numeric(work[TIME_COL], errors="coerce")
    work[TEMP_COL] = pd.to_numeric(work[TEMP_COL], errors="coerce")
    work = work.dropna(subset=[TIME_COL, TEMP_COL]).sort_values(TIME_COL).reset_index(drop=True)
    return work


def first_bracket_interpolate_all_columns(df, target_K):
    """
    EXACT target temperature via interpolation if it jumps across target.

    Rule:
    - If any row has T == target (within floating epsilon), return that exact row.
    - Else find the first consecutive pair that brackets target (jump over),
      and linearly interpolate EVERY numeric column at the same fraction.

    Returns:
      feasible (bool), out (dict)
    """
    work = _clean_df_for_interp(df)

    t = work[TIME_COL].to_numpy(dtype=float)
    T = work[TEMP_COL].to_numpy(dtype=float)

    if len(t) < 2:
        return False, {"reason": "insufficient_points"}

    y = T - target_K

    # Exact match (first)
    exact = np.where(np.isclose(y, 0.0, atol=1e-12))[0]
    if exact.size > 0:
        i = int(exact[0])
        out = work.iloc[i].to_dict()
        out[TEMP_COL] = float(target_K)
        out["_method"] = "exact"
        out["_bracket_i0"] = i
        out["_bracket_i1"] = i
        return True, out

    # Find first bracketing pair (jump over target)
    # Condition: y[i]*y[i+1] < 0  (strict sign change)
    # We also allow y[i]*y[i+1] == 0 if one endpoint hits target (handled above),
    # so strict is fine here.
    prod = y[:-1] * y[1:]
    idxs = np.where(prod < 0)[0]
    if idxs.size == 0:
        return False, {"reason": "no_bracketing_jump"}

    i0 = int(idxs[0])
    i1 = i0 + 1

    t0, t1 = t[i0], t[i1]
    T0, T1 = T[i0], T[i1]

    if np.isclose(T1, T0):
        return False, {"reason": "flat_segment"}

    frac = (target_K - T0) / (T1 - T0)
    t_star = t0 + frac * (t1 - t0)

    out = {
        TIME_COL: float(t_star),
        TEMP_COL: float(target_K),
        "_method": "linear",
        "_bracket_i0": i0,
        "_bracket_i1": i1,
        "_frac": float(frac),
    }

    # Interpolate ALL numeric columns
    for col in work.columns:
        if col in (TIME_COL, TEMP_COL):
            continue

        v0 = pd.to_numeric(work.loc[i0, col], errors="coerce")
        v1 = pd.to_numeric(work.loc[i1, col], errors="coerce")

        # Only interpolate numeric columns where both endpoints are numeric
        if pd.isna(v0) or pd.isna(v1):
            continue

        out[col] = float(v0 + frac * (v1 - v0))

    return True, out


# -----------------------------
# Main processing
# -----------------------------
files = sorted(DATA_DIR.glob(FILE_GLOB))
if not files:
    raise FileNotFoundError(f"No files found in {DATA_DIR} matching {FILE_GLOB}")

records_long = []

for fp in files:
    tin_c = parse_tin_c_from_filename(fp.name)
    df = pd.read_csv(fp)
    df = apply_export_filter(df)

    for tgt_c in TARGETS_C:
        tgt_k = float(tgt_c + 273.15)

        feasible, out = first_bracket_interpolate_all_columns(df, tgt_k)

        base = {
            "file": fp.name,
            "Tin[C]": tin_c,
            "target_Tout[C]": float(tgt_c),
            "target_Tout[K]": float(tgt_k),
            "feasible": bool(feasible),
        }

        if feasible:
            base["t_residence[s]"] = out.get(TIME_COL, np.nan)
            base["z_at_target[m]"] = out.get(Z_COL, np.nan) if Z_COL in out else np.nan

            # store interpolated values for all columns (exclude internal keys)
            for k, v in out.items():
                if k.startswith("_"):
                    continue
                base[k] = v

            snap = pd.DataFrame([base])
            snap_out = OUT_PROFILES / f"{fp.stem}_Tout={int(tgt_c)}C.csv"
            snap.to_csv(snap_out, index=False)
        else:
            base["t_residence[s]"] = np.nan
            base["z_at_target[m]"] = np.nan
            base["NA_reason"] = out.get("reason", "unknown")

        records_long.append(base)

# Long summary
df_long = pd.DataFrame(records_long)
df_long.to_csv(OUT_DIR / "summary_targets_long.csv", index=False)

# Wide summary (residence time + feasibility)
wide_time = (
    df_long.pivot_table(
        index=["Tin[C]"],
        columns=["target_Tout[C]"],
        values="t_residence[s]",
        aggfunc="first",
    )
    .sort_index(axis=1)
)
wide_time.columns = [f"t_residence[s]_Tout={int(c)}C" for c in wide_time.columns]
wide_time = wide_time.reset_index()

wide_feas = (
    df_long.pivot_table(
        index=["Tin[C]"],
        columns=["target_Tout[C]"],
        values="feasible",
        aggfunc="first",
    )
    .sort_index(axis=1)
)
wide_feas.columns = [f"feasible_Tout={int(c)}C" for c in wide_feas.columns]
wide_feas = wide_feas.reset_index()

df_wide = pd.merge(wide_time, wide_feas, on="Tin[C]", how="outer")
df_wide.to_csv(OUT_DIR / "summary_targets_wide.csv", index=False)

print(f"Done. Outputs written to: {OUT_DIR}")
print(f"Files processed: {len(files)}")
