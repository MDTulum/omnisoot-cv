import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# FILES (adjust path if needed)
# -------------------------------
files = {
    1400: "C:\\_\\OneDrive - Tulum Energy S.A\\Documents\\Github_Codes\\omnisoot-cv\\results\\Adiabatic\\sim_results_Tin=1400C.csv",
    1600: "C:\\_\\OneDrive - Tulum Energy S.A\\Documents\\Github_Codes\\omnisoot-cv\\results\\Adiabatic\\sim_results_Tin=1600C.csv",
    1800: "C:\\_\\OneDrive - Tulum Energy S.A\\Documents\\Github_Codes\\omnisoot-cv\\results\\Adiabatic\\sim_results_Tin=1800C.csv",
    2000: "C:\\_\\OneDrive - Tulum Energy S.A\\Documents\\Github_Codes\\omnisoot-cv\\results\\Adiabatic\\sim_results_Tin=2000C.csv",
}

# -------------------------------
# TIME WINDOWS & FLOW FRACTIONS
# -------------------------------
time_bins = [
    (3.4, 5.2, 9/53),
    (5.2, 6.9, 14/53),
    (6.9, 8.7, 5/53),
    (8.7, 10.0, 25/53),
]

summary_rows = []

for Tin, fname in files.items():
    df = pd.read_csv(fname)

    # ---- Methane conversion definition
    MW_CH4 = 16.043e-3  # kg/mol

    # proportional methane mass flow (area cancels in the ratio)
    mCH4_rel = df["CH4_X"] * (df["density[kg/m3]"] * df["velocity[m/s]"] / df["mean_MW[kg/mol]"]) * MW_CH4

    mCH4_in = mCH4_rel.iloc[0]
    df["X_CH4"] = 1.0 - (mCH4_rel / mCH4_in)


    weighted_X = 0.0

    print(f"\nTin = {Tin} °C")
    print("-" * 40)

    for tmin, tmax, w in time_bins:
        mask = (df["t[s]"] >= tmin) & (df["t[s]"] < tmax)
        X_avg = df.loc[mask, "X_CH4"].mean()

        weighted_X += w * X_avg

        summary_rows.append({
            "Tin_C": Tin,
            "t_min_s": tmin,
            "t_max_s": tmax,
            "flow_fraction": w,
            "X_CH4_avg": X_avg
        })

        print(f"{tmin:4.1f}–{tmax:4.1f} s | "
              f"flow = {w:6.3f} | "
              f"X_CH4_avg = {X_avg:7.4f}")

    print(f"Weighted-average X_CH4 = {weighted_X:.4f}")

    # ---- Histogram
    plt.figure()
    plt.hist(df["X_CH4"].dropna(), bins=30)
    plt.xlabel("Methane Conversion")
    plt.ylabel("Count")
    plt.title(f"Methane Conversion Histogram (Tin = {Tin} °C)")
    plt.show()

    summary_rows.append({
        "Tin_C": Tin,
        "t_min_s": "ALL",
        "t_max_s": "ALL",
        "flow_fraction": 1.0,
        "X_CH4_avg": weighted_X
    })

# -------------------------------
# SUMMARY TABLE
# -------------------------------
summary = pd.DataFrame(summary_rows)
print("\n=== Summary Table ===")
print(summary)
