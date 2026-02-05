import numpy as np
from simulate_flowreactor import simulate_flowreactor

PAH_growth_model_types = ["ReactiveDimerization"]
particle_dynamics_model_types = ["Monodisperse"]

# Temperature sweep (°C → K)
# Your current range is 1400–2000 °C in steps of 200 °C.
T_C_list = np.arange(1400, 2001, 200)
T_K_list = T_C_list + 273.15

# Base arguments (initial_T_K added later)
arg_dict = dict(
    mech_name="ABF.yaml",  # must be a valid Cantera mechanism file (YAML)
    PAH_growth_model_type=PAH_growth_model_types[0],
    particle_dynamics_model_type=particle_dynamics_model_types[0],
    # precursors=["C10H8", "C14H10", "C16H10", "C12H8"],
    precursors=["A2", "A3", "A4", "A2R5"],
    inlet_composition={
        "CH4": 0.968277,
        "C2H6": 0.010412,
        "CO2": 0.009968,
        "N2":  0.008941,
    },
    mdot=0.02,
    P=101350,
    reactor_length=25,
    # optional overrides if you want:
    # output_dir="results/Adiabatic",
    # snapshot_every=20,
    # max_step=1e-4,
)

# Run sweep
for T_C, T_K in zip(T_C_list, T_K_list):
    print(f"\nRunning simulation at T = {T_C:.0f} °C")

    arg_dict["initial_T_K"] = float(T_K)

    for pdm in particle_dynamics_model_types:
        arg_dict["particle_dynamics_model_type"] = pdm
        simulate_flowreactor(**arg_dict)
