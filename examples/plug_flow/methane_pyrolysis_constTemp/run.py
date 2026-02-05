from collections import namedtuple
import numpy as np
from simulate_flowreactor import simulate_flowreactor

PAH_growth_model_types = ["ReactiveDimerization"]
particle_dynamics_model_types = ["Monodisperse"]

TemperatureProfile = namedtuple("TemperatureProfile", ["z", "T"])

# Reactor grid
z_max = 1000.0      # cm
n_pts = 1000
z = np.linspace(0.0, z_max, n_pts)

# Temperature sweep (°C → K)
T_C_list = np.arange(1000, 2501, 500)   # 1000–2500 °C
T_K_list = T_C_list + 273.15

# Base arguments (temperature_profile added later)
arg_dict = dict(
    mech_name="ABF",
    PAH_growth_model_type=PAH_growth_model_types[0],
    particle_dynamics_model_type=particle_dynamics_model_types[0],
    #precursors=["C10H8", "C14H10", "C16H10", "C12H8"],
    precursors=["A2", "A3", "A4", "A2R5"],
    inlet_composition={
        "CH4": 0.968277,
        "C2H6": 0.010412,
        "CO2": 0.009968,
        "N2":  0.008941,
    },
    mdot=0.02,
    P=101350,
    reactor_length=10,
)

# Run sweep
for T_K in T_K_list:
    print(f"\nRunning simulation at T = {T_K - 273.15:.0f} °C")

    T_profile = np.full_like(z, T_K)
    temperature_profile = TemperatureProfile(z=z, T=T_profile)

    arg_dict["temperature_profile"] = temperature_profile

    for particle_dynamics_model_type in particle_dynamics_model_types:
        arg_dict["particle_dynamics_model_type"] = particle_dynamics_model_type
        simulate_flowreactor(**arg_dict)
