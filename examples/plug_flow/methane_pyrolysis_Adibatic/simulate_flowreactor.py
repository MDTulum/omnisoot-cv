import os
import time
import csv
import numpy as np
import cantera as ct
from omnisoot import PlugFlowReactor, SootGas

Av = 6.0221408e23  # 1/mol


def simulate_flowreactor(
    mech_name,
    PAH_growth_model_type,
    particle_dynamics_model_type,
    precursors,
    inlet_composition,
    mdot,
    P,
    initial_T_K,
    reactor_length,
    output_dir="results/Adiabatic",
    snapshot_every=20,
    max_step=1e-4,
):
    """
    Simulate an OmniSoot PlugFlowReactor and write results to CSV.

    Parameters
    ----------
    mech_name : str
        Cantera mechanism YAML path (e.g., 'ABF.yaml')
    PAH_growth_model_type : str
    particle_dynamics_model_type : str
    precursors : list[str]
        Names of soot precursors
    inlet_composition : str or dict
        Cantera composition
    mdot : float
        Inlet mass flow rate [kg/s]
    P : float
        Pressure [Pa]
    initial_T_K : float
        Inlet temperature [K]
    reactor_length : float
        Reactor length [m]
    output_dir : str
        Output directory
    snapshot_every : int
        Append a state every N solver steps
    max_step : float
        PFR max step [m] (or solver step control, per OmniSoot API)
    """

    # --- Gas / soot wrapper
    gas = ct.Solution(mech_name)
    soot_gas = SootGas(gas)

    # --- Reactor config
    pfr = PlugFlowReactor(soot_gas)
    pfr.inlet.TPX = initial_T_K, P, inlet_composition
    pfr.inlet.mdot = mdot
    pfr.temperature_solver_type = "energy_equation"
    pfr.max_step = max_step

    # --- Soot model config
    soot = pfr.soot
    soot.particle_dynamics_model_type = particle_dynamics_model_type
    soot.PAH_growth_model_type = PAH_growth_model_type
    soot.set_precursor_names(precursors)

    # --- Start integration
    pfr.start()

    # --- SolutionArray snapshots
    states = ct.SolutionArray(
        gas,
        1,
        extra={
            "restime": [0.0],
            "z": [0.0],
            "velocity": [pfr.u],
            "N_agg": [soot.N_agg],
            "N_pri": [soot.N_pri],
            "C_tot": [soot.C_tot],
            "H_tot": [soot.H_tot],
            "A_tot": [soot.A_tot],
            "d_p": [soot.d_p],
            "d_m": [soot.d_m],
            "d_v": [soot.d_v],
            "d_g": [soot.d_g],
            "n_p": [soot.n_p],
            "total_mass": [soot.total_mass],
            "volume_fraction": [soot.volume_fraction],
            "SSA": [soot.SSA],
            "inception_mass": [soot.inception_mass],
            "coagulation_mass": [soot.coagulation_mass],
            "PAH_adsorption_mass": [soot.PAH_adsorption_mass],
            "surface_growth_mass": [soot.surface_growth_mass],
            "oxidation_mass": [soot.oxidation_mass],
            "total_carbon_flux": [pfr.total_carbon_flux],
            "soot_carbon_flux": [pfr.soot_carbon_flux],
            "k_gas": [gas.thermal_conductivity],
            "cp_gas": [gas.cp_mass],
            "mu_gas": [gas.viscosity],
            "total_carbon_mass_flow": [pfr.total_carbon_mass_flow],
            "total_hydrogen_mass_flow": [pfr.total_hydrogen_mass_flow],
            "soot_mass_flow": [pfr.soot_mass_flow],
            "soot_carbon_mass_flow": [pfr.soot_carbon_mass_flow],
        },
    )

    start_time = time.time()
    step = 0

    while pfr.z < reactor_length:
        pfr.step()
        step += 1

        if snapshot_every and (step % snapshot_every == 0):
            states.append(
                gas.state,
                z=pfr.z,
                restime=pfr.restime,
                velocity=pfr.u,
                N_agg=soot.N_agg,
                N_pri=soot.N_pri,
                C_tot=soot.C_tot,
                H_tot=soot.H_tot,
                A_tot=soot.A_tot,
                d_p=soot.d_p,
                d_m=soot.d_m,
                d_v=soot.d_v,
                d_g=soot.d_g,
                n_p=soot.n_p,
                total_mass=soot.total_mass,
                volume_fraction=soot.volume_fraction,
                SSA=soot.SSA,
                inception_mass=soot.inception_mass,
                coagulation_mass=soot.coagulation_mass,
                PAH_adsorption_mass=soot.PAH_adsorption_mass,
                surface_growth_mass=soot.surface_growth_mass,
                oxidation_mass=soot.oxidation_mass,
                total_carbon_flux=pfr.total_carbon_flux,
                soot_carbon_flux=pfr.soot_carbon_flux,
                k_gas=gas.thermal_conductivity,
                cp_gas=gas.cp_mass,
                mu_gas=gas.viscosity,
                total_carbon_mass_flow=pfr.total_carbon_mass_flow,
                total_hydrogen_mass_flow=pfr.total_hydrogen_mass_flow,
                soot_mass_flow=pfr.soot_mass_flow,
                soot_carbon_mass_flow=pfr.soot_carbon_mass_flow,
            )

    end_time = time.time()
    print(f"The simulation time took {end_time - start_time:.6f} s")

    # -----------------------------
    # Build outputs
    # -----------------------------
    soot_columns, soot_data = [], []
    flow_columns, flow_data = [], []
    species_columns, species_data = [], []

    # Soot block
    soot_columns += ["N_agg[mol/kg]", "N_pri[mol/kg]", "C_tot[mol/kg]", "H_tot[mol/kg]"]
    soot_data += [states.N_agg, states.N_pri, states.C_tot, states.H_tot]

    soot_columns += ["A_tot[m2/kg]", "N_agg[#/cm3]", "N_pri[#/cm3]", "N_agg[#/g]", "N_pri[#/g]"]
    soot_data += [
        states.A_tot,
        states.N_agg * Av * states.density / 1e6,
        states.N_pri * Av * states.density / 1e6,
        states.N_agg * Av / 1e3,
        states.N_pri * Av / 1e3,
    ]

    soot_columns += ["d_p[nm]", "d_m[nm]", "d_g[nm]", "d_v[nm]", "n_p"]
    soot_data += [states.d_p * 1e9, states.d_m * 1e9, states.d_g * 1e9, states.d_v * 1e9, states.n_p]

    # Robust carbon yield
    denom = float(states.total_carbon_flux[0])
    if abs(denom) > 0.0:
        carbon_yield = states.soot_carbon_flux / denom
    else:
        carbon_yield = np.full(states.soot_carbon_flux.shape, np.nan)

    soot_columns += [
        "soot_mass[ug/g]",
        "volume_fraction[-]",
        "SSA[m2/g]",
        "total_carbon_flux[kg/m2-s]",
        "carbon_yield[-]",
    ]
    soot_data += [states.total_mass * 1e6, states.volume_fraction, states.SSA, states.total_carbon_flux, carbon_yield]

    soot_columns += [
        "inception_mass[mol/kg-s]",
        "coagulation_mass[mol/kg-s]",
        "PAH_adsorption_mass[mol/kg-s]",
        "surface_growth_mass[mol/kg-s]",
    ]
    soot_data += [states.inception_mass, states.coagulation_mass, states.PAH_adsorption_mass, states.surface_growth_mass]

    # Flow block (include ALL columns you declare)
    flow_columns += [
        "t[s]",
        "z[m]",
        "T[K]",
        "density[kg/m3]",
        "mean_MW[kg/mol]",
        "velocity[m/s]",
        "k_gas[W/m-K]",
        "cp_gas[J/kg-K]",
        "mu_gas[Pa-s]",
        "total_carbon_mass_flow[kg/s]",
        "total_hydrogen_mass_flow[kg/s]",
        "soot_mass_flow[kg/s]",
        "soot_carbon_mass_flow[kg/s]",
    ]
    flow_data += [
        states.restime,
        states.z,
        states.T,
        states.density,
        states.mean_molecular_weight / 1000.0,
        states.velocity,
        states.k_gas,
        states.cp_gas,
        states.mu_gas,
        states.total_carbon_mass_flow,
        states.total_hydrogen_mass_flow,
        states.soot_mass_flow,
        states.soot_carbon_mass_flow,
    ]

    # Species block
    # -------------------------------------------------
    # Species mole fractions (X) and mass fractions (Y)
    # -------------------------------------------------
    species_X_columns = [f"{sp}_X" for sp in gas.species_names]
    species_Y_columns = [f"{sp}_Y" for sp in gas.species_names]

    species_X_data = [states.X[:, i] for i in range(len(gas.species_names))]
    species_Y_data = [states.Y[:, i] for i in range(len(gas.species_names))]

    species_columns = species_X_columns + species_Y_columns
    species_data = species_X_data + species_Y_data


    columns = flow_columns + soot_columns + species_columns
    data = np.array(flow_data + soot_data + species_data, dtype=float).T

    # -----------------------------
    # Write CSV
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    temp0_c = initial_T_K - 273.15
    file_name = f"{output_dir}/sim_results_Tin={int(round(temp0_c))}C.csv"

    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(data)

    print(f"{file_name} was written!")
    return file_name
