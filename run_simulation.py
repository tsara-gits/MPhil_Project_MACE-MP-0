import os
import sys
import torch
import warnings
from setup_simulation import *

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("""
    
    Welcome to the simulation of a mini-protein with the foundational machine learning model MACE-MP-0! 
    In this simulation, you will have the option to choose a protein that you would like to simulate with various 
    parameters such as solvation conditions, simulation box margin, number of water molecules, and the number of steps for molecular dynamics.
    See further documentation about this simulation at https://github.com/tsara-gits/MPhil_Project_MACE-MP-0/

    """)

    """
    Defining the simulation parameters:

    - protein_name:    Protein structure input file (use pdb)
    - additional_name: Additional name for the simulation (string, optional)
    - solvated:        Defines if the simulation is run in a solvated environment or in vacuum (bool, True: solvated, False: vacuum)
    - sim_box_margin:  Extra space (in Å) added around the protein for defining the simulation box
    - num_waters:      Number of water molecules added to the system if solvated (int, required only if solvated=True)
    - n_steps:         Number of molecular dynamics steps (1 step = 1 fs)
    - sampling_freq:   Frequency of logging/saving simulation data (interval in steps)
    - output_dir_base: Directory where simulation trajectory and log files will be stored. Require large storage  

    If the the simulation is run solvated, it will create a directory named (protein_name + solv/unsolv + additional_name) at the current directory   
    and saves the packmol script (inp) needed for the solvation and the solvated structure (pdb) in it.
    """

    current_dir = os.getcwd()
    while True:
        protein_name = input(" - Full name of the protein file in the INPUTS folder that you would like to simulate: ")
        protein_path = os.path.join(current_dir, "INPUTS", protein_name)
        if os.path.isfile(protein_path):
            break
        else:
            print("Error: The protein file does not exist in the INPUTS folder. Please enter a valid file name.")
    
    additional_name = input(" - Additional name of the simulation: (none: press enter, else: type the name) ")
    additional_name = "_" + additional_name if additional_name else ""
    solvated_input = input(" - Run the simulation solvated? (yes/no): ")
    solvated = True if solvated_input.lower() == "yes" else False
    sim_box_margin = float(input(" - Margin of the simulation box around the protein (Å): "))
    num_waters = int(input(" - Number of water molecules to solvate the system: ")) if solvated else 0
    n_steps = int(input(" - Number of simulation steps (1 step = 1 fs): "))                                 # 1 ns = 1,000,000 steps (1 fs per step)
    sampling_freq = int(input(" - Sampling frequency (sampling the system every x steps): "))               # 1 ps = 1000 fs, so sample every 1000 steps 

    output_dir_default = "/local/data/public/st958/PROTEIN_SIM2/"
    change_output_dir = input(f" - Default directory to store the MD trajectory: {output_dir_default}. Change it? (yes/no): ")
    if change_output_dir.lower() == "yes":
        output_dir_base = input(" - Path to the new directory for storing the MD trajectory: ")
    else:
        output_dir_base = output_dir_default
    print("\n Starting simulation! ")

    # Hard-coded simulation parameters
    tolerance = 2               # Packmol packing tolerance for solvation setup, minimum allowed distance between molecules (Å)
    model = 'medium'            # MACE potential model used for the simulation
    dispersion = False          # Enables dispersion interactions if set to True
    enable_cueq = False         # Enables charge equilibration if set to True
    T_init = 300                # Initial temperature of the simulation (K)
    timestep = 1.0 * units.fs   # Simulation timestep (fs)
    friction = 0.01 / units.fs  # Friction coefficient for Langevin dynamics (fs^-1)

    # Define paths
    env = "solv" if solvated else "unsolv"
    protein_file = os.path.join(current_dir, "INPUTS", protein_name)
    water_pdb = os.path.join(current_dir, "INPUTS", "water.pdb")
    protein_name = protein_name.replace(".pdb", "")
    output_dir = os.path.join(current_dir, f"{protein_name}_{env}{additional_name}")
    packmol_script_path = os.path.join(output_dir, "packmol_script.inp")
    solvated_protein_pdb = os.path.join(output_dir, "solvated_protein.pdb")
    traj_file = os.path.join(output_dir_base, f"{protein_name}_{env}{additional_name}/trajectory.pdb")
    data_file = os.path.join(output_dir_base, f"{protein_name}_{env}{additional_name}/data_log.txt")

    # Get simulation coordinates and translation vector to center the protein
    sim_box_coords, cell, translation_vector = get_simulation_box(protein_file, sim_box_margin)

    # Set up the simulation
    if solvated:
        write_packmol_script(sim_box_coords, protein_file, water_pdb, packmol_script_path, solvated_protein_pdb, num_waters, tolerance)
        solvate_protein(packmol_script_path)
        initial_structure = solvated_protein_pdb
    else:
        
        initial_structure = protein_file

    # Run the simulation with centering
    run_NVT_ASE_simulation(protein_name, num_waters, sim_box_margin, solvated, initial_structure, cell, traj_file, data_file, T_init, timestep, n_steps, friction, 
                           sampling_freq, model, dispersion, enable_cueq, translation_vector)

if __name__ == "__main__":
    main()
