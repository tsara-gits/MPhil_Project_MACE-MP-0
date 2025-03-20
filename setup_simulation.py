import os
import subprocess
import numpy as np

from ase import units, io
from ase.build import molecule
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_mp
from ase.io import read, write
import time

def get_simulation_box(protein_file, sim_box_margin):
    """
    Computes the simulation box dimensions around the protein with a given margin,
    and translates the protein so that it is centered in the box.
    
    Inputs:
    - protein_file: path to the protein structure file
    - sim_box_margin: extra space in Å to add around the protein
    
    Outputs:
    - sim_box_coords: list of coordinates defining the box
    - cell: 3x3 matrix defining the simulation cell
    - translation_vector: vector by which the protein should be shifted to center it
    """
    protein_atoms = read(protein_file, index=0)
    positions = protein_atoms.get_positions()
    center_of_mass = protein_atoms.get_center_of_mass()
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    margin = sim_box_margin  # extra space (in Å) around the protein
    box_min = min_coords - margin
    box_max = max_coords + margin
    x_min, y_min, z_min = box_min
    x_max, y_max, z_max = box_max

    # Create a cell from the box dimensions
    cell = [[x_max - x_min, 0, 0],
        [0, y_max - y_min, 0],
        [0, 0, z_max - z_min]]

    sim_box_coords = [x_min, y_min, z_min, x_max, y_max, z_max]
    translation_vector = np.array([0,0,0])
 
    return sim_box_coords, cell, translation_vector

def write_packmol_script(sim_box_coords, protein_file, water_pdb, packmol_script_path, solvated_protein_pdb, num_waters, tolerance):
    """
    Creates a Packmol input script for solvating the protein with water molecules.
    
    Inputs:
    - sim_box_coords: box coordinates defining solvation region
    - protein_pdb: protein structure file path
    - water_pdb: water molecule structure file path
    - packmol_script_path: path to Packmol script
    - solvated_protein_pdb: output solvated structure path
    - num_waters: number of water molecules to add
    - tolerance: Packmol packing tolerance
    """
    os.makedirs(os.path.dirname(packmol_script_path), exist_ok=True)
    for file in [packmol_script_path, solvated_protein_pdb]:
        if os.path.exists(file): os.remove(file)
        open(file, 'w').close()
    
    input_text = f"""tolerance {tolerance}
filetype pdb
output {solvated_protein_pdb}

structure {protein_file}
  fixed 0.0 0.0 0.0 0.0 0.0 0.0
end structure

structure {water_pdb}
  number {num_waters}
  inside box {sim_box_coords[0]} {sim_box_coords[1]} {sim_box_coords[2]} {sim_box_coords[3]} {sim_box_coords[4]} {sim_box_coords[5]}
end structure
"""
    with open(packmol_script_path, "w") as f:
        f.write(input_text)
    print("Packmol script file created")



def solvate_protein(packmol_script_path):
    """
    Runs Packmol to solvate the protein.
    
    Inputs:
    - packmol_script_path: path to Packmol input script
    """
    packmol_cmd = f"packmol < {packmol_script_path}"
    print("Running Packmol...")
    result = subprocess.run(packmol_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Packmol error:", result.stderr)
        exit(1)
    print("Solvation complete")

def run_NVT_ASE_simulation(protein_name, num_waters, sim_box_margin, solvated, initial_structure, cell, traj_file, data_file, T_init, timestep, n_steps, friction, sampling_freq, model, dispersion, enable_cueq, translation_vector):
    """
    Runs a molecular dynamics simulation using ASE's Langevin integrator.
    
    Inputs:
    - protein_name: name of the protein
    - num_waters: number of water molecules
    - solvated: whether the system is solvated
    - initial_structure: input structure file
    - cell: simulation box dimensions
    - traj_file: path to output trajectory file
    - data_file: path to output energy log file
    - T_init: initial temperature (K)
    - timestep: simulation timestep (fs)
    - n_steps: number of MD steps
    - friction: friction coefficient for Langevin dynamics
    - sampling_freq: frequency for logging data
    - model: MACE potential model to use
    - dispersion: whether dispersion interactions are enabled
    - enable_cueq: whether to enable charge equilibration
    - translation_vector: vector to shift protein to the center of the box
    """
    # Prepare output files
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    for file in [traj_file, data_file]:
        if os.path.exists(file): 
            os.remove(file)
        open(file, 'w').close()
    with open(traj_file, 'w') as f:
        f.write("")
    with open(data_file, 'w') as f:
        f.write("Step Potential_Energy(eV) Kinetic_Energy(eV) Total_Energy(eV) Temperature(K) Interval_Time(s)\n")
    
    # Set up simulation
    atoms = read(initial_structure)
    positions = atoms.get_positions()
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    bounding_box_center = (max_coords + min_coords) / 2  # midpoint of the bounding box
    box_center = np.array([cell[0][0] / 2, cell[1][1] / 2, cell[2][2] / 2])  # sim box center
    translation_vector = box_center - bounding_box_center
    #atoms.set_positions(atoms.get_positions() + translation_vector)   # center the atoms for better visualisation
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    calc = mace_mp(enable_cueq=enable_cueq, dispersion=dispersion, model=model)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_init)
    dyn = Langevin(atoms, timestep, T_init * units.kB, friction)

    # Create attachments to write output files
    def write_frame():
        atoms.write(traj_file, append=True)

    last_log_time = time.time()
    def log_data():
        nonlocal last_log_time
        current_time = time.time()                 # Calc elapsed wall-clock time in seconds
        elapsed_time = current_time - last_log_time
        last_log_time = current_time
        epot = dyn.atoms.get_potential_energy()    # Potential Energy (eV)
        ekin = dyn.atoms.get_kinetic_energy()      # Kinetic Energy (eV)
        etot = epot + ekin                         # Total Energy (eV)
        temp = ekin / (1.5 * units.kB * len(atoms))  # Temperature (K)

        with open(data_file, 'a') as f:
            f.write(f"{dyn.nsteps} {epot:.6f} {ekin:.6f} {etot:.6f} {temp:.2f} {elapsed_time:.3f}\n")

    dyn.attach(write_frame, interval=sampling_freq)
    dyn.attach(log_data, interval=sampling_freq)

    # Run the simulation
    if solvated == True:
        print(f"Running MD simulation for {n_steps} steps...\n - protein: {protein_name}\n - solvated: {solvated}\n - n_waters: {num_waters}\n - simulation box margin: {sim_box_margin}\n")
    else:
        print(f"Running MD simulation for {n_steps} steps...\n - protein: {protein_name}\n - solvated: {solvated}")
    dyn.run(n_steps)
    print("MD simulation complete.\n")
    print(f"Outputs files are found at:\n  - PDB trajectory: {traj_file}\n  - Data log: {data_file}\n")