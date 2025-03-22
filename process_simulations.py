import numpy as np
import matplotlib.pyplot as plt
import os
from ase import units
from ase.io import read, write

TIMESTEP = 1.0 * units.fs  # 1 fs per step


def extract_protein_frames(traj_file):
    """Removes all lines containing 'MOL' in the 4th column from a PDB trajectory file and saves the cleaned version."""
    
    water_free_traj_file = traj_file.replace(".pdb", "_water_free.pdb")

    with open(traj_file, "r") as infile, open(water_free_traj_file, "w") as outfile:
        for line in infile:
            columns = line.split()
            if len(columns) > 3 and columns[3] == "MOL":
                continue  # skip lines containing 'MOL' in the residue column
            outfile.write(line)  # write all other lines

    return water_free_traj_file

def filter_atoms(traj_file, exclude_atom_idx):
    """
    Removes the atoms from the PDB trajectory file with index that in the
 excluded_atoms list, and saves the cleaned version.
    
    Inputs:
        traj_file (str): path to the input PDB file.
        excluded_atoms (list of int): List of atom names to exclude (e.g., [1, 40]).
    
    Outputs:
        str: Path to the cleaned PDB file.
    """
    filtered_traj_file = traj_file.replace(".pdb", "_filtered.pdb")

    with open(traj_file, "r") as infile, open(filtered_traj_file, "w") as outfile:
        for line in infile:
            columns = line.split()
            if len(columns) > 3 and columns[1] in exclude_atom_idx:
                continue  # skip lines with excluded atom names
            outfile.write(line)

    return filtered_traj_file

def get_data(data_file):
    """Extracts simulation time and temperature from the data file using hardcoded timestep."""
    steps, temps, Epots, Ekins = [], [], [], []

    with open(data_file, "r") as f:
        next(f)  # Skip header line
        for line in f:
            columns = line.split()
            steps.append(int(columns[0]))
            Epots.append(float(columns[1]))
            Ekins.append(float(columns[2]))
            temps.append(float(columns[4]))

    steps = np.array(steps) * (TIMESTEP / units.fs)
    return steps, temps, Epots, Ekins

def calc_rg(frames):
    """Calculates the radius of gyration (Rg) for each frame."""
    rgs = []
    for atoms in frames:
        positions = atoms.get_positions()
        com = np.mean(positions, axis=0)  # Compute center of mass
        rg = np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))  # Compute Rg
        rgs.append(rg)
    return np.array(rgs)

def save_rg_data(steps, rg_vals, output_dir):
    """Saves the Radius of Gyration vs. Time data."""
    rg_output_file = os.path.join(output_dir, "rg_values.txt")
    np.savetxt(rg_output_file, np.column_stack((steps / 1000, rg_vals)), 
               header="Time (ps) Radius_of_Gyration (Å)", fmt="%.6f")
    print(f"Saved Rg data to {rg_output_file}")

def plot_data(x_vals, y_vals, xlabel, ylabel, title, filename, output_dir_analysis):
    """Generic function to plot and save simulation data."""
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals / 1000, y_vals, linestyle="-", color="tab:blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir_analysis, filename), dpi=100)
    plt.close()

def process_simulation(traj_file, data_file, output_dir_analysis, output_dir_traj, solvated, protein_name, exclude_atom_idx):
    """Processes the simulation trajectory and plots various properties."""
    
    # If solvated, remove water molecules and use the cleaned trajectory
    if solvated:
        traj_file = extract_protein_frames(traj_file)
    
    # Read in the trajectory
    frames = read(traj_file, index=":")

    # If the exclude exclude_atom_idx is not empty, filter the atoms with the index specified in the list
    if exclude_atom_idx:
        filtered_frames = [atoms[[i for i in range(len(atoms)) if i not in exclude_atom_idx]]
                           for atoms in frames]

        filtered_traj_file = os.path.join(output_dir_traj, "trajectory_filtered.pdb")
        write(filtered_traj_file, filtered_frames)
        print(f"Saved filtered trajectory to: {filtered_traj_file}")
        
        frames = filtered_frames

    rg_vals = calc_rg(frames)
    steps, temps, Epots, Ekins = get_data(data_file)

    save_rg_data(steps, rg_vals, output_dir_analysis)

    plot_data(steps, rg_vals, "Time (ps)", "Radius of Gyration (Å)", 
              f"Radius of Gyration vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_rg.png", output_dir_analysis)

    plot_data(steps, temps, "Time (ps)", "Temperature (K)", 
              f"Temperature vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_temp.png", output_dir_analysis)

    plot_data(steps, Epots, "Time (ps)", "Potential Energy (eV)", 
              f"Potential Energy vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_Epots.png", output_dir_analysis)

    plot_data(steps, Ekins, "Time (ps)", "Kinetic Energy (eV)", 
              f"Kinetic Energy vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_Ekins.png", output_dir_analysis)

def main():
    protein_names = ["5awl_folded", "5awl_unfolded", "1uao", "1uao_neutral"]
    solvated_list = ["solv", "unsolv"]
    traj_dir = "/local/data/public/st958/PROTEIN_SIM_SAVED"
    
    filter_atoms_dict = {"5awl_folded_solv": [80, 81, 82],
                            "5awl_folded_unsolv": [80, 81, 31],
                            "5awl_unfolded_solv": [163, 164, 165, 146, 16, 37],
                            "5awl_unfolded_unsolv": [145, 146, 144],
                            "1uao_neutral_unsolv": [],
                            "1uao_solv": [133, 132, 134, 0, 4, 5, 6, 135],
                            "1uao_unsolv": [],}

    current_dir = os.getcwd()
    for protein_name in protein_names:
        for solvated in solvated_list:
            
            traj_file = os.path.join(traj_dir, f"{protein_name}_{solvated}/trajectory.pdb")
            if not os.path.exists(traj_file):
                print(f"Trajectory file does not exist: {traj_file}")
                continue
            data_file = os.path.join(traj_dir, f"{protein_name}_{solvated}/data_log.txt")
            output_dir_analysis = os.path.join(current_dir, f"{protein_name}_{solvated}")
            output_dir_traj = os.path.join(traj_dir, f"{protein_name}_{solvated}")
            os.makedirs(output_dir_analysis, exist_ok=True)
    
            key = f"{protein_name}_{solvated}"
            exclude_atom_idx = filter_atoms_dict.get(key, [])
            process_simulation(traj_file, data_file, output_dir_analysis, output_dir_traj, solvated, protein_name, exclude_atom_idx)

if __name__ == "__main__":
    main()
