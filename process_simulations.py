import numpy as np
import matplotlib.pyplot as plt
import os
from ase import units
from ase.io import read, write

TIMESTEP = 1.0 * units.fs  # 1 fs per step

def get_frames(traj_file):
    """Extract atomic coordinates from a multi-frame PDB trajectory file."""
    frames = []
    atom_names = []
    residue_names = []
    current_frame = []

    with open(traj_file, "r") as f:
        for line in f:
            columns = line.split()
            
            if line.startswith("MODEL"):
                current_frame = []
                atom_names = []
                residue_names = []
            
            elif line.startswith("ATOM"):
                x, y, z = float(columns[6]), float(columns[7]), float(columns[8])
                atom_name = columns[2]
                residue_name = columns[3]
                current_frame.append([x, y, z])
                atom_names.append(atom_name)
                residue_names.append(residue_name)
            
            elif line.startswith("ENDMDL"):
                frames.append((np.array(current_frame), atom_names, residue_names))

    return frames

def extract_protein_frames(traj_file):
    """Removes all lines containing 'MOL' in the 4th column from a PDB trajectory file and saves the cleaned version."""
    
    cleaned_traj_file = traj_file.replace(".pdb", "_cleaned.pdb")

    with open(traj_file, "r") as infile, open(cleaned_traj_file, "w") as outfile:
        for line in infile:
            columns = line.split()
            if len(columns) > 3 and columns[3] == "MOL":
                continue  # Skip lines containing 'MOL' in the residue column
            outfile.write(line)  # Write all other lines

    return cleaned_traj_file



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

def plot_data(x_vals, y_vals, xlabel, ylabel, title, filename, output_dir):
    """Generic function to plot and save simulation data."""
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals / 1000, y_vals, linestyle="-", color="tab:blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.show()

def process_simulation(traj_file, data_file, output_dir, solvated, protein_name):
    """Processes the simulation trajectory and plots various properties."""
    
    # If solvated, remove water molecules and use the cleaned trajectory
    if solvated:
        traj_file = extract_protein_frames(traj_file)

    frames = read(traj_file, index=":")
    rg_vals = calc_rg(frames)
    steps, temps, Epots, Ekins = get_data(data_file)

    save_rg_data(steps, rg_vals, output_dir)

    plot_data(steps, rg_vals, "Time (ps)", "Radius of Gyration (Å)", 
              f"Radius of Gyration vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_rg.png", output_dir)

    plot_data(steps, temps, "Time (ps)", "Temperature (K)", 
              f"Temperature vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_temp.png", output_dir)

    plot_data(steps, Epots, "Time (ps)", "Potential Energy (eV)", 
              f"Potential Energy vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_Epots.png", output_dir)

    plot_data(steps, Ekins, "Time (ps)", "Kinetic Energy (eV)", 
              f"Kinetic Energy vs. Time (ns), {protein_name} {'solvated' if solvated else 'not-solvated'}", 
              "plot_Ekins.png", output_dir)

def main():
    traj_dir = "/local/data/public/st958/PROTEIN_SIM2"
    current_dir = os.getcwd()
    protein_names = ["5awl_folded", "5awl_unfolded"]
    solvated_list = ["solv", "unsolv"]
    for protein_name in protein_names:
        for solvated in solvated_list:
            traj_file = os.path.join(traj_dir, f"{protein_name}_{solvated}/trajectory.pdb")
            data_file = os.path.join(traj_dir, f"{protein_name}_{solvated}/data_log.txt")
            output_dir = os.path.join(current_dir, f"{protein_name}_{solvated}")
            os.makedirs(output_dir, exist_ok=True)
            process_simulation(traj_file, data_file, output_dir, solvated, protein_name)


if __name__ == "__main__":
    main()