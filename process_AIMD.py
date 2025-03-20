import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

def main():
    # Paths and settings
    traj_file = "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/INPUTS/5awl_AIMD_traj.pdb"
    output_dir = "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_AIMD"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    timestep = 1.0  # Assuming 1 fs timestep
    solvated = False
    protein_name = "AIMD_chignolin"
    max_x = 1000  # Set max x-axis limit for the plot (adjust as needed, None for auto)

    # Calculate radius of gyration
    frames = read(traj_file, index=":")  # ":" loads all frames in ASE
    rgs = []
    for atoms in frames:
        positions = atoms.get_positions()
        com = np.mean(positions, axis=0)  # Compute center of mass
        rg = np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))  # Compute Rg
        rgs.append(rg)

    rgs = np.array(rgs)
    steps = np.arange(len(rgs))  # Time steps

    # Save rg values and steps to a txt file
    rg_output_file = os.path.join(output_dir, "rg_values.txt")
    np.savetxt(rg_output_file, np.column_stack((steps, rgs)),  # Changed `rg_vals` -> `rgs`
            header="Time(ps) Radius_of_Gyration(A)", fmt="%.6f")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(steps, rgs, linestyle="-", color="tab:red")  # Changed `rg_vals` -> `rgs`
    plt.xlabel("Time (ps)") 
    plt.ylabel("Radius of Gyration (Å)")
    if solvated:
        plt.title(f"Radius of Gyration vs. Time (ns), {protein_name} protein, solvated")
    else:
        plt.title(f"Radius of Gyration vs. Time (ns), {protein_name} not-solvated")
    plt.grid(True)
    if max_x is not None:
        plt.xlim(0, max_x)
    plot_output_path = os.path.join(output_dir, "plot_rg.png")
    plt.savefig(plot_output_path, dpi=300)

    print(f"Saved radius of gyration values to {rg_output_file}")
    print(f"Saved plot to {plot_output_path}")
    
if __name__ == "__main__":
    main()
