import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define the directories containing the rg_values.txt files
    directories = {
        "1uao unsolvated, neutral": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM/1uao_neutral_unsolv_box",
        "1uao unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM/1uao_unsolv_box",
        "1uao solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM/1uao_solv_box_FILTERED",
        "5awl AIMD reference data": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_AIMD",
        "5awl folded solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_folded_solv",
        "5awl folded unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_folded_unsolv",
        "5awl unfolded solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_unfolded_solv",
        "5awl unfolded unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_unfolded_unsolv",
    }

    colors = [ "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:cyan" ]

    current_dir = os.getcwd()
    save_plot = os.path.join(current_dir, "rg_comparison.png")

    # Initialize plot
    plt.figure(figsize=(10, 6))
    for (label, directory), color in zip(directories.items(), colors):
        file_path = os.path.join(directory, "rg_values.txt")
        if os.path.exists(file_path):
            data = np.loadtxt(file_path, skiprows=1)
            plt.plot(data[:, 0], data[:, 1], linestyle="-", color=color, label=label)

    # Plot limits
    plt.xlim(0, 1000)
    plt.ylim(0, 10)
    plt.xlabel("Time (ps)")
    plt.ylabel("Radius of Gyration (Å)")
    plt.title("Comparison of Radius of Gyration Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_plot, dpi=300)
    plt.show()

    print(f"Plot saved as {save_plot}")

if __name__ == "__main__":
    main()