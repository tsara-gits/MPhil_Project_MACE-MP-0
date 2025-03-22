import os
import numpy as np
import matplotlib.pyplot as plt

def moving_avg_std(data, avg_window=10):
    """
    Averages Rg data over fixed-size time windows (e.g. 10 ps per window)

    Inputs:
        - data (np.ndarray): 2D array with shape (N, 2), where:
                           - column 0 = time (ps)
                           - column 1 = radius of gyration
        - avg_window (int): number of data points (ps) to average over

    Outputs:
        - bin_centers (np.ndarray): center of each time window
        - mean_rg (np.ndarray): mean Rg per window
        - std_rg (np.ndarray): standard deviation of Rg per window
    """
    time = data[:, 0]
    rg = data[:, 1]

    mean_rg = []
    std_rg = []
    bin_centers = []

    for i in range(0, len(time), avg_window):
        time_slice = time[i:i+avg_window]
        rg_slice = rg[i:i+avg_window]

        if len(time_slice) == 0:
            continue

        mean_rg.append(np.mean(rg_slice))
        std_rg.append(np.std(rg_slice))
        bin_centers.append(np.mean(time_slice))

    bin_centers = np.array(bin_centers)
    mean_rg =  np.array(mean_rg)
    std_rg = np.array(std_rg)

    return bin_centers, mean_rg, std_rg 

def plot_rg_comparison(directories, colors, avg_window):
    
    current_dir = os.getcwd()
    save_plot = os.path.join(current_dir, "rg_comparison.png")

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    for i in range(len(directories)):
        label = list(directories.keys())[i]
        directory = list(directories.values())[i]
        color = colors[i]
        file_path = os.path.join(directory, "rg_values.txt")

        if os.path.exists(file_path):
            data = np.loadtxt(file_path, skiprows=1)
            time_avg, mean_rg, std_rg = moving_avg_std(data, avg_window)
            plt.plot(time_avg, mean_rg, label=label, color=color)
            plt.fill_between(time_avg, mean_rg - std_rg, mean_rg + std_rg, color=color, alpha=0.2)


    plt.xlim(0, 1000)
    plt.ylim(5, 11)
    plt.xlabel("Time (ps)", fontsize=12)
    plt.ylabel("Radius of Gyration (Å)", fontsize=12)
    plt.title("Comparison of Radius of Gyration Over Time", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300)
    plt.close()


    print(f"Plot saved as {save_plot}")

def main():
    directories = {
        "1uao unsolvated, neutral": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/1uao_neutral_unsolv",
        "1uao unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/1uao_unsolv",
        "1uao solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/1uao_solv",
        "5awl AIMD reference data": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_AIMD",
        "5awl folded solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_folded_solv",
        "5awl folded unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_folded_unsolv",
        "5awl unfolded solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_unfolded_solv",
        "5awl unfolded unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_unfolded_unsolv",
    }

    colors = [ "tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown", "tab:pink", "tab:cyan" ]

    plot_rg_comparison(directories, colors, avg_window=10)
    



if __name__ == "__main__":
    main()
