import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

def moving_avg_std(data, avg_window=10):
    '''
    Calculates the  average and standard deviation of the dataover a sliding window.

    Inputs:
        - data (ndarray): 2D np array
            - first column: time
            - second column: data (radius of gyration)
        - avg_window (int): number of data points per window for averaging

    Outputs:
        - bin_centers: average time per window
        - mean_values: average of the data in each window
        - std_values: the standard deviation in each window
    '''
    
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

    return np.array(bin_centers), np.array(mean_rg), np.array(std_rg)


# Define maps for plotting
    # The labels correspond to a specific initial protein conformation and simulation environment 
    # for example folded vs unfolded, solvated vs unsolvated, or ab initio reference,
    # and is used to differentiate the conditions in the plot legends and color map.

label_map = {"5awl unfolded solvated": "5AWL - Unfolded (solvated)",
            "5awl unfolded unsolvated": "5AWL - Unfolded (vacuum)",
            "5awl AIMD reference data": "5AWL - Ab-Initio reference",
            "5awl AIMD reference data (1uao)": "5AWL - Ab-initio reference (implicint solvent)",
            "5awl folded solvated": "5AWL - Folded (solvated)",
            "5awl folded unsolvated": "5AWL - Folded (vacuum)",
            "1uao unsolvated, neutral": "1UAO - Neutral residues (vacuum)",
            "1uao unsolvated": "1UAO - Charged Residues (vacuum)"}

color_map = {"5awl unfolded solvated": "tab:cyan",
            "5awl unfolded unsolvated": "tab:pink",
            "5awl AIMD reference data": "#FFD700",
            "5awl folded solvated": "tab:green",
            "5awl folded unsolvated": "tab:red",
            "5awl AIMD reference data (1uao)": "#FFD700",
            "1uao unsolvated": "tab:blue",
            "1uao unsolvated, neutral": "tab:purple"}

legend_title_map = {0: "Initial conformation and simulation environment:",
                        1: "Initial conformation and simulation environment:"}


def plot_stacked_rg(directories, color_map, plot_orders, ylims_list, save_filename, avg_window=10):
    '''
    Plots vertically stacked subplots showing the radius of gyration over time for different simulation conditions.

    Parameters:
        - directories (dict): dictionary mapping labels to folders that contain the'rg_values.txt' data
        - color_map (dict): maps each label to its plot colour
        - plot_orders (list of lists): two lists, each defining in what order to plot in the subplots
        - ylims_list (list of tuples): y-axis limits for each subplot
        - save_filename (str): name of the output file to save the figure
        - avg_window (int): window size for computing the moving average and standard deviation

    Outputs:
        - each subplot corresponds to a different protein (5AWL and 1UAO)
        - in the plotting, smoothing is applied to the raw data using `moving_avg_std`
        - the AIMD reference data is highlighted with distinct yellow background
        - the resulting figure is saved a as a PNG file
    '''
    # initialization
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, save_filename)
    fig, axs = plt.subplots(2, 1, figsize=(12, 15), sharex=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # parameters for plot- sizing
    subtitles = ["a) Gyration Radius of 5AWL Protein vs Simulation Time",
                "b) Gyration Radius of 1UAO Protein vs Simulation Time"]
    label_fontsize = 28
    title_fontsize = 30
    legend_fontsize = 23
    legend_title_fontsize= 26
    tick_fontsize = 28
    line_width = 3
    tick_pads = 11
    title_pad = 35
    subplots_spacing = 0.4
    legend_pos = (1, 1.03)

    # do the subplots
    for ax, plot_order, ylims, subtitle in zip(axs, plot_orders, ylims_list, subtitles):
        for label in plot_order:
            directory = directories[label]
            color = color_map[label]
            file_path = os.path.join(directory, "rg_values.txt")

            if os.path.exists(file_path):
                data = np.loadtxt(file_path, skiprows=1)
                time_avg, mean_rg, std_rg = moving_avg_std(data, avg_window)

                if label == "5awl AIMD reference data":
                    ax.plot(time_avg, mean_rg, label=label_map.get(label, label), color='#696969', alpha=1, linewidth=line_width)
                    ax.fill_between(time_avg, mean_rg - std_rg, mean_rg + std_rg, color='#FFD700', alpha=0.4)
                    ax.plot([], [], color='none', label="              (implicit solvent)")
                
                elif label == "5awl AIMD reference data (1uao)":
                    ax.plot(time_avg, mean_rg, label=label_map.get(label, label), color='#696969', alpha=1, linewidth=line_width)
                    ax.fill_between(time_avg, mean_rg - std_rg, mean_rg + std_rg, color='#FFD700', alpha=0.4)

                else:
                    ax.plot(time_avg, mean_rg, label=label_map.get(label, label), color=color, linewidth=line_width)
                    ax.fill_between(time_avg, mean_rg - std_rg, mean_rg + std_rg, color=color, alpha=0.3)

        # set the subplot attributes
        ax.set_xlim(0, 1000)
        ax.set_ylim(ylims[0], ylims[1])
        if subtitle.startswith("b) Gyration Radius of 1UAO"):
            ax.set_yticks(np.arange(5, 9, 1))  # Show ticks only at integers
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Grid at every 0.5
            ax.grid(which='both', linestyle='-', linewidth=1.0, alpha=1.0)  # Same style for all gridlines

        ax.set_ylabel("Radius of Gyration (Å)", fontsize=label_fontsize)
        #ax.set_xlabel("Time (ps)", fontsize=label_fontsize)
        ax.set_title(subtitle, fontsize=title_fontsize, weight='bold', pad=title_pad)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.tick_params(axis='x', pad=tick_pads)
        ax.tick_params(axis='y', pad=tick_pads)
        ax.grid(True)

        # set the subplot legend
        legend = ax.legend(title=legend_title_map.get(axs.tolist().index(ax), None),
                            title_fontproperties=FontProperties(size=legend_title_fontsize),
                            fontsize=legend_fontsize,
                            handletextpad=0.5,
                            handlelength=1,
                            labelspacing=0.3,
                            loc='upper right',
                            frameon=False,
                            alignment = 'right',
                            bbox_to_anchor=legend_pos)
        for legobj in legend.legend_handles:
                legobj.set_linewidth(4.0)

    # set main plot attributes
    axs[1].set_xlabel("Time (ps)", fontsize=label_fontsize)
    axs[0].set_xlabel("Time (ps)", fontsize=label_fontsize)
    axs[0].tick_params(labelbottom=True)  # x-ticks on top plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=subplots_spacing)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved as {save_path}")


# Run the main function
def main():
    directories = {"1uao unsolvated, neutral": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/1uao_neutral_unsolv",
                "1uao unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/1uao_unsolv",
                "5awl AIMD reference data": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_AIMD",
                "5awl AIMD reference data (1uao)": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_AIMD",
                "5awl folded solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_folded_solv",
                "5awl folded unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_folded_unsolv",
                "5awl unfolded solvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_unfolded_solv",
                "5awl unfolded unsolvated": "/home/raid/st958/mphil_assignment_mace/PROTEIN_SIM2/5awl_unfolded_unsolv",}

    plot_order_5awl = ["5awl unfolded solvated",
                        "5awl unfolded unsolvated",
                        "5awl folded solvated",
                        "5awl folded unsolvated",
                        "5awl AIMD reference data"]

    plot_order_1uao = ["5awl AIMD reference data (1uao)",
                    "1uao unsolvated",
                    "1uao unsolvated, neutral"]

    ylims_list = [[5.5, 11],  [5, 8]]

    plot_stacked_rg(directories=directories, color_map=color_map, plot_orders=[plot_order_5awl, plot_order_1uao], 
    ylims_list=ylims_list, save_filename="rg_comparison_stacked.png")

if __name__ == "__main__":
    main()
