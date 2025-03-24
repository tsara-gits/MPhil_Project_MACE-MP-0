# MPhil_Project: A foundation model for atomistic materials chemistry: MACE-MP-0, supplementary code

This project simulates the chignoling protein using the pre-trained MACE-MP-0 potential of the MACE architecture (https://github.com/ACEsuit/mace.git)
The following packages are used in this project in a Python environment 3.12.3:

  - numpy 1.26.4
  - ase 3.24.0
  - mace.calculators (MACE package) 0.3.10
  - torch (PyTorch) 2.5.1
  - packmol (https://m3g.github.io/packmol/)


## Input files

The INPUTS folder contains the initial structures for the protein variants 1UAO and 5AWL and a water molecule:
  - 1uao.pdb
  - 1uao_neutral.pdb
  - 5awl_folded.pdb
  - 5awl_unfolded.pdb
  - water.pdb

The reference ab-inito data set was obtained from (https://figshare.com/articles/dataset/_strong_AIMD-Chig_exploring_the_conformational_space_of_166-atom_protein_strong_em_strong_Chignolin_strong_em_strong_with_strong_em_strong_ab_initio_strong_em_strong_molecular_dynamics_strong_/22786730)
    
# Simulation setup files
  - setup_simulation.py: Sets up the MD simulation.
  - run_simulation.py: Runs the MD simulation, using functions form setup_simulation.py.
  - process_simulations.py: Processes the trajectories of the simulations (calculates and plots radius of gyration vs time, energies vs time, temperature vs time)
  - process_rg_comparison.py: Creates a plot that compares the radius of gyrations of the simulations ran
  - process_AIMD.py: Processes the trajectory of the AIMD reference data (calculates and plots radius of gyration vs time)
  - visualize_chignolin_2D.py: A code for plotting the 2D visualizing of the chignolin protein
