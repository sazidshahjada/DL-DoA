import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context="paper")
sns.set_style(style="dark")
sys.path.insert(0, '/home/sajid/Work/DL-DoA')

from simulators.coprime_array_monte_carlo_simulator import (
    doa_simulation_with_num_sources, 
    doa_simulation_with_snapshots
)
from tools.array_config import Coprime_Array


def _plot_results(simulation_results, snr_range, title, label_prefix, save_path="results"):
    """
    Plots RMSE vs SNR for different simulation parameters.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.figure(figsize=(10, 6))
    
    for param_value, rmse_values in simulation_results.items():
        plt.plot(snr_range, rmse_values, marker='o', label=f'{label_prefix}: {param_value}')
    
    # plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (Degrees)')
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    filename = title.replace(' ', '_').replace('.', '') + ".png"
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.show()


def run_coprime_simulations(M, N, d, snr_range, num_sources_list, snapshot_list):
    """
    Configures and runs Coprime Array-specific simulations.
    """
    array = Coprime_Array(M, N, d=d, model_type="CATARCS")
    num_sensors = array.number_of_sensors
    print(f"Running Coprime Array Simulation with {num_sensors} sensors.")

    # 2. Run Simulation: Performance vs. Number of Sources
    print("\nRun Simulation: Performance vs. Number of Sources")
    results_sources = doa_simulation_with_num_sources(
        array_config=array,
        snr_range=snr_range, 
        num_sources_list=num_sources_list, 
        num_trials=100, 
        n_snapshot=500,
        min_sep=5
    )

    # 3. Run Simulation: Performance vs. Number of Snapshots
    print("\nRun Simulation: Performance vs. Number of Snapshots")
    target_source_count = 5
    results_snapshots = doa_simulation_with_snapshots(
        array_config=array,
        snr_range=snr_range, 
        num_sources=target_source_count, 
        snapshot_range=snapshot_list, 
        num_trials=100,
        min_sep=5
    )

    # 4. Plotting
    save_dir = f"/home/sajid/Work/DL-DoA/results/figures"
    _plot_results(
        results_sources, 
        snr_range, 
        title=f"Coprime Array DOA RMSE vs. Number of Sources (M={M}, N={N})", 
        label_prefix="Sources",
        save_path=save_dir
    )
    
    _plot_results(
        results_snapshots, 
        snr_range, 
        title=f"Coprime Array DOA RMSE vs. Number of Snapshots (Sources={target_source_count}, M={M}, N={N})", 
        label_prefix="Snapshots",
        save_path=save_dir
    )


if __name__ == "__main__":
    # Coprime Array Configuration
    M = 4
    N = 5
    d = 1

    # Parameters for simulations
    SNR_VALS = [0, 5, 10, 15, 20]
    SOURCES = [2, 4, 6, 8, 10]
    SNAPSHOTS = [200, 500, 1000, 2000, 5000]

    # Run the simulations
    run_coprime_simulations(M, N, d=d, snr_range=SNR_VALS, num_sources_list=SOURCES, snapshot_list=SNAPSHOTS)