import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context="paper")
sns.set_style(style="dark")
sys.path.insert(0, '/home/sajid/Work/DL-DoA')

from simulators.ula_monte_carlo_simulator import (
    doa_simulation_with_num_sources, 
    doa_simulation_with_snapshots
)
from tools.array_config import Coprime_Array


def _plot_comparison(ula_results, coprime_results, x_range, xlabel, title, param_name, save_path="results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes = [ax1, ax2]
    titles = ['Uniform Linear Array (ULA)', 'Coprime Array (CATARCS)']
    data = [ula_results, coprime_results]

    for i, ax in enumerate(axes):
        # Plot data
        for param_value, rmse_values in data[i].items():
            marker = 'o' if i == 0 else 's'
            ls = '-' if i == 0 else '--'
            ax.plot(x_range, rmse_values, marker=marker, linestyle=ls, 
                    label=f'{param_name}: {param_value}', linewidth=1.5)

        ax.grid(True, which='both', color='#666666', linestyle='-', alpha=0.6)
        
        # Axis Styling
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlabel(xlabel)
        if i == 0: ax.set_ylabel('RMSE (Degrees)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = title.replace(' ', '_').replace('.', '').replace(',', '') + ".png"
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.show()


def run_comparison_simulations(M, N, d, snr_range, num_sources_list, snapshot_list, n_trials=100, min_sep=5):
    """
    Configures and runs both ULA and Coprime Array simulations and compares them.
    """
    # 1. Configure Arrays
    # Coprime Array setup
    coprime_array = Coprime_Array(M, N, d=d, model_type="CATARCS")
    coprime_positions = coprime_array.virtual_positions
    num_physical_sensors = coprime_array.number_of_sensors
    coprime_dof = coprime_array.dof

    # ULA setup (Using the same number of physical sensors for a fair benchmark)
    ula_positions = np.arange(coprime_dof+1) * d

    print(f"--- Array Configuration ---")
    print(f"Coprime Array (M={M}, N={N}): {num_physical_sensors} physical sensors")
    print(f"Degrees of Freedom (DOF): {coprime_dof}")
    print(f"Uniform Linear Array: {coprime_dof+1} sensors (to match DOF of Coprime Array)")
    
    # 2. Simulation: RMSE vs SNR (Varying Sources)
    print("\n[1/2] Running Simulation: Performance vs. Number of Sources")
    
    print(" -> Simulating ULA...")
    ula_results_sources = doa_simulation_with_num_sources(
        sensor_positions=ula_positions, 
        snr_range=snr_range, 
        num_sources_list=num_sources_list, 
        num_trials=n_trials, 
        n_snapshot=500,
        min_sep=min_sep
    )

    print(" -> Simulating Coprime Array...")
    coprime_results_sources = doa_simulation_with_num_sources(
        sensor_positions=coprime_positions, 
        snr_range=snr_range, 
        num_sources_list=num_sources_list, 
        num_trials=n_trials, 
        n_snapshot=500,
        min_sep=min_sep
    )

    # 3. Simulation: RMSE vs SNR (Varying Snapshots)
    print("\n[2/2] Running Simulation: Performance vs. Number of Snapshots")
    target_source_count = 5
    
    print(" -> Simulating ULA...")
    ula_results_snapshots = doa_simulation_with_snapshots(
        sensor_positions=ula_positions, 
        snr_range=snr_range, 
        num_sources=target_source_count, 
        snapshot_range=snapshot_list, 
        num_trials=n_trials,
        min_sep=min_sep
    )

    print(" -> Simulating Coprime Array...")
    coprime_results_snapshots = doa_simulation_with_snapshots(
        sensor_positions=coprime_positions, 
        snr_range=snr_range, 
        num_sources=target_source_count, 
        snapshot_range=snapshot_list, 
        num_trials=n_trials,
        min_sep=min_sep
    )

    # 4. Plotting Comparisons
    save_dir = f"/home/sajid/Work/DL-DoA/results/figures/comparisons"
    
    print("\nGenerating Comparison Plots...")
    _plot_comparison(
        ula_results_sources, 
        coprime_results_sources, 
        x_range=snr_range, 
        xlabel='SNR (dB)',
        title=f"RMSE vs SNR Comparison (Sensors={num_physical_sensors})", 
        param_name="Sources",
        save_path=save_dir
    )
    
    _plot_comparison(
        ula_results_snapshots, 
        coprime_results_snapshots, 
        x_range=snr_range, 
        xlabel='SNR (dB)',
        title=f"RMSE vs Snapshots Comparison (Sources={target_source_count})", 
        param_name="Snapshots",
        save_path=save_dir
    )
    print(f"Saved plots to: {save_dir}")


if __name__ == "__main__":
    # Coprime array parameters
    M = 4
    N = 5
    d = 0.5

    # Simulation parameters
    min_sep = 5  # Minimum separation in degrees for sources
    SNR_VALS = [0, 5, 10, 15, 20]
    SOURCES = [2, 4, 6, 8, 10] 
    SNAPSHOTS = [100, 200, 500, 1000, 2000]
    n_trials = 100  # Number of Monte Carlo trials for averaging results

    # Run the comparison simulations
    run_comparison_simulations(
        M, N, d=d, snr_range=SNR_VALS, num_sources_list=SOURCES,
        snapshot_list=SNAPSHOTS, n_trials=n_trials, min_sep=min_sep
    )