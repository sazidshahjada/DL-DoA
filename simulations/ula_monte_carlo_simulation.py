import sys
sys.path.insert(0, '/home/sajid/Work/DL-DoA')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from simulations.signal_simulator import generate_composite_radio_signals
from tools.doa_algorithms import estimate_doa_music
from tools.covariance_matrix import array_signal_model, covariance_matrix
from tools.metrics import calculate_robust_rmse

def generate_separated_sources(n_src, min_sep=5):
    while True:
        angles = np.sort(np.random.uniform(-60, 60, n_src))
        if np.all(np.diff(angles) >= min_sep):
            return angles

def monte_carlo_doa_simulation(
    sensor_positions, 
    snr_range, 
    num_sources_list, 
    num_trials=50, 
    num_samples=1000
):
    results = {}
    
    for n_src in num_sources_list:
        rmse_per_snr = []
        print(f"\nSimulating {n_src} sources...")
        
        snr_pbar = tqdm(snr_range, desc=f"Sources: {n_src}")
        for snr in snr_pbar:
            trial_errors = []
            
            for _ in range(num_trials):
                # 1. Generate DOAs with guaranteed separation
                true_doas = generate_separated_sources(n_src, min_sep=5)
                
                # 2. Generate CLEAN signals
                signals, _, _ = generate_composite_radio_signals(
                    num_sources=n_src,
                    num_samples=num_samples,
                    fs=2000,
                    snr_db=None
                )
                
                # 3. Generate array received signal with target SNR
                X, _, _ = array_signal_model(
                    source_signals=signals,
                    sensor_positions=sensor_positions,
                    doa_angles=true_doas,
                    snr_db=snr,
                    n_snapshots=num_samples,
                    wavelength=1.0
                )
                
                # 4. Compute Covariance
                R = covariance_matrix(X)
                
                # 5. Estimate DOA
                est_angles, _, _ = estimate_doa_music(
                    Rxx=R,
                    num_sources=n_src,
                    sensor_positions=sensor_positions,
                    n_snapshots=num_samples,
                    scan_angles=np.linspace(-90, 90, 1801) 
                )
                
                # 6. Calculate Robust Error
                error = calculate_robust_rmse(true_doas, est_angles)
                trial_errors.append(error)
            
            # 7. Final RMSE calculation
            trial_errors_arr = np.array(trial_errors)
            snr_rmse = np.sqrt(np.mean(trial_errors_arr**2))
            
            rmse_per_snr.append(snr_rmse)
            snr_pbar.set_postfix({"RMSE": f"{snr_rmse:.3f}"})
            
        results[n_src] = np.array(rmse_per_snr)
        
    return results

if __name__ == "__main__":
    # Standard ULA Parameters
    d = 0.5
    m_sensors = 46
    pos = np.arange(m_sensors) * d

    save_path = "/home/sajid/Work/DL-DoA/results/figures"
    
    # Testing a range of SNRs to see the "Waterline" effect
    snrs = np.arange(0, 11, 2) 
    sources_to_test = [2, 4, 6, 8, 10, 12]

    simulation_results = monte_carlo_doa_simulation(
        sensor_positions=pos,
        snr_range=snrs,
        num_sources_list=sources_to_test,
        num_trials=100
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    for n_src, rmse_values in simulation_results.items():
        plt.plot(snrs, rmse_values, marker='o', label=f'{n_src} Sources')

    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (Degrees) - Log Scale')
    plt.title('Monte Carlo Simulation on ULA: RMSE vs SNR')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(f"{save_path}/ula_monte_carlo_rmse_vs_snr_0-10.png", dpi=300)
    plt.show()