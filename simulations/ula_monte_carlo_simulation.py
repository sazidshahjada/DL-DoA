import sys
sys.path.insert(0, '/home/sajid/Work/DL-DoA')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from simulations.signal_simulator import generate_composite_radio_signals
from tools.doa_algorithms import estimate_doa_music
from tools.covariance_matrix import array_signal_model, covariance_matrix
from tools.metrics import calculate_robust_rmse

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
        
        # Outer bar for SNR levels
        snr_pbar = tqdm(snr_range, desc=f"Sources: {n_src}")
        for snr in snr_pbar:
            trial_errors = []
            
            for _ in range(num_trials):
                # 1. Generate random DOAs with min separation to avoid merging
                true_doas = np.sort(np.random.uniform(-60, 60, n_src))
                
                # 2. Generate signals
                signals, _, _ = generate_composite_radio_signals(
                    num_sources=n_src,
                    num_samples=num_samples,
                    fs=2000,
                    snr_db=snr
                )
                
                # 3. Generate array received signal
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
            
            # 7. Final RMSE calculation for this SNR
            trial_errors_arr = np.array(trial_errors)
            snr_rmse = np.sqrt(np.mean(trial_errors_arr**2))

                
            rmse_per_snr.append(snr_rmse)
            snr_pbar.set_postfix({"Last RMSE": f"{snr_rmse:.2f}"})
            
        results[n_src] = np.array(rmse_per_snr)
        
    return results

if __name__ == "__main__":
    d = 0.5
    m_sensors = 36
    pos = np.arange(m_sensors) * d

    save_path = "/home/sajid/Work/DL-DoA/results/figures"
    
    snrs = np.arange(-10, 11, 2) 
    sources_to_test = [4, 6, 8, 10, 12, 14]

    simulation_results = monte_carlo_doa_simulation(
        sensor_positions=pos,
        snr_range=snrs,
        num_sources_list=sources_to_test,
        num_trials=200
    )

    plt.figure(figsize=(10, 6))
    for n_src, rmse_values in simulation_results.items():
        plt.plot(snrs, rmse_values, marker='o', linewidth=2, label=f'{n_src} Sources')

    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (Degrees)')
    plt.title('Monte Carlo Simulation: RMSE vs SNR (ULA)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.savefig(f"{save_path}/ula_monte_carlo_simulation_rmse_v_snr.png", dpi=300)
    plt.show()


    print("\nSimulation completed. Results saved to figures folder.")