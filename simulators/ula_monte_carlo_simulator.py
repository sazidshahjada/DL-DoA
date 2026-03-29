import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/sajid/Work/DL-DoA')

from simulators.signal_simulator import generate_composite_radio_signals
from tools.doa_algorithms import estimate_doa_music
from tools.covariance_matrix import array_signal_model, covariance_matrix
from tools.metrics import calculate_robust_rmse


def generate_separated_sources(n_src, min_sep=5):
    """
    Generates random DOA angles with a guaranteed minimum angular separation.
    """
    while True:
        angles = np.sort(np.random.uniform(-60, 60, n_src))
        if n_src < 2 or np.all(np.diff(angles) >= min_sep):
            return angles


def doa_simulation_with_num_sources(
    sensor_positions, 
    snr_range, 
    num_sources_list, 
    num_trials=50, 
    n_snapshot=1000,
    min_sep=5
):
    """
    Monte Carlo Simulation: Performance vs. Number of Sources.
    """
    results = {}
    
    for n_src in num_sources_list:
        rmse_per_snr = []
        print(f"\n[Sources Simulation] Testing {n_src} sources...")
        
        snr_pbar = tqdm(snr_range, desc=f"Sources: {n_src}")
        for snr in snr_pbar:
            trial_errors = []
            for _ in range(num_trials):
                true_doas = generate_separated_sources(n_src, min_sep=min_sep)
                signals, _, _ = generate_composite_radio_signals(
                    num_sources=n_src, num_samples=n_snapshot, fs=2000, snr_db=None
                )
                X, _, _ = array_signal_model(
                    source_signals=signals, sensor_positions=sensor_positions,
                    doa_angles=true_doas, snr_db=snr, n_snapshots=n_snapshot, wavelength=1.0
                )
                R = covariance_matrix(X)
                est_angles, _, _ = estimate_doa_music(
                    Rxx=R, num_sources=n_src, sensor_positions=sensor_positions,
                    n_snapshots=n_snapshot, scan_angles=np.linspace(-90, 90, 1801) 
                )
                trial_errors.append(calculate_robust_rmse(true_doas, est_angles))
            
            snr_rmse = np.sqrt(np.mean(np.array(trial_errors)**2))
            rmse_per_snr.append(snr_rmse)
            snr_pbar.set_postfix({"RMSE": f"{snr_rmse:.3f}"})
            
        results[n_src] = np.array(rmse_per_snr)
    return results


def doa_simulation_with_snapshots(
    sensor_positions, 
    snr_range, 
    num_sources, 
    snapshot_range, 
    num_trials=50, 
    min_sep=5
):
    """
    Monte Carlo Simulation: Performance vs. Number of Snapshots.
    """
    results = {}
    
    for n_snap in snapshot_range:
        rmse_per_snr = []
        print(f"\n[Snapshots Simulation] Testing {n_snap} snapshots...")
        
        snr_pbar = tqdm(snr_range, desc=f"Snapshots: {n_snap}")
        for snr in snr_pbar:
            trial_errors = []
            for _ in range(num_trials):
                true_doas = generate_separated_sources(num_sources, min_sep=min_sep)
                signals, _, _ = generate_composite_radio_signals(
                    num_sources=num_sources, num_samples=n_snap, fs=2000, snr_db=None
                )
                X, _, _ = array_signal_model(
                    source_signals=signals, sensor_positions=sensor_positions,
                    doa_angles=true_doas, snr_db=snr, n_snapshots=n_snap, wavelength=1.0
                )
                R = covariance_matrix(X)
                est_angles, _, _ = estimate_doa_music(
                    Rxx=R, num_sources=num_sources, sensor_positions=sensor_positions,
                    n_snapshots=n_snap, scan_angles=np.linspace(-90, 90, 1801) 
                )
                trial_errors.append(calculate_robust_rmse(true_doas, est_angles))
            
            snr_rmse = np.sqrt(np.mean(np.array(trial_errors)**2))
            rmse_per_snr.append(snr_rmse)
            snr_pbar.set_postfix({"RMSE": f"{snr_rmse:.3f}"})
            
        results[n_snap] = np.array(rmse_per_snr)
    return results



if __name__ == "__main__":
    physical_sensors = np.array([0, 1, 4, 7, 9, 12, 15, 20]) 
    snrs = np.arange(-10, 21, 5)
    
    # 1. Run Source Simulation
    res_sources = doa_simulation_with_num_sources(
        sensor_positions=physical_sensors,
        snr_range=snrs,
        num_sources_list=[1, 2, 3],
        min_sep=5
    )
    print("=" * 50)

    # 3. Run Snapshot Simulation
    res_snapshots = doa_simulation_with_snapshots(
        sensor_positions=physical_sensors,
        snr_range=snrs,
        num_sources=2,
        snapshot_range=[100, 500, 1000, 2000],
        min_sep=5
    )
    print("=" * 50)


    print("\nSimulations Complete.")