import sys
sys.path.insert(0, '/home/sajid/Work/DL-DoA')

import os
import numpy as np
from tqdm import tqdm
from tools.array_config import Coprime_Array
from simulators.signal_simulator import generate_composite_radio_signals
from tools.covariance_matrix import array_signal_model, covariance_matrix


def generate_data(array_config: Coprime_Array, sample_count: int, save_path: str):
    for i in tqdm(range(sample_count), desc="Generating Data"):
        num_sources = np.random.randint(1, 11)
        angles = np.arange(-90, 90, 1)
        source_angles = np.random.choice(angles, size=num_sources, replace=False)
        snr = np.random.uniform(0, 30)
        n_snapshots = np.random.randint(100, 1000)

        signals, _, _ = generate_composite_radio_signals(
            num_sources=num_sources,
            num_samples=n_snapshots,
            fs=n_snapshots*2,
        )

        virtual_positions = array_config.virtual_positions
        ula_positions = array_config.uniform_positions

        X_obs, _, _ = array_signal_model(
            source_signals=signals,
            sensor_positions=virtual_positions,
            doa_angles=source_angles,
            wavelength=1.0,
            snr_db=snr,
            n_snapshots=n_snapshots,
        )

        X_true, _, _ = array_signal_model(
            source_signals=signals,
            sensor_positions=ula_positions,
            doa_angles=source_angles,
            wavelength=1.0,
            snr_db=snr,
            n_snapshots=n_snapshots,
        )

        R_obs = covariance_matrix(X_obs)
        R_true = covariance_matrix(X_true)

        os.makedirs(save_path, exist_ok=True)

        obs_path = f"{save_path}/obs"
        os.makedirs(obs_path, exist_ok=True)
        true_path = f"{save_path}/true"
        os.makedirs(true_path, exist_ok=True)

        np.savez_compressed(f"{obs_path}/sample_{i}.npz", R_obs=R_obs)
        np.savez_compressed(f"{true_path}/sample_{i}.npz", R_true=R_true)           


if __name__ == "__main__":
    M = 4
    N = 5
    d = 0.5
    array_config = Coprime_Array(M=M, N=N, d=d)
    sample_count = 1000
    save_path = "/home/sajid/Garbage/DoA_Data"

    generate_data(array_config, sample_count, save_path)
    print(f"Data generation complete. Samples saved to: {save_path}")


