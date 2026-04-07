import sys

sys.path.insert(0, '/home/iot/Sajid/Repositories/DL-DoA')

import os
import numpy as np
from tqdm import tqdm
from tools.array_config import Coprime_Array
from simulators.signal_simulator import generate_composite_radio_signals
from tools.covariance_matrix import array_signal_model, covariance_matrix, sparse_covariance_matrix


def generate_data(array_config: Coprime_Array, sample_count: int, save_path: str):
    print(f"Array Configuration: M={array_config.M}, N={array_config.N}, d={array_config.d}, DOF={array_config.dof}")
    for i in tqdm(range(sample_count), desc="Generating Data Samples"):
        n = np.random.randint(1, 15)
        angles = np.arange(-90, 90, 1)
        source_angles = np.random.choice(angles, size=n, replace=False)
        source_angles = np.sort(np.unique(source_angles))
        num_sources = len(source_angles)
        snr = np.random.uniform(-10, 30)
        n_snapshots = np.random.choice([100, 200, 500, 1000, 1500, 2000])

        signals, _, _ = generate_composite_radio_signals(
            num_sources=num_sources,
            num_samples=n_snapshots,
            fs=4000,
        )

        virtual_positions = array_config.virtual_positions
        hole_positions = array_config.hole_positions
        ula_positions = array_config.uniform_positions
        dof = array_config.dof
        wavelength = 2

        X_obs, _, _ = array_signal_model(
            source_signals=signals,
            sensor_positions=virtual_positions,
            doa_angles=source_angles,
            wavelength=wavelength,
            snr_db=snr,
            n_snapshots=n_snapshots,
        )

        X_true, _, _ = array_signal_model(
            source_signals=signals,
            sensor_positions=ula_positions,
            doa_angles=source_angles,
            wavelength=wavelength,
            snr_db=snr,
            n_snapshots=n_snapshots,
        )

        R_obs = sparse_covariance_matrix(X_obs, virtual_positions, hole_positions, dof)
        R_true = covariance_matrix(X_true)

        os.makedirs(save_path, exist_ok=True)
        obs_path = f"{save_path}/obs"
        os.makedirs(obs_path, exist_ok=True)
        true_path = f"{save_path}/true"
        os.makedirs(true_path, exist_ok=True)

        np.savez_compressed(f"{obs_path}/sample_{i}.npz", R_obs=R_obs)
        np.savez_compressed(f"{true_path}/sample_{i}.npz", R_true=R_true)           


if __name__ == "__main__":
    M = 6
    N = 7
    d = 1
    array_config = Coprime_Array(M=M, N=N, d=d)
    sample_count = 100000
    dof = array_config.dof
    save_path = f"data/CATARACS_M{M}_N{N}_DOF{dof}"

    generate_data(array_config, sample_count, save_path)
    print(f"Data generation complete. Samples saved to: {save_path}")


