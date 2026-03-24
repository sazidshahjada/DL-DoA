"""
Covariance Matrix Computation for Sparse Arrays
This module provides functions to compute the covariance matrix of array signals, including handling of sparse arrays with holes in the virtual array.
"""

import numpy as np




def _compute_steering_matrix(
    sensor_positions,
    angles_deg,
    wavelength=1.0
):
    """
    Compute the array steering matrix.

    This function generates the steering matrix for a set of
    sensor positions and source directions.

    Parameters
    ----------
    sensor_positions : array-like
        Positions of sensors in units of wavelength (or meters if wavelength specified).

    angles_deg : array-like
        Directions of arrival (DOAs) in degrees.

    wavelength : float, optional
        Signal wavelength. Default is 1 (normalized).

    Returns
    -------
    A : ndarray
        Steering matrix of shape (num_sensors, num_angles)

    Notes
    -----
    The steering vector is defined as:

        a(theta) = exp(-j 2π p sin(theta) / λ)

    where:
        p = sensor position
        θ = DOA angle
        λ = wavelength
    """

    sensor_positions = np.array(sensor_positions)
    angles_deg = np.array(angles_deg)

    # Convert angles to radians
    angles_rad = np.deg2rad(angles_deg)

    # Compute phase shifts
    phase_shifts = (
        -1j
        * 2
        * np.pi
        * np.outer(sensor_positions, np.sin(angles_rad))
        / wavelength
    )

    # Steering matrix
    A = np.exp(phase_shifts)

    return A




def array_signal_model(
    source_signals,
    sensor_positions,
    doa_angles,
    snr_db=None,
    wavelength=1.0,
    n_snapshots=100,  # Default integer
    random_seed=None
):
    """
    Generate the array received signal X from source signals.

    Parameters
    ----------
    source_signals : ndarray
        Source signal matrix of shape (num_sources, num_samples).
    sensor_positions : array-like
        Positions of sensors in units of wavelength (or meters if wavelength specified).
    doa_angles : array-like
        Directions of arrival (DOAs) in degrees.
    snr_db : float, optional
        Desired signal-to-noise ratio in dB. If None, no noise is added.
    wavelength : float, optional
        Signal wavelength. Default is 1 (normalized).
    n_snapshots : int, optional
        Number of snapshots (samples) to use from the source signals. Default is 100.
    random_seed : int, optional
        Seed for random number generator to ensure reproducibility. Default is None.

    Returns
    -------
    X : ndarray
        Received signal matrix at the array (num_sensors, n_snapshots).
    A : ndarray
        Steering matrix of shape (num_sensors, num_sources).
    N : ndarray
        Noise matrix of shape (num_sensors, n_snapshots) if snr_db is specified, otherwise zeros.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Force integer conversion and ensure it is at least 1
    n_snapshots = int(max(1, n_snapshots))
    
    # Check against available source data
    available_samples = source_signals.shape[1]
    if n_snapshots > available_samples:
        print(f"Warning: Requested {n_snapshots} snapshots, but only {available_samples} exist. Clipping to max.")
        n_snapshots = available_samples

    # Slice source signals to the specific snapshot count
    S = source_signals[:, :n_snapshots]

    # Compute steering matrix A (num_sensors, num_sources)
    A = _compute_steering_matrix(sensor_positions, doa_angles, wavelength=wavelength)

    # X = A * S
    X_signal = A @ S

    # Initialize noise
    N = np.zeros_like(X_signal, dtype=complex)

    if snr_db is not None:
        signal_power = np.mean(np.abs(X_signal)**2)
        noise_power = signal_power / (10**(snr_db / 10))

        # Complex Gaussian noise
        N = np.sqrt(noise_power/2) * (
            np.random.randn(*X_signal.shape) + 1j * np.random.randn(*X_signal.shape)
        )

    # Received signal with noise
    X = X_signal + N

    return X, A, N




def covariance_matrix(X):
    """
    Compute the sample covariance matrix of the array signals.

    Parameters
    ----------
    X : ndarray
        Received signal matrix at the array (num_sensors, num_samples).

    Returns
    -------
    R : ndarray
        Sample covariance matrix of shape (num_sensors, num_sensors).
    """

    n_snapshots = X.shape[1]

    # Sample covariance matrix
    R = (X @ X.conj().T) / n_snapshots

    return R




def sparse_covariance_matrix(X, virtual_positions, hole_positions, dof):
    """
    Compute the covariance matrix for a sparse array, filling holes with zeros.

    Parameters
    ----------
    X : ndarray
        Received array signal, shape (num_sensors, num_samples)
    virtual_positions : array-like
        Full virtual array positions
    hole_positions : array-like
        Positions in virtual array with no physical sensor (holes)
    dof : int
        Degrees of freedom (The maximum virtual position)

    Returns
    -------
    R_full : ndarray
        Full covariance matrix including zeros at hole positions
    """
    virtual_positions = np.array(virtual_positions)
    hole_positions = np.array(hole_positions)

    # Compute array covariance
    R = covariance_matrix(X)

    # Initialize full virtual covariance matrix
    R_full = np.zeros((dof + 1, dof + 1), dtype=complex)

    for i, pos_i in enumerate(virtual_positions):
        for j, pos_j in enumerate(virtual_positions):
            if pos_i in hole_positions or pos_j in hole_positions:
                R_full[pos_i, pos_j] = 0  # Hole position
            else:
                R_full[pos_i, pos_j] = R[i, j]  # Fill with computed covariance

    return R_full





# Example usage
if __name__ == "__main__":
    from array_config import Coprime_Array
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define parameters
    doa_angles = [30, 60]  # Two sources at 30 and 60 degrees
    num_samples = 1000
    snr_db = 20

    array = Coprime_Array(M=3, N=4, d=1, model_type='GoldenRatio')
    num_sensors = array.number_of_sensors
    sensor_positions = array.sensor_positions
    virtual_positions = array.virtual_positions
    hole_positions = array.hole_positions
    dof = array.dof


    print("Virtual Array Positions:", virtual_positions)
    print("Hole Positions:", hole_positions)
    print("Degrees of Freedom (DOF):", dof)

    # Generate random source signals (2 sources)
    source_signals = np.random.randn(2, num_samples) + 1j * np.random.randn(2, num_samples)

    # Generate array signals
    X, A, N = array_signal_model(
        source_signals,
        virtual_positions,
        doa_angles,
        snr_db=snr_db,
        wavelength=1.0,
        n_snapshots=1000,
        random_seed=42
    )

    # Generate covariance matrix
    R_full = sparse_covariance_matrix(X, virtual_positions, hole_positions, dof)
    print(f"Full covariance matrix shape: {R_full.shape}")