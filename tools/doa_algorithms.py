import numpy as np
from scipy.signal import find_peaks



def _find_peaks(P, scan_angles, num_sources):
    P_db = 10 * np.log10(P / np.max(P))
    
    peaks, properties = find_peaks(
        P_db, 
        height=-30, 
        distance=20, 
        prominence=1.5
    )
    
    if len(peaks) > num_sources:
        prominences = properties['prominences']
        highest_peaks_idx = np.argsort(prominences)[-num_sources:]
        peaks = peaks[highest_peaks_idx]
    
    estimated_angles = np.sort(scan_angles[peaks])
    return estimated_angles



def estimate_doa_music(Rxx, num_sources, sensor_positions, n_snapshots, wavelength=1.0, scan_angles=np.arange(-90, 90.1, 0.1)):
    """
    MUSIC DOA Estimation with snapshot validation.
    
    Parameters:
    -----------
    Rxx : ndarray
        Covariance matrix (M x M).
    num_sources : int
        Number of signals.
    sensor_positions : array-like
        Sensor locations.
    n_snapshots : int
        Number of time samples used to create Rxx (Must be >= 1).
    """
    
    # 1. Enforce n_snapshots as a valid positive integer
    n_snapshots = int(max(1, n_snapshots))
    
    # 2. Basic dimensions
    M = Rxx.shape[0]
    sensor_positions = np.array(sensor_positions).reshape(-1, 1)
    
    # 3. Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(Rxx)
    
    # Sort eigenvalues and vectors in descending order
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 4. Subspace Separation
    if n_snapshots < num_sources:
        print(f"Warning: n_snapshots ({n_snapshots}) is less than num_sources ({num_sources}). "
              "Results may be highly inaccurate.")
              
    En = eigvecs[:, num_sources:] # Noise subspace
    
    # 5. Compute MUSIC Spectrum
    angles_rad = np.deg2rad(scan_angles)
    
    # Vectorized steering vector computation across all scan angles
    A_scan = np.exp(-1j * 2 * np.pi * sensor_positions * np.sin(angles_rad) / wavelength)
    denominator = np.sum(np.abs(En.conj().T @ A_scan)**2, axis=0)
    
    # Prevent division by zero
    denominator[denominator == 0] = 1e-12
    P_music = 1.0 / denominator

    # 6. Peak Finding (passing validated integer to helper)
    estimated_angles = _find_peaks(P_music, scan_angles, num_sources)
    
    return estimated_angles, P_music, scan_angles




def estimate_doa_capon(Rxx, num_sources, sensor_positions, wavelength=1.0, 
                      scan_angles=np.arange(-90, 90.1, 0.1), diagonal_loading=1e-6):
    """
    Capon (MVDR) DOA Estimation.
    
    Parameters:
    -----------
    Rxx : ndarray
        Covariance matrix (M x M).
    num_sources : int
        Number of signals to find.
    sensor_positions : array-like
        Sensor locations (ULA or Sparse).
    wavelength : float
        Wavelength of the signal.
    scan_angles : ndarray
        The angular range to scan.
    diagonal_loading : float
        Regularization factor to ensure Rxx is invertible.
    """
    
    M = Rxx.shape[0]
    sensor_positions = np.array(sensor_positions).reshape(-1, 1)
    
    # 1. Robust Matrix Inversion with Diagonal Loading
    R_inv = np.linalg.inv(Rxx + diagonal_loading * np.eye(M))
    
    # 2. Compute Steering Vectors for all scan angles
    angles_rad = np.deg2rad(scan_angles)
    A_scan = np.exp(-1j * 2 * np.pi * sensor_positions * np.sin(angles_rad) / wavelength)
    
    # 3. Compute Capon Power Spectrum
    term1 = R_inv @ A_scan
    # denominator = sum(conjugate(A_scan) * term1) -> shape (num_scan_angles,)
    denominator = np.real(np.sum(np.conj(A_scan) * term1, axis=0))
    
    # Prevent division by zero
    denominator[denominator <= 0] = 1e-12
    P_capon = 1.0 / denominator

    # 4. Peak Finding (using your existing helper)
    estimated_angles = _find_peaks(P_capon, scan_angles, num_sources)
    
    return estimated_angles, P_capon, scan_angles
