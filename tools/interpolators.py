import numpy as np
from scipy.linalg import solve

def build_Av_vectorized(V, r, m, p, obs_map, i_arr, j_arr):
    """Constructs matrix Av such that Av @ vec(U) = y_xx (Vectorized)"""
    Av = np.zeros((p, m * r), dtype=complex)
    k = np.arange(p)
    for l in range(r):
        col_idx = i_arr + m * l
        Av[k, col_idx] = V[l, j_arr]
    return Av


def build_Au_vectorized(U, r, n, p, obs_map, i_arr, j_arr):
    """Constructs matrix Au such that Au @ vec(V) = y_xx (Vectorized)"""
    Au = np.zeros((p, r * n), dtype=complex)
    k = np.arange(p)
    for l in range(r):
        col_idx = l + r * j_arr
        Au[k, col_idx] = U[i_arr, l]
    return Au


def enforce_toeplitz_hermitian(R):
    """Averages diagonals to enforce Toeplitz structure and ensures Hermitian symmetry."""
    N = R.shape[0]
    R_toeplitz = np.zeros_like(R, dtype=complex)
    
    for d in range(-N + 1, N):
        diag_mean = np.mean(np.diag(R, k=d))
        row_indices, col_indices = np.diag_indices(N)
        if d >= 0:
            R_toeplitz[row_indices[:N-d], col_indices[d:]] = diag_mean
        else:
            R_toeplitz[row_indices[-d:], col_indices[:N+d]] = diag_mean
            
    return (R_toeplitz + R_toeplitz.conj().T) / 2


def solve_ipf(R_obs, virtual_indices, target_n, max_rank=5, max_iter=50, lmbda=1e-4, 
              make_toeplitz=True, hard_imputation=False):
    """
    Rank-Incremented Iterative Power Factorization for Matrix Completion.
    
    Args:
        R_obs: Your existing sparse covariance matrix with holes as 0.
        virtual_indices: Indices of the physical sensors in the grid.
        target_n: Total elements in the full grid (dof + 1).
        max_rank: Maximum rank to increment to.
        lmbda: Regularization parameter.
        make_toeplitz: Enforce physical Toeplitz/Hermitian constraints.
        hard_imputation: Force original observed values to remain unchanged.
    """
    obs_map = [(i, j) for i in virtual_indices for j in virtual_indices]
    
    # Extract only the known elements to align perfectly with matrix rows
    y_xx = np.array([R_obs[i, j] for i, j in obs_map])
    p = len(y_xx)
    y_norm = np.linalg.norm(y_xx)
    
    # Pre-extract coordinate arrays for fast vectorized mapping
    i_arr = np.array([i for i, j in obs_map])
    j_arr = np.array([j for i, j in obs_map])
    
    U = None
    V = None

    for r in range(1, max_rank + 1):
        # 1. Initialize or Increment Rank
        if U is None:
            U = np.random.randn(target_n, r) + 1j * np.random.randn(target_n, r)
            V = np.random.randn(r, target_n) + 1j * np.random.randn(r, target_n)
        else:
            u_new = (np.random.randn(target_n, 1) + 1j * np.random.randn(target_n, 1)) * 0.1
            v_new = (np.random.randn(1, target_n) + 1j * np.random.randn(1, target_n)) * 0.1
            U = np.hstack([U, u_new])
            V = np.vstack([V, v_new])

        stabilized = False
        
        # 2. Alternating Optimization
        for q in range(max_iter):
            # Update U
            Av = build_Av_vectorized(V, r, target_n, p, obs_map, i_arr, j_arr)
            lhs_v = Av.conj().T @ Av + lmbda * np.eye(Av.shape[1])
            rhs_v = Av.conj().T @ y_xx
            vec_U = solve(lhs_v, rhs_v, assume_a='her')
            U = vec_U.reshape((r, target_n)).T

            # Update V
            Au = build_Au_vectorized(U, r, target_n, p, obs_map, i_arr, j_arr)
            lhs_u = Au.conj().T @ Au + lmbda * np.eye(Au.shape[1])
            rhs_u = Au.conj().T @ y_xx
            vec_V = solve(lhs_u, rhs_u, assume_a='her')
            V = vec_V.reshape((target_n, r)).T

            # 3. Check for stabilization
            R_rec = U @ V
            y_hat = np.array([R_rec[i, j] for (i, j) in obs_map])
            
            error = np.linalg.norm(y_hat - y_xx) / (y_norm + 1e-12)
            if error < 1e-5:
                stabilized = True
                break
                
        if stabilized:
            break
                
    R_final = U @ V
    
    # 4. Enforce Toeplitz/Hermitian Constraints
    if make_toeplitz:
        R_final = enforce_toeplitz_hermitian(R_final)
        
    # 5. Optional Hard Imputation (Force original values to stay the same)
    if hard_imputation:
        for i, j in obs_map:
            R_final[i, j] = R_obs[i, j]
            
    return R_final