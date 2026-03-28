import numpy as np
from scipy.linalg import solve


def build_Av(V, r, m, p, obs_map):
    """Constructs matrix Av such that Av @ vec(U) = y_xx"""
    Av = np.zeros((p, m * r), dtype=complex)
    for k, (i, j) in enumerate(obs_map):
        for l in range(r):
            col_idx = i + m * l
            Av[k, col_idx] = V[l, j]
    return Av


def build_Au(U, r, n, p, obs_map):
    """Constructs matrix Au such that Au @ vec(V) = y_xx"""
    Au = np.zeros((p, r * n), dtype=complex)
    for k, (i, j) in enumerate(obs_map):
        for l in range(r):
            col_idx = l + r * j
            Au[k, col_idx] = U[i, l]
    return Au


def solve_ipf(R_obs, virtual_indices, target_n, max_rank=5, max_iter=50, lmbda=1e-4):
    """
    Rank-Incremented Iterative Power Factorization (Standalone Function).
    
    Args:
        R_obs: Observed sparse covariance matrix.
        virtual_indices: Indices of the physical sensors.
        target_n: Total elements in the full ULA (2MN).
        max_rank: Maximum rank to increment to.
        lmbda: Regularization parameter to prevent numerical explosion.
    """
    y_xx = R_obs.flatten()
    p = len(y_xx)
    y_norm = np.linalg.norm(y_xx)
    obs_map = [(i, j) for i in virtual_indices for j in virtual_indices]
    
    U = None
    V = None

    for r in range(1, max_rank + 1):
        # 1. Initialize or Increment Rank
        if U is None:
            U = np.random.randn(target_n, r) + 1j * np.random.randn(target_n, r)
            V = np.random.randn(r, target_n) + 1j * np.random.randn(r, target_n)
        else:
            # Rank-Incremented initialization
            u_new = (np.random.randn(target_n, 1) + 1j * np.random.randn(target_n, 1)) * 0.1
            v_new = (np.random.randn(1, target_n) + 1j * np.random.randn(1, target_n)) * 0.1
            U = np.hstack([U, u_new])
            V = np.vstack([V, v_new])

        # 2. Alternating Optimization
        for q in range(max_iter):
            # Update U
            Av = build_Av(V, r, target_n, p, obs_map)
            lhs_v = Av.conj().T @ Av + lmbda * np.eye(Av.shape[1])
            rhs_v = Av.conj().T @ y_xx
            vec_U = solve(lhs_v, rhs_v, assume_a='her')
            U = vec_U.reshape((r, target_n)).T

            # Update V
            Au = build_Au(U, r, target_n, p, obs_map)
            lhs_u = Au.conj().T @ Au + lmbda * np.eye(Au.shape[1])
            rhs_u = Au.conj().T @ y_xx
            vec_V = solve(lhs_u, rhs_u, assume_a='her')
            V = vec_V.reshape((target_n, r)).T

            # 3. Check for stabilization
            R_rec = U @ V
            y_hat = np.array([R_rec[i, j] for (i, j) in obs_map])
            if (np.linalg.norm(y_hat - y_xx) / y_norm) < 1e-5:
                break
                
    return U @ V