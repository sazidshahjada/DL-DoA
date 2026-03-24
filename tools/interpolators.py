import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline
from scipy.linalg import toeplitz

class CovarianceMatrixInterpolator:
    def __init__(self, R_obs, virtual_indices, target_dof=45):
        r"""
        Initialize the interpolator for sparse array covariance reconstruction.

        Args:
            R_obs (np.ndarray, Complex): The (N, N) observed covariance matrix from sparse physical sensors.
            virtual_indices (np.ndarray): The specific spatial indices/positions of the virtual sensors.
            target_dof (int): The maximum spatial lag to reconstruct. The output matrix 
                              will be of shape (target_dof + 1, target_dof + 1).
        """

        self.R_obs = np.asarray(R_obs, dtype=complex)
        self.indices = np.asarray(virtual_indices, dtype=int)
        self.target_n = target_dof + 1 
        self.in_n = R_obs.shape[0]
        
        # Spectral grids for SBL, IAA, and Co-LASSO
        self.angles = np.linspace(-90, 90, 180)
        self.A_obs = np.exp(1j * np.pi * self.indices[:, None] * np.sin(np.radians(self.angles)))
        self.A_full = np.exp(1j * np.pi * np.arange(self.target_n)[:, None] * np.sin(np.radians(self.angles)))


    def solve_toeplitz_nnm(self):
        r"""
        Toeplitz Nuclear Norm Minimization (NNM).
        
        Reconstructs the full covariance matrix by minimizing the nuclear norm (a proxy for rank) 
        subject to PSD and Toeplitz constraints. This method is highly effective when the 
        number of sources is unknown as it naturally seeks the lowest-rank solution that fits 
        the observed data.
        """

        R_filled = cp.Variable((self.target_n, self.target_n), complex=True)
        constraints = [R_filled >> 0]
        
        for i, idx_i in enumerate(self.indices):
            for j, idx_j in enumerate(self.indices):
                constraints += [R_filled[idx_i, idx_j] == self.R_obs[i, j]]
        
        for k in range(self.target_n - 1):
            diag = cp.diag(R_filled, k=k)
            constraints += [diag[1:] == diag[:-1]]
            
        prob = cp.Problem(cp.Minimize(cp.norm(R_filled, "nuc")), constraints)
        prob.solve(solver=cp.SCS)
        return R_filled.value


    def solve_redundancy_avg(self):
        r"""
        Redundancy Averaging.
        
        A non-parametric baseline method that maps observed cross-correlations to their 
        corresponding spatial lags in the co-array. Multiple observations of the same 
        lag are averaged to reduce noise variance. Missing lags are left as zero.
        """

        lags = {}
        for i, idx_i in enumerate(self.indices):
            for j, idx_j in enumerate(self.indices):
                lag = idx_i - idx_j
                if lag not in lags: lags[lag] = []
                lags[lag].append(self.R_obs[i, j])
        
        r_vec = np.zeros(self.target_n, dtype=complex)
        for lag, values in lags.items():
            if 0 <= lag < self.target_n:
                r_vec[lag] = np.mean(values)
        return toeplitz(r_vec.conj(), r_vec)


    def solve_power_factorization(self, max_iter=100, energy_threshold=0.98):
        r"""
        Adaptive Iterative Power Factorization.
        
        An iterative algorithm that alternates between enforcing the observed sensor data 
        and projecting the matrix onto a low-rank subspace. Since the source count is unknown, 
        the rank is dynamically estimated in each iteration by retaining enough singular 
        values to meet the specified cumulative energy threshold.
        """

        R_full = np.zeros((self.target_n, self.target_n), dtype=complex)
        mask = np.zeros((self.target_n, self.target_n), dtype=bool)
        
        for i, idx_i in enumerate(self.indices):
            for j, idx_j in enumerate(self.indices):
                R_full[idx_i, idx_j] = self.R_obs[i, j]
                mask[idx_i, idx_j] = True
        
        for _ in range(max_iter):
            u, s, vh = np.linalg.svd(R_full)
            cumulative_power = np.cumsum(s) / np.sum(s)
            k_est = np.where(cumulative_power >= energy_threshold)[0][0] + 1
            
            R_low = u[:, :k_est] @ np.diag(s[:k_est]) @ vh[:k_est, :]
            R_full[~mask] = R_low[~mask]
            
        return R_full


    def solve_iaa(self, iters=10):
        r"""
        Iterative Adaptive Approach (IAA).
        
        A non-parametric, robust spectral estimation algorithm. It iteratively estimates 
        the signal power at every potential angle on a grid without requiring prior 
        knowledge of the source count. The final covariance is synthesized from the 
        resulting spatial power spectrum.
        """

        p = np.abs(np.diag(self.A_obs.conj().T @ self.R_obs @ self.A_obs)) / (self.in_n**2)
        for _ in range(iters):
            R_model = self.A_obs @ np.diag(p) @ self.A_obs.conj().T + 1e-6 * np.eye(self.in_n)
            R_inv = np.linalg.pinv(R_model)
            for k in range(len(self.angles)):
                ak = self.A_obs[:, k:k+1]
                num = np.abs(ak.conj().T @ R_inv @ self.R_obs @ R_inv @ ak).item()
                den = np.abs(ak.conj().T @ R_inv @ ak).item()**2
                p[k] = num / den
        return self.A_full @ np.diag(p) @ self.A_full.conj().T


    def solve_co_lasso(self, lmbda=0.05):
        r"""
        Compressive Covariance Lasso (Co-LASSO).
        
        Solves a convex optimization problem to find a sparse spatial power spectrum that 
        explains the observed covariance. The $L_1$ penalty term ($\lambda$) automatically 
        promotes a sparse solution, effectively identifying the number of sources by 
        suppressing noise floor components.
        """
        p = cp.Variable(len(self.angles), nonneg=True)
        R_model = self.A_obs @ cp.diag(p) @ self.A_obs.conj().T
        obj = cp.Minimize(cp.norm(self.R_obs - R_model, 'fro') + lmbda * cp.sum(p))
        cp.Problem(obj).solve(solver=cp.SCS)
        return self.A_full @ np.diag(p.value) @ self.A_full.conj().T


    def solve_sbl(self, max_iter=100):
        r"""
        Sparse Bayesian Learning (SBL).
        
        A hierarchical Bayesian framework for sparse signal recovery. It treats the source 
        powers as hyperparameters to be learned from the data. SBL is exceptionally 
        robust for unknown source counts as it naturally provides a sparse spectrum 
        and estimates the noise variance automatically.
        """
        gamma = np.ones(len(self.angles))
        sigma2 = 0.1
        for _ in range(max_iter):
            Sigma_y = self.A_obs @ np.diag(gamma) @ self.A_obs.conj().T + sigma2 * np.eye(self.in_n)
            inv_Sigma = np.linalg.pinv(Sigma_y)
            for k in range(len(self.angles)):
                ak = self.A_obs[:, k:k+1]
                num = np.abs(ak.conj().T @ inv_Sigma @ self.R_obs @ inv_Sigma @ ak).item()
                den = np.abs(ak.conj().T @ inv_Sigma @ ak).item()
                gamma[k] = gamma[k] * np.sqrt(num / den)
        return self.A_full @ np.diag(gamma) @ self.A_full.conj().T



# Testing block
if __name__ == "__main__":
    DOF = 45 
    TOTAL_ELEMENTS = DOF + 1 # 46
    
    # Simulate an array of 46 sensors, then select only 34 of them
    full_indices = np.arange(TOTAL_ELEMENTS)
    
    # Let's drop exactly 12 sensors randomly to simulate "holes"
    np.random.seed(42)
    dropped_indices = np.random.choice(full_indices[1:-1], 12, replace=False) # Keep 0 and 45
    virtual_indices = np.setdiff1d(full_indices, dropped_indices) # This has length 34
    
    # Ground Truth Full Covariance Matrix (46x46)
    R_true = toeplitz(0.9 ** np.arange(TOTAL_ELEMENTS))
    
    R_obs = np.zeros((34, 34), dtype=complex)
    for i, idx_i in enumerate(virtual_indices):
        for j, idx_j in enumerate(virtual_indices):
            R_obs[i, j] = R_true[idx_i, idx_j]
            
    print(f"Observed Matrix Shape: {R_obs.shape}")
    print(f"Target DoF: {DOF} (Output should be {TOTAL_ELEMENTS}x{TOTAL_ELEMENTS})\n")
    
    # Initialize the interpolator
    interpolator = CovarianceMatrixInterpolator(R_obs, virtual_indices, target_dof=DOF)

    methods = [
        ("Toeplitz NNM", interpolator.solve_toeplitz_nnm),
        ("Redundancy Averaging", interpolator.solve_redundancy_avg),
        ("Power Factorization", interpolator.solve_power_factorization),
        ("IAA", interpolator.solve_iaa),
        ("Co-LASSO", interpolator.solve_co_lasso),
        ("Sparse Bayesian Learning", interpolator.solve_sbl)
    ]
    
    for name, method in methods:
        try:
            R_out = method()
            print(f"{name:25} | Output Shape: {R_out.shape}")
        except Exception as e:
            print(f"{name:25} | FAILED: {str(e)}")