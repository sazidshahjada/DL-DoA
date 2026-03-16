import numpy as np
from scipy.optimize import linear_sum_assignment



def calculate_robust_rmse(true_angles, est_angles):
    """
    Calculate a robust RMSE between true and estimated angles, accounting for mismatches in the number of angles.
    Uses the Hungarian algorithm to find the optimal assignment between true and estimated angles, and applies a
    penalty for missed detections.
    
    Parameters:
    true_angles (array-like): Array of true angles (in degrees).
    est_angles (array-like): Array of estimated angles (in degrees).
    """
    true_angles = np.array(true_angles).reshape(-1, 1)
    est_angles = np.array(est_angles).reshape(-1, 1)
    dist_matrix = np.abs(true_angles - est_angles.T)
    true_idx, est_idx = linear_sum_assignment(dist_matrix) # Hungarian algorithm for optimal assignment
    sq_errors = (true_angles[true_idx].flatten() - est_angles[est_idx].flatten())**2
    num_missed = len(true_angles) - len(est_idx)
    if num_missed > 0:
        penalty = np.ones(num_missed) * 100 # Adjust penalty as needed
        sq_errors = np.concatenate([sq_errors, penalty])
        
    return np.sqrt(np.mean(sq_errors))

