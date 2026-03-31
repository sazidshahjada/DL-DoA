import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.functional as F
from sklearn.model_selection import train_test_split


def complex_to_tensor(matrix):
    """
    Converts a complex numpy array or torch tensor (M, M) 
    into a real tensor (2, M, M) for CNN input.
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    
    # Extract real and imaginary parts
    real_part = matrix.real
    imag_part = matrix.imag
    
    # Stack along a new channel dimension: [2, M, M]
    return torch.stack([real_part, imag_part], dim=0).float()


def tensor_to_complex(tensor):
    """
    Converts a (2, M, M) or (Batch, 2, M, M) tensor back 
    to a complex torch tensor.
    """
    # Assuming channel dim is 1 for batched or 0 for single
    if tensor.dim() == 4:
        return torch.complex(tensor[:, 0, :, :], tensor[:, 1, :, :])
    return torch.complex(tensor[0, :, :], tensor[1, :, :])


class DoACovarianceDataset(Dataset):
    """
    A PyTorch Dataset for loading observed and true covariance matrices 
    for DOA estimation tasks.

    Args:
        obs_paths (list of str): List of file paths to observed covariance matrices (.npz).
        true_paths (list of str): List of file paths to true covariance matrices (.npz).
    """
    def __init__(self, obs_paths, true_paths):
        self.obs_paths = obs_paths
        self.true_paths = true_paths

    def __len__(self):
        return len(self.obs_paths)

    def _load_npz(self, path):
        with np.load(path) as data:
            return data[data.files[0]]

    def __getitem__(self, idx):
        R_obs = self._load_npz(self.obs_paths[idx])
        R_true = self._load_npz(self.true_paths[idx])

        norm_factor = np.trace(np.abs(R_obs)) + 1e-12
        
        x = complex_to_tensor(R_obs / norm_factor)
        y = complex_to_tensor(R_true / norm_factor)

        return x, y, torch.tensor(norm_factor, dtype=torch.float32)
    

def get_dataloaders(obs_dir, true_dir, val_size=0.1, test_size=0.1, seed=42):
    """
    Splits file paths into train, validation, and test sets and returns 
    three instances of DoACovarianceDataset.
    """
    obs_paths = sorted(glob.glob(os.path.join(obs_dir, "*.npz")))
    true_paths = sorted(glob.glob(os.path.join(true_dir, "*.npz")))

    train_val_obs, test_obs, train_val_true, test_true = train_test_split(
        obs_paths, true_paths, test_size=test_size, random_state=seed, shuffle=True
    )

    relative_val_size = val_size / (1 - test_size)
    train_obs, val_obs, train_true, val_true = train_test_split(
        train_val_obs, train_val_true, test_size=relative_val_size, random_state=seed, shuffle=True
    )

    train_ds = DoACovarianceDataset(train_obs, train_true)
    val_ds = DoACovarianceDataset(val_obs, val_true)
    test_ds = DoACovarianceDataset(test_obs, test_true)

    return train_ds, val_ds, test_ds
    




if __name__ == "__main__":
    M = 4
    A = np.random.randn(M, M) + 1j * np.random.randn(M, M)
    R_obs = A @ A.conj().T
    
    R_obs_tensor = complex_to_tensor(R_obs)
    R_obs_back = tensor_to_complex(R_obs_tensor)

    is_correct = np.allclose(R_obs, R_obs_back.numpy())
    
    print(f"Reconstruction Match: {is_correct}")
    print("Tensor Shape:", R_obs_tensor.shape) # Expected: torch.Size([2, 4, 4])