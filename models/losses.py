import torch
import torch.nn as nn


def toeplitz_loss(output):
    loss = 0
    M = output.shape[-1]
    for i in range(-(M-1), M):
        diag_real = torch.diagonal(output[:, 0, :, :], dim1=-2, dim2=-1, offset=i)
        diag_imag = torch.diagonal(output[:, 1, :, :], dim1=-2, dim2=-1, offset=i)
        loss += torch.var(diag_real) + torch.var(diag_imag)
    return loss



class PINNLoss(nn.Module):
    def __init__(self, lambda_toeplitz=1.0, lambda_hermitian=0.5):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lt = lambda_toeplitz
        self.lh = lambda_hermitian

    def forward(self, R_pred, R_true, R_obs_mask):
        # 1. Standard MSE (only on known/observed points if using a mask)
        loss_data = self.mse(R_pred, R_true)

        # 2. Toeplitz Physics Loss (Variance along diagonals)
        loss_toeplitz = 0
        M = R_pred.shape[-1]
        
        # We iterate through all diagonals (from -(M-1) to M-1)
        for i in range(-(M-1), M):
            diag_r = torch.diagonal(R_pred[:, 0, :, :], dim1=-2, dim2=-1, offset=i)
            diag_i = torch.diagonal(R_pred[:, 1, :, :], dim1=-2, dim2=-1, offset=i)
            
            if diag_r.size(-1) > 1:
                loss_toeplitz += torch.mean(torch.var(diag_r, dim=-1))
                loss_toeplitz += torch.mean(torch.var(diag_i, dim=-1))

        # 3. Hermitian Symmetry Loss (R_ij = conj(R_ji))
        real_part = R_pred[:, 0, :, :]
        imag_part = R_pred[:, 1, :, :]
        
        loss_herm_real = self.mse(real_part, real_part.transpose(-1, -2))
        loss_herm_imag = self.mse(imag_part, -imag_part.transpose(-1, -2))
        loss_hermitian = loss_herm_real + loss_herm_imag

        return loss_data + (self.lt * loss_toeplitz) + (self.lh * loss_hermitian)
    


if __name__ == "__main__":
    # Example usage
    batch_size, channels, M = 4, 2, 83
    R_pred = torch.randn(batch_size, channels, M, M)
    R_true = torch.randn(batch_size, channels, M, M)
    R_obs_mask = torch.ones(batch_size, channels, M, M)

    criterion = PINNLoss(lambda_toeplitz=1.0, lambda_hermitian=0.5)
    loss = criterion(R_pred, R_true, R_obs_mask)
    print("PINN Loss:", loss.item())