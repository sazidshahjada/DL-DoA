import torch
import torch.nn as nn


class PINNLoss(nn.Module):
    """
    Physics-Informed Loss for Toeplitz Hermitian covariance matrix reconstruction.

    Loss = MSE + lambda_toeplitz * ToeplitzLoss + lambda_hermitian * HermitianLoss

    Args:
        lambda_toeplitz (float): Weight for the Toeplitz loss term.
        lambda_hermitian (float): Weight for the Hermitian loss term.
    """

    def __init__(self, lambda_toeplitz=1.0, lambda_hermitian=0.5):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lt = lambda_toeplitz
        self.lh = lambda_hermitian


    def toeplitz_loss(self, R_pred):
        """
        Enforces Toeplitz structure:
        elements along each diagonal should be constant
        """
        loss = 0
        M = R_pred.shape[-1]

        real = R_pred[:, 0]
        imag = R_pred[:, 1]

        for i in range(-(M - 1), M):

            diag_r = torch.diagonal(real, dim1=-2, dim2=-1, offset=i)
            diag_i = torch.diagonal(imag, dim1=-2, dim2=-1, offset=i)

            if diag_r.size(-1) > 1:
                loss += torch.mean(torch.var(diag_r, dim=-1))
                loss += torch.mean(torch.var(diag_i, dim=-1))

        return loss


    def hermitian_loss(self, R_pred):
        """
        Enforces Hermitian symmetry:
        R_ij = conj(R_ji)
        """

        real = R_pred[:, 0]
        imag = R_pred[:, 1]

        loss_real = torch.mean((real - real.transpose(-1, -2)) ** 2)
        loss_imag = torch.mean((imag + imag.transpose(-1, -2)) ** 2)

        return loss_real + loss_imag


    def forward(self, R_pred, R_true, R_obs_mask=None):

        # Data Loss
        if R_obs_mask is not None:
            loss_data = self.mse(R_pred * R_obs_mask, R_true * R_obs_mask)
        else:
            loss_data = self.mse(R_pred, R_true)

        # Physics Loss
        loss_toeplitz = self.toeplitz_loss(R_pred)
        loss_hermitian = self.hermitian_loss(R_pred)

        # Total Loss
        loss_total = loss_data + self.lt * loss_toeplitz + self.lh * loss_hermitian

        return loss_total