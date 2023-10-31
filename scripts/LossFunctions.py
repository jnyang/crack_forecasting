# Tutorial for building custom PINN loss fns: https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4

import torch
import torch.nn as nn
import pytorch_msssim as ssim

from torch.func import functional_call, grad, vmap

class SSIMLoss(nn.Module):
    """ 
    Structural Similarity Index (SSIM) measures the structural similarity between two images 
    and considers luminance, contrast, and structure. Higher SSIM values indicate better 
    similarity between the images.

    Example usage:
    ssim_loss_fn = SSIMLoss()
    predicted_image = torch.randn(1, 3, 256, 256)  # Example predicted image tensor
    ground_truth = torch.randn(1, 3, 256, 256)  # Example ground truth tensor
    loss = ssim_loss_fn(predicted_image, ground_truth)
    """
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, predicted_image, ground_truth):
        """
        Calculate SSIM loss between predicted and ground truth images.

        Args:
            predicted_image (torch.Tensor): Predicted crack image.
            ground_truth (torch.Tensor): Ground truth crack image.

        Returns:
            torch.Tensor: SSIM loss.
        """
        ssim_score = 1 - ssim.ssim(predicted_image, ground_truth, data_range=1.0)
        return ssim_score


class PhaseFieldLoss(nn.Module):
    """
    Phase field method is a numerical technique for modeling fracture.
    Wiki: https://en.wikipedia.org/wiki/Phase_field_method
    """

    def __init__(self, alpha=1.0, delta_t=0.1):
        super(PhaseFieldLoss, self).__init__()
        self.alpha = alpha  # Weighting factor
        self.delta_t = delta_t  # Time step
    
    def forward(self, predicted_image, ground_truth):
        """
        Calculate phase field loss between predicted and ground truth images.

        Args:
            predicted_image (torch.Tensor): Predicted crack image.
            ground_truth (torch.Tensor): Ground truth crack image.

        Returns:
            torch.Tensor: Phase field loss.
        """

        # Check and print the shapes of the tensors
        print(f"Predicted Image Shape: {predicted_image.shape}")
        print(f"Ground Truth Shape: {ground_truth.shape}")

        # MSE Loss between prediction and ground truth
        mse_loss = nn.MSELoss()(predicted_image, ground_truth)
        
        # Approximate the spatial gradient using finite differences
        grad_phi_x = (predicted_image[:, :, 1:] - predicted_image[:, :, :-1])[:, :, :-1]
        grad_phi_y = (predicted_image[:, 1:, :] - predicted_image[:, :-1, :])[:, :-1, :]
        
        print(f"Grad Phi X Shape: {grad_phi_x.shape}")
        print(f"Grad Phi Y Shape: {grad_phi_y.shape}")

        # Ensure the gradients and the residual are calculated over matching dimensions
        pde_residual = (predicted_image[:, 1:-1, 1:-1] - ground_truth[:, 1:-1, 1:-1]) / self.delta_t \
                       - torch.sqrt(grad_phi_x[:, 1:-1, 1:-1]**2 + grad_phi_y[:, 1:-1, 1:-1]**2)
        
        print(f"PDE Residual Shape: {pde_residual.shape}")

        phase_field_loss = torch.mean(pde_residual**2)
        
        # Combine the two losses with a weighting factor
        total_loss = mse_loss + self.alpha * phase_field_loss
        
        return total_loss



class SIFLoss(nn.Module):
    """
    The stress intensity factor (SIF) is used to predict the stress state ("stress intensity") 
    near the tip of a crack or notch caused by a remote load or residual stresses.
    The magnitude of K depends on specimen geometry, the size and location of the crack or notch, 
    and the magnitude and the distribution of loads on the material.
    Wiki: https://en.wikipedia.org/wiki/Stress_intensity_factor
    """

    def __init__(self):
        super(SIFLoss, self).__init__()

    def forward(self, predicted_image, ground_truth):
        """
        Calculate SIF loss between predicted and ground truth images.

        Args:
            predicted_image (torch.Tensor): Predicted crack image.
            ground_truth (torch.Tensor): Ground truth crack image.

        Returns:
            torch.Tensor: SIF loss.
        """
        pass
        

    
class CTODLoss(nn.Module):
    """
    Crack tip opening displacement (CTOD) is the distance between the opposite faces of a crack tip at the 90° intercept position. 
    The position behind the crack tip at which the distance is measured is arbitrary but commonly used is the point where two 45° lines, 
    starting at the crack tip, intersect the crack faces.The parameter is used in fracture mechanics to characterize the loading on a 
    crack and can be related to other crack tip loading parameters such as the stress intensity factor K and the elastic-plastic J-integral.
    Wiki: https://en.wikipedia.org/wiki/Crack_tip_opening_displacement

    """

    def __init__(self):
        super(CTODLoss, self).__init__()

    def forward(self, predicted_image, ground_truth):
        """
        Calculate CTOD loss between predicted and ground truth images.

        Args:
            predicted_image (torch.Tensor): Predicted crack image.
            ground_truth (torch.Tensor): Ground truth crack image.

        Returns:
            torch.Tensor: CTOD loss.
        """
        pass


class JIntegralLoss(nn.Module):
    """
    The J-integral is a method of analysis used to determine the stress 
    intensity factor at the tip of a crack in a linear elastic material
    Wiki: https://en.wikipedia.org/wiki/J-integral
    """

    def __init__(self):
        super(JIntegralLoss, self).__init__()

    def forward(self, predicted_image, ground_truth):
        """
        Calculate J-integral loss between predicted and ground truth images.
        """
        pass