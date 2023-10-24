import torch
import torch.nn as nn
import pytorch_msssim as ssim

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

