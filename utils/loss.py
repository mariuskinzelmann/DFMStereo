import torch
import torch.nn as nn
import torch.nn.functional as F

class LossHyperparameters(nn.Module):
    """
    A module to hold all learnable hyperparameters for the loss functions. These
    learnable hyperparameters weigh the ground-truth loss vs the distillation loss.

    alpha: Weight for disparity loss.
    """

    def __init__(self, initial_alpha=0.8):
        super().__init__()
        # Convert the initial percentage to a logit (logistic unit)
        logit_alpha = torch.logit(torch.tensor(initial_alpha), eps=0.0)
        self.logit_alpha_param = nn.Parameter(logit_alpha)

    def forward(self):
        # Apply sigmoid to constrain logits to [0,1]
        weights = {
            'alpha': torch.sigmoid(self.logit_alpha_param),
        }
        return weights

class LogL1Loss(nn.Module):
    """Numerically stable version of LogL1-Loss. Applied to disparity loss."""
    def __init__(self, epsilon=1.0):
        super(LogL1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, f_s, f_t) -> torch.Tensor:
        loss = torch.log(torch.abs(f_s - f_t) + self.epsilon).mean()
        return loss
    
def gram_matrix_loss(features_student, features_teacher):
    assert features_student.shape == features_teacher.shape
    B, C, H, W = features_student.shape
    X_flat = features_student.view(B,C,-1)
    Y_flat = features_teacher.view(B,C,-1)

    X_norm = F.normalize(X_flat, p=2, dim=2)
    Y_norm = F.normalize(Y_flat, p=2, dim=2)

    gram_matrix = torch.bmm(X_norm, Y_norm.transpose(1,2))
    assert gram_matrix.shape == (B,C,C), f"Expected Gram Matrix Shape: {B}x{C}x{C}. Instead got {gram_matrix.shape}"
    assert torch.all((gram_matrix>=-1) & (gram_matrix<=1)), f"Gram Matrix Correlation values must be in range [-1,1]."
    diagonal_mask = torch.eye(C, dtype=torch.bool, device=features_student.device).expand(B, -1, -1)
    off_diagonal_mask = ~diagonal_mask
    
    diagonal_gram_matrix = gram_matrix[diagonal_mask]
    diagonal_identity = torch.ones_like(diagonal_gram_matrix, dtype=features_student.dtype, device=features_student.device)
    diagonal_loss = F.mse_loss(diagonal_identity, diagonal_gram_matrix)

    off_diagonal_gram_matrix = gram_matrix[off_diagonal_mask]
    off_diagonal_loss = torch.mean(off_diagonal_gram_matrix**2)

    alpha = 1
    loss = diagonal_loss + alpha * off_diagonal_loss

    return loss

class CostVolumeCosineSimilarity(nn.Module):
    def __init__(self, dim=1):
        super(CostVolumeCosineSimilarity, self).__init__()
        self.dim = dim
        self.CosineSimilarity = nn.CosineSimilarity(dim=self.dim)

    def forward(self, cost_volume_student, cost_volume_teacher):
        assert cost_volume_student.shape == cost_volume_teacher.shape, f"Student and Teacher Cost Volume Dimensions do not match. Student: {cost_volume_student.shape}. Teacher: {cost_volume_teacher.shape}"
        assert len(cost_volume_student.shape) == 5 or len(cost_volume_student.shape) == 4, f"Expected 5D Tensor: [Batch, Correlation, Disp, Height, Width] or 4D Tensor: [Batch, Disp, Height, Width]. Received {cost_volume_student.shape}."
        loss = 1 - self.CosineSimilarity(cost_volume_student, cost_volume_teacher)
        return loss.mean()

def stem_distillation_loss(stem_features_student, stem_features_teacher):
    """Distillation Loss for Steam Features at 1/2 and 1/4 resolution."""

    assert len(stem_features_student) == len(stem_features_teacher)
    weights = [1.0, 1.0, 1.0, 1.0]
    loss_fn = nn.CosineSimilarity(dim=1)
    losses = []
    for stem_feature_student, stem_feature_teacher, weight in zip(stem_features_student, stem_features_teacher, weights):
        assert stem_feature_student.shape == stem_feature_teacher.shape
        loss = weight * ((1 - loss_fn(stem_feature_student, stem_feature_teacher)).mean())
        losses.append(loss)

    return sum(losses) / sum(weights)

def igev_sequence_loss_geo_vol(init_disp, iter_preds, disparity_gt, loss_gamma=0.9, max_disp=416) -> torch.Tensor:
    """Loss function defined over sequence of flow predictions. Works for IGEV Models which only predict a single initial disparity."""

    n_predictions = len(iter_preds)
    assert n_predictions >= 1

    disp_loss = 0.0
    mag = disparity_gt.abs()
    mask = ((disparity_gt >= 0.5) & (mag < max_disp)).unsqueeze(1) # [B, 1 , H, W]
    disparity_gt = disparity_gt.unsqueeze(1) # [B, 1 , H, W]
    assert mask.shape == disparity_gt.shape, [mask.shape, disparity_gt.shape]
    assert not torch.isinf(disparity_gt[mask.bool()]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(init_disp[0][mask.bool()], disparity_gt[mask.bool()], reduction='mean')

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (iter_preds[i] - disparity_gt).abs()
        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disparity_gt.shape, iter_preds[i].shape]
        disp_loss += i_weight * i_loss[mask.bool()].mean()

    return disp_loss
