import torch
import torch.nn.functional as F
import os

def EPE(prediction_disp: torch.Tensor, groundtruth_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Compute the End-Point-Error between predictions and ground-truth for a single batch."""

    assert prediction_disp.shape == groundtruth_disp.shape, f"{prediction_disp.shape} != {groundtruth_disp.shape}"
    if mask is not None:
        prediction_disp, groundtruth_disp = prediction_disp[mask], groundtruth_disp[mask]
    epe = F.l1_loss(prediction_disp, groundtruth_disp, reduction='mean')
    return epe

def D1(prediction_disp: torch.Tensor, groundtruth_disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Compute the D1-Error between prediction and ground-truth for a single batch."""

    assert prediction_disp.shape == groundtruth_disp.shape, f"{prediction_disp.shape} != {groundtruth_disp.shape}"
    if mask is not None:
        prediction_disp, groundtruth_disp = prediction_disp[mask], groundtruth_disp[mask]
    error = torch.abs(prediction_disp - groundtruth_disp)
    error_mask = (error > 3) & (error / groundtruth_disp.abs() > 0.05)
    d1 = torch.mean(error_mask.float())

    return d1 * 100.0

def Threshold(prediction_disp: torch.Tensor, groundtruth_disp: torch.Tensor, mask: torch.Tensor = None, threshold: float = 1.0) -> torch.Tensor:
    """
    Computes the percentage of pixels where the disparity error is larger than 'threshold' pixels.
    Commonly also refered to as BP-X or BadPixels-X, where 'X' is the treshold.
    """

    assert prediction_disp.shape == groundtruth_disp.shape, f"{prediction_disp.shape} != {groundtruth_disp.shape}"
    if mask is not None:
        prediction_disp, groundtruth_disp = prediction_disp[mask], groundtruth_disp[mask]
    error = torch.abs(prediction_disp - groundtruth_disp)
    error_mask = (error > threshold)
    error = torch.mean(error_mask.float())

    return error * 100.0

def log_memory(run, device):
    """Measure and log memory usage at the beginning of training."""
    torch.cuda.synchronize(device=device)
    peak_memory_reserved_bytes = torch.cuda.max_memory_reserved(device)
    peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_reserved_gib = peak_memory_reserved_bytes / (1024**3)
    peak_memory_allocated_gib = peak_memory_allocated_bytes / (1024**3)
    run.summary["Peak Memory Reserved"] = f"{peak_memory_reserved_gib:.2f} GiB"
    run.summary["Peak Memory Allocated"] = f"{peak_memory_allocated_gib:.2f} GiB"

def log_memory_val(run, device):
    """Measure and log memory usage at the beginning of training."""
    torch.cuda.synchronize(device=device)
    peak_memory_reserved_bytes = torch.cuda.max_memory_reserved(device)
    peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_reserved_gib = peak_memory_reserved_bytes / (1024**3)
    peak_memory_allocated_gib = peak_memory_allocated_bytes / (1024**3)
    run.summary["Peak Memory Reserved (Validation)"] = f"{peak_memory_reserved_gib:.2f} GiB"
    run.summary["Peak Memory Allocated (Validation)"] = f"{peak_memory_allocated_gib:.2f} GiB"

