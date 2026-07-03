from torch.nn import SmoothL1Loss
from .loss import LogL1Loss

__loss_functions__ = {
    "SmoothL1Loss": (SmoothL1Loss, {"reduction": "mean"}),
    "LogL1Loss": (LogL1Loss, {"epsilon": 1.0}),
}