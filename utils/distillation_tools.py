import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
import torch.distributed as dist
from models.FoundationStereo.core.submodule import BasicConv
from torch.amp import autocast
        
class Paraphraser(nn.Module):
    """
    https://arxiv.org/abs/1802.04977
    """
    def __init__(self, in_channels=224, latent_dim=128, use_bn=False):
        super(Paraphraser, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=in_channels, bn=use_bn, kernel_size=3, padding=1),
            BasicConv(in_channels=in_channels, out_channels=latent_dim, bn=use_bn, kernel_size=1, padding=0),
            BasicConv(in_channels=latent_dim, out_channels=latent_dim, bn=use_bn, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            BasicConv(in_channels=latent_dim, out_channels=latent_dim, bn=use_bn, kernel_size=3, padding=1),
            BasicConv(in_channels=latent_dim, out_channels=in_channels, bn=use_bn, kernel_size=1, padding=0),
            BasicConv(in_channels=in_channels, out_channels=in_channels, bn=use_bn, kernel_size=3, padding=1)
        )
    
    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        return latent if return_latent else self.decoder(latent)

class Translator(nn.Module):
    """
    https://arxiv.org/abs/1802.04977
    """
    def __init__(self, in_channels=128, latent_dim=128, use_bn=False):
        super(Translator, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=in_channels, bn=use_bn, kernel_size=1, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)
        
    