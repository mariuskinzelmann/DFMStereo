# External Imports
import os
import sys
import torch

import warnings

# Internal Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.transform import *
from utils.loss import *
from utils.distillation_tools import *
from utils.experiment import *
from utils.metrics import *
from utils.complexity import *

from models.FoundationStereo.core.foundation_stereo import init_foundation_stereo
from models.FastFoundationStereo.core.foundation_stereo import init_fast_foundation_stereo
from models.DFMStereo.core.dfmstereo_large import init_dfmstereo_large
from models.DFMStereo.core.dfmstereo_medium import init_dfmstereo_medium
from models.DFMStereo.core.dfmstereo_small import init_dfmstereo_small

from dataset.dataset import *

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser(description='Model Complexity Analysis')

    args = parser.parse_args()
    args.rank = 0
    input_resolution = [480, 640]
    args.dfmstereo_ckpt = 'million_scale'

    dfmstereo_large = init_dfmstereo_large(args, eval=True)
    with torch.no_grad():
        get_model_complexity(dfmstereo_large, input_shape=input_resolution)
    dfmstereo_large.cpu()
    del dfmstereo_large

    dfmstereo_medium = init_dfmstereo_medium(args, eval=True)
    with torch.no_grad():
        get_model_complexity(dfmstereo_medium, input_shape=input_resolution)
    dfmstereo_medium.cpu()
    del dfmstereo_medium

    dfmstereo_small = init_dfmstereo_small(args, eval=True)
    with torch.no_grad():
        get_model_complexity(dfmstereo_small, input_shape=input_resolution)
    dfmstereo_small.cpu()
    del dfmstereo_small

    fast_foundation_stereo = init_fast_foundation_stereo(args)
    with torch.no_grad():
        get_model_complexity(fast_foundation_stereo, input_shape=input_resolution)
    fast_foundation_stereo.cpu()
    del fast_foundation_stereo

    foundation_stereo = init_foundation_stereo(args)
    with torch.no_grad():
        get_model_complexity(foundation_stereo, input_shape=input_resolution)
    foundation_stereo.cpu()
    del foundation_stereo


    


