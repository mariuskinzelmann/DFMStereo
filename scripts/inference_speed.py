"""
Script for measuring inference speed. Models run using their respective default configurations.
"""

# External Imports
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from argparse import Namespace
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
from models.DFMStereo.core.utils.utils import InputPadder

from collections import defaultdict

def measure_inference_speed(model, inputs, warmup_runs=10, benchmark_runs=100, use_triton=False):
    """
    Correctly measures GPU inference speed using CUDA events.
    
    Args:
        model: PyTorch model (should be in eval mode on CUDA)
        inputs: tuple of input tensors (already on CUDA)
        warmup_runs: number of warmup iterations (not measured)
        benchmark_runs: number of timed iterations to average over
    
    Returns:
        avg_ms: average inference time in milliseconds
        std_ms: standard deviation in milliseconds
    """
    model.eval()

    # --- 1. Warmup ---
    # Ensures CUDA kernels are compiled and memory is allocated
    if not use_triton:
        with torch.no_grad():
            for _ in range(warmup_runs):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(*inputs, iters=8, test_mode=True)
    else:
        with torch.inference_mode():
            for _ in range(warmup_runs):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(*inputs, iters=8, test_mode=True)
    # Wait for all warmup ops to finish before starting the clock
    torch.cuda.synchronize()

    # --- 2. Benchmark using CUDA Events ---
    # CUDA events are placed directly on the GPU timeline — far more
    # accurate than CPU wall-clock time for GPU workloads
    timings = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if not use_triton:
        with torch.no_grad():
            for _ in range(benchmark_runs):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    start_event.record()
                    _ = model(*inputs, iters=8, test_mode=True)
                    end_event.record()

                # Block CPU until GPU is done for this iteration
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))  # ms
    else:
        with torch.inference_mode():
            for _ in range(benchmark_runs):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    start_event.record()
                    _ = model(*inputs, iters=8, test_mode=True)
                    end_event.record()

                # Block CPU until GPU is done for this iteration
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))  # ms

    avg_ms = sum(timings) / len(timings)
    std_ms = (sum((t - avg_ms) ** 2 for t in timings) / len(timings)) ** 0.5
    return avg_ms, std_ms

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    import torch._dynamo
    
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Teacher Knowledge Distillation for Stereo Vision.')
    parser.add_argument("--use_triton", action='store_true')
    parser.add_argument("--mixed_precision", action='store_true')
    parser.add_argument("--precision_dtype", default="float32")

    args = parser.parse_args()
    #1280x720
    W = 640
    H = 480
    dummy_left = torch.ones(size=(3, H, W), dtype=torch.float32).cuda().unsqueeze(0)
    dummy_right = torch.ones(size=(3, H, W), dtype=torch.float32).cuda().unsqueeze(0)
    args.mixed_precision = True
    args.precision_dtype = "float16"
    args.use_triton = True
    args.optimise_volume_build = True

    print(f"Testing Inference Speed on Resolution (WxH): {W} x {H}")
    print(f"Using Triton: {args.use_triton}")
    print(f"Mixed Precison: {args.mixed_precision}")
    print(f"precision_dtype: {args.precision_dtype}")

    dfmstereo_large = init_dfmstereo_large(args, eval=True)
    dfmstereo_large.cuda().half()
    #torch._dynamo.config.suppress_errors = True
    if args.use_triton:
        #dfmstereo_large  = torch.compile(dfmstereo_large, mode="max-autotune")
        dfmstereo_large  = torch.compile(dfmstereo_large, mode="max-autotune-no-cudagraphs")
        #dfmstereo_large  = torch.compile(dfmstereo_large, mode="reduce-overhead")

    avg, std = measure_inference_speed(
        model=dfmstereo_large,
        inputs=(dummy_left, dummy_right),
        warmup_runs=30,
        benchmark_runs=100,
        use_triton=args.use_triton
        )
    
    print(f"DFMStereo-Large:   {avg:.2f} ms ± {std:.2f} ms")
    dfmstereo_large.cpu()
    del dfmstereo_large





    """ dfmstereo_medium = init_dfmstereo_medium(args, eval=True)
    


    dfmstereo_medium.cpu()
    del dfmstereo_medium





    dfmstereo_small = init_dfmstereo_small(args, eval=True)
    
    

    dfmstereo_small.cpu()
    del dfmstereo_small





    fast_foundation_stereo = init_fast_foundation_stereo(args)
    


    del fast_foundation_stereo





    foundation_stereo = init_foundation_stereo(args)
    


    foundation_stereo.cpu()
    del foundation_stereo """
    
    print(torch._dynamo.utils.compile_times())