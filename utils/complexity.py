import torch
import torch.nn.functional as F
import os
from fvcore.nn import FlopCountAnalysis

def get_model_size(mdl):
    """Calculates model size in MB by saving its state_dict."""
    torch.save(mdl.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt") / 1e6
    os.remove('tmp.pt')
    return model_size

def get_model_complexity(model, args=None, run=None, input_shape = [544, 960]):
    """Computes model complexity."""

    if args is not None and hasattr(args, 'device'):
        DEVICE = args.device
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module 

    model_name = type(model).__name__
    model.eval()
    model.to(DEVICE)

    model_size_mb = get_model_size(model)
    model_size_gb = model_size_mb / 1024
    
    input_res = (1, 3, input_shape[0], input_shape[1]) # approx. SceneFlow Resolution
    print(f"Input Shape: {input_res}")
    
    dummy_input = (torch.randn(input_res).to(DEVICE), torch.randn(input_res).to(DEVICE))
    inputs_for_fvcore = dummy_input

    flops_analysis = FlopCountAnalysis(model, inputs_for_fvcore)
    
    # Clean output settings
    flops_analysis.uncalled_modules_warnings(False)
    flops_analysis.unsupported_ops_warnings(False)

    total_params = sum(p.numel() for p in model.parameters())
    total_macs = flops_analysis.total()
    total_flops = 2 * total_macs # Standard definition: 1 MAC = 2 FLOPs

    print(f"Model: {model_name}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Model Size: {model_size_gb:.2f} GB")
    print(f"Total MACs: {total_macs / 1e9:.2f} GMacs")
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPS")
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    if run is not None: 
        run.summary[f"{model_name} Size (MB)"] = f"{model_size_mb:.2f} MB"
        run.summary[f"{model_name} Size (GB)"] = f"{model_size_gb:.2f} GB"
        run.summary[f"{model_name} Total MACs"] = f"{total_macs / 1e9:.2f} GMacs"
        run.summary[f"{model_name} Total FLOPs"] = f"{total_flops / 1e9:.2f} GFLOPS"
        run.summary[f"{model_name} Parameters"] = f"{total_params / 1e6:.2f} M"