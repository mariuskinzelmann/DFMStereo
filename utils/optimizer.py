import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from enum import IntEnum
from typing import List, Tuple

def get_parameter_groups_edge_next(feature_backbone):
    """
    Groups Feature Backbone Parameters into Decay and No-Decay.
    Traverses all submodules to collect no_weight_decay parameters,
    since no_weight_decay() is defined at the submodule level (e.g. XCA).
    """
    skip = set()
    
    # Traverse all submodules to find no_weight_decay definitions
    for module_name, module in feature_backbone.named_modules():
        if hasattr(module, 'no_weight_decay'):
            for param_name in module.no_weight_decay():
                # Construct the full parameter path
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                skip.add(full_name)

    if skip:
        print(f"Skipping weight decay for: {skip}")

    decay_params = []
    no_decay_params = []

    for name, param in feature_backbone.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
            print(f"No Decay: {name}")
            no_decay_params.append(param)
        else:
            print(f"Decay: {name}")
            decay_params.append(param)

    return decay_params, no_decay_params



def fetch_optimizer_scheduler(model, args):
    feature_backbone = model.feature
    feature_backbone_decay, feature_backbone_no_decay = get_parameter_groups_edge_next(feature_backbone)
    edge_next_learning_rate = args.edge_next_learning_rate if hasattr(args, 'edge_next_learning_rate') else 2e-4
    edge_next_weight_decay = args.edge_next_weight_decay if hasattr(args, 'edge_next_weight_decay') else 0.05
    if args.rank == 0:
        print(f"EdgeNext Learning Rate: {edge_next_learning_rate}")
        print(f"EdgeNext Weight Decay: {edge_next_weight_decay}")
    feature_backbone_params = [
        {'params': feature_backbone_decay, 'lr': edge_next_learning_rate, 'weight_decay': edge_next_weight_decay},
        {'params': feature_backbone_no_decay, 'lr': edge_next_learning_rate, 'weight_decay': 0.0}
    ]
    

    model_decay = []
    model_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('feature.'):
            continue
        
        if len(param.shape) == 1 or name.endswith(".bias"):
            print(f"No Decay: {name}")
            model_no_decay.append(param)
        else:
            print(f"Decay: {name}")
            model_decay.append(param)

    
    model_learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 2e-4
    model_weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 1e-4

    if args.rank == 0:    
        print(f"Model Learning Rate: {model_learning_rate}")
        print(f"Model Weight Decay: {model_weight_decay}")

    params = [
        {'params': model_decay, 'lr': model_learning_rate, 'weight_decay': model_weight_decay},
        {'params': model_no_decay, 'lr': model_learning_rate, 'weight_decay': 0.0},
    ]

    
    params = feature_backbone_params + params
    max_lr=[edge_next_learning_rate, edge_next_learning_rate, model_learning_rate, model_learning_rate]

    optimizer = optim.AdamW(params, eps=1e-8)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=args.total_steps,
        pct_start=args.pct_start,
        cycle_momentum=False,
        anneal_strategy='cos',
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        three_phase=False,
    )

    return optimizer, lr_scheduler



class TrainingStage(IntEnum):
    FEATURE_EXTRACTION    = 0
    COST_FILTERING        = 1
    DISPARITY_REFINEMENT  = 2
 


def get_parameter_groups_edge_next(
    feature_backbone: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Splits EdgeNext (CNN-Transformer) backbone parameters into two groups:
      - decay_params    : multi-dim weights that are NOT in any no_weight_decay set
      - no_decay_params : 1-D params, biases, and params listed in no_weight_decay()
 
    Traverses *all* submodules so that no_weight_decay() defined at any
    submodule level (e.g. XCA) is respected.
    """
    skip: set[str] = set()
 
    for module_name, module in feature_backbone.named_modules():
        if hasattr(module, "no_weight_decay"):
            for param_name in module.no_weight_decay():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                skip.add(full_name)
 
    #if skip:
        #print(f"[EdgeNext] Skipping weight decay for: {skip}")
 
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
 
    for name, param in feature_backbone.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
            #print(f"[EdgeNext] No-decay: {name}")
            no_decay_params.append(param)
        else:
            #print(f"[EdgeNext] Decay:    {name}")
            decay_params.append(param)
 
    return decay_params, no_decay_params



def get_parameter_groups_generic(
    modules: List[nn.Module],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Generic weight-decay split for non-EdgeNext stages (cost filtering,
    disparity refinement).
 
    Rules (standard AdamW convention):
      - 1-D tensors  → no decay  (BatchNorm/LayerNorm scale & bias, standalone bias)
      - bias params  → no decay
      - everything else → decay
    """
    seen: set[int] = set()   # deduplicate by param data_ptr
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
 
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
 
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
 
    return decay_params, no_decay_params


 
def build_param_groups(
    model,
    stage: TrainingStage,
    args,
) -> List[dict]:
    """
    Returns a list of AdamW parameter-group dicts for `stage`.
 
    Each previously-trained stage gets its LR reduced by
    `args.progressive_kd_decay_factor`:
 
    Parameter groups are **named** via the 'name' key so the caller and the
    LR scheduler can identify them unambiguously.
 
    Args:
        model : the stereo-matching model instance
        stage : current TrainingStage
        args  : argparse Namespace; must expose
                    args.learning_rate
                    args.weight_decay
                    args.progressive_kd_decay_factor
    """
    lr_decay = args.pkd_lr_decay_factor
    base_lr = args.learning_rate
    weight_decay = args.weight_decay
 
    groups: List[dict] = []
 
    fe_lr = base_lr * lr_decay if stage > TrainingStage.FEATURE_EXTRACTION else base_lr
    fe_decay, fe_no_decay = get_parameter_groups_edge_next(model.feature)
 
    if fe_decay:
        groups.append({
            "name":          "feature_extraction/decay",
            "params":        fe_decay,
            "lr":            fe_lr,
            "weight_decay":  weight_decay,
        })
    if fe_no_decay:
        groups.append({
            "name":          "feature_extraction/no_decay",
            "params":        fe_no_decay,
            "lr":            fe_lr,
            "weight_decay":  0.0,
        })
 
    if stage >= TrainingStage.COST_FILTERING:
        cf_lr = base_lr * lr_decay if stage > TrainingStage.COST_FILTERING else base_lr
        cf_decay, cf_no_decay = get_parameter_groups_generic(
            model.cost_filtering_modules
        )
 
        if cf_decay:
            groups.append({
                "name":          "cost_filtering/decay",
                "params":        cf_decay,
                "lr":            cf_lr,
                "weight_decay":  weight_decay,
            })
        if cf_no_decay:
            groups.append({
                "name":          "cost_filtering/no_decay",
                "params":        cf_no_decay,
                "lr":            cf_lr,
                "weight_decay":  0.0,
            })
 
    if stage >= TrainingStage.DISPARITY_REFINEMENT:
        dr_lr = base_lr * lr_decay if stage > TrainingStage.DISPARITY_REFINEMENT else base_lr
        dr_decay, dr_no_decay = get_parameter_groups_generic(
            model.disparity_refinement_modules
        )
 
        if dr_decay:
            groups.append({
                "name":          "disparity_refinement/decay",
                "params":        dr_decay,
                "lr":            dr_lr,
                "weight_decay":  weight_decay,
            })
        if dr_no_decay:
            groups.append({
                "name":          "disparity_refinement/no_decay",
                "params":        dr_no_decay,
                "lr":            dr_lr,
                "weight_decay":  0.0,
            })
 
    return groups


 
def _set_modules_grad(modules: List[nn.Module], requires_grad: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(requires_grad)
        if requires_grad:
            m.train()
        else:
            m.eval()


 
def freeze_all_stages(model) -> None:
    """Freeze every stage — call before Stage 0 training begins."""
    _set_modules_grad(model.feature_extraction_modules,   False)
    _set_modules_grad(model.cost_filtering_modules,       False)
    _set_modules_grad(model.disparity_refinement_modules, False)


 
def unfreeze_stage(model, stage: TrainingStage) -> None:
    """
    Unfreeze only the modules that belong to `stage`.
    Previously unfrozen stages remain trainable.
    """
    if stage == TrainingStage.FEATURE_EXTRACTION:
        _set_modules_grad(model.feature_extraction_modules, True)
        print("Unfroze: feature_extraction")
    elif stage == TrainingStage.COST_FILTERING:
        _set_modules_grad(model.cost_filtering_modules, True)
        print("Unfroze: cost_filtering")
    elif stage == TrainingStage.DISPARITY_REFINEMENT:
        _set_modules_grad(model.disparity_refinement_modules, True)
        print("Unfroze: disparity_refinement")



def get_optimizer(
    model,
    stage: TrainingStage,
    args,
) -> optim.AdamW:
    """
    Builds a **fresh** AdamW for `stage`, resetting all momentum / second-
    moment estimates.  This is intentional: when a new stage is unlocked the
    loss landscape changes significantly due to the new KD targets, so stale
    momentum from the previous stage would bias the update direction.
 
    Weight decay stays constant (args.weight_decay) across all stages.
 
    Args:
        model : stereo-matching model
        stage : current TrainingStage
        args  : Namespace with lr, weight_decay, progressive_kd_decay_factor,
                and optionally betas / eps (falls back to AdamW defaults)
 
    Returns:
        A freshly initialised AdamW optimiser.
    """
    param_groups = build_param_groups(model, stage, args)
 
    betas = getattr(args, "betas", (0.9, 0.999))
    eps   = getattr(args, "eps",   1e-8)
 
    optimizer = optim.AdamW(
        param_groups,
        betas=betas,
        eps=eps,
        # lr and weight_decay are set per-group; these are just fallback defaults
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.rank == 0:
        _log_optimizer_groups(optimizer, stage)
    return optimizer 



def _log_optimizer_groups(optimizer: optim.AdamW, stage: TrainingStage) -> None:
    print(f"\n[Optimizer] Stage {stage.name} — {len(optimizer.param_groups)} param groups:")
    for g in optimizer.param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(
            f"[{g['name']}]  lr={g['lr']:.2e}"
            f"wd={g['weight_decay']:.1e}  params={n_params:,}"
        )
 


def get_lr_scheduler(
    optimizer: optim.AdamW,
    args,
    stage: TrainingStage,
):
    """
    Returns a LR scheduler for the current stage.
 
    Currently a simple placeholder — replace the lambda body with your
    chosen schedule (cosine annealing, one-cycle, warmup + decay, etc.).
 
    The lambda receives `epoch` (0-indexed within the current stage) and
    returns a *multiplicative factor* applied on top of each group's
    base lr.  All groups are scaled by the same factor; the progressive
    LR differences between groups are already baked into the group LRs
    at optimizer-construction time and should not be disturbed here.
 
    Args:
        optimizer : AdamW returned by get_optimizer()
        args      : Namespace; expose any schedule hyperparams here
        stage     : current TrainingStage (available for stage-conditional logic)
 
    Returns:
        A torch LambdaLR scheduler.
    """
 
    def lr_lambda(epoch: int) -> float:
        return 1.0
 
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler
