from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.optim.optimizer import Optimizer
from collections import defaultdict
import torch.distributed as dist
import torch
import math
from pathlib import Path
#import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

def adjust_learning_rate(optimizer, epoch: int, frequency: int = 1, adjustment_factor: float = 0.1):
    """
    Adjusts the learning rate for the student_model's parameter group.

    The learning rate is adjusted every `frequency` epoch by `adjustment_factor`.
    """
    if (epoch + 1) % frequency == 0:
        # The student_model's parameters are in the first param_group, index 0
        optimizer.param_groups[0]['lr'] *= adjustment_factor

def create_lr_scheduler(
    optimizer: Optimizer,
    decay_epochs_list: list = None,
    decay_frequency: int = None,
    decay_factor: float = None
):
    """
    Creates a learning rate scheduler.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        decay_epochs_list: A list of epochs at which to decay the learning rate.
        decay_frequency: The frequency in epochs at which to decay the learning rate.
        decay_factor: The factor by which to decay the learning rate.

    Returns:
        A PyTorch learning rate scheduler.
    """
    if decay_epochs_list is not None and decay_factor is not None:
        return MultiStepLR(optimizer, milestones=decay_epochs_list, gamma=decay_factor)
    elif decay_frequency is not None and decay_factor is not None:
        return StepLR(optimizer, step_size=decay_frequency, gamma=decay_factor)
    else:
        raise ValueError(f"Incorrect inputs. Please provide either 'decay_epochs_list': {decay_epochs_list} or 'decay_frequency': {decay_frequency} along with 'decay_factor': {decay_factor}.")
    
class AverageMetricsDict(object):
    """
    Computes and stores the accumulated values and counts for a dictionary of metrics.
    Handles both Tensors (converts to float) and standard floats.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, metrics):
        """
        Args:
            dict: A dictionary where values are floats or Tensors.
        """
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if math.isnan(v):
                print(f"[rank {dist.get_rank()}] Warning: invalid value ({v}) for key '{k}', skipping.")
                continue
            self.sum[k] += v
            self.count[k] += 1

    def mean(self):
        """Returns a dictionary of averages."""
        return {k: self.sum[k] / self.count[k] for k in self.sum if self.count[k] > 0}
    
def reduce_metrics(metrics_dict, device):
    """
    Aggregates local averages across all GPUs using dist.all_reduce.
    
    Args:
        metrics_dict: Dictionary of local averages (floats).
        device: Current torch device.
    Returns:
        Dictionary of globally averaged metrics.
    """

    # Sort keys to ensure consistent order across devices
    keys = sorted(metrics_dict.keys())
    values = [metrics_dict[k] for k in keys]

    # Stack into a single tensor for efficient communication
    tensor = torch.tensor(values, device=device, dtype=torch.float32)

    # Average across devices using DDP
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    
    # Map back to dictionary
    reduced_dict = {k: v.item() for k, v in zip(keys, tensor)}

    return reduced_dict

METRIC_CONFIG = {
    # Key: (Log Name, Scale Factor, Data Type)
    # Use scale factor 100.0 for percentages.

    # --- Training Metrics ---
    "total_loss":               ("Average Total Training Loss", 1.0, float),
    "train_step_time_ms":       ("Average Training Step Time (ms)", 1.0, int),
    "model_loss":               ("Average Model Loss", 1.0, float),
    #"stud_teach_disp_loss":     ("Average Distillation Disparity Loss", 1.0, float),
    #"cost_volume_loss":         ("Average Distillation Cost Volume Loss", 1.0, float),
    #"stem_loss":                ("Average Distillation Stem Feature Loss", 1.0, float),
    #"initial_disparity_loss":   ("Average Distillation Initial Disparity Loss", 1.0, float),
    #"attention_volume_loss":    ("Average Distillation Attention Volume Loss", 1.0, float),
    #"feature_loss":             ("Average Distillation Feature Loss", 1.0, float),
    #"concatenation_volume_loss":("Average Distillation Concatenation Volume Loss", 1.0, float),

    # --- Validation Metrics ---
    "val_loss":             ("Average Validation Loss", 1.0, float),
    "val_epe_192_final_disp":("Average Validation EPE", 1.0, float),
    "val_epe_416_final_disp":("Average Validation EPE 416", 1.0, float),
    "d1":                   ("Average Validation D1 (%)", 100.0, float),
    "treshold_3px":         ("Average Validation 3px (%)", 100.0, float),
    "treshold_2px":         ("Average Validation 2px (%)", 100.0, float),
    "treshold_1px":         ("Average Validation 1px (%)", 100.0, float),
    "inference_time_ms":    ("Average Inference time (ms)", 1.0, int),
    "epe_disp_416":    ("Average Validation EPE Disp Mask 416", 1.0, float),
    "epe_disp_768":    ("Average Validation EPE Disp Mask 768", 1.0, float),
}

def format_log_dict(metrics):
    """
    Maps raw metric dictionary to a pretty logging dictionary.
    Handles scaling and datatype casting automatically based on config.
    """
    log_dict = {}

    for key, value in metrics.items():
        if key in METRIC_CONFIG:
            name, scale, dtype = METRIC_CONFIG[key]
            log_dict[name] = dtype(value * scale)
        else:
            # Fallback for metrics not in config
            log_dict[key] = value

    return log_dict

def save_checkpoint(root_dir, model, optimizer, epoch, epe,  is_best, run=None):
    """Save checkpoint after every epoch and copy best checkpoints to additional directory."""

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
                'state_dict': model.state_dict(),
                'epoch': epoch+1,
                'epe': epe,
            }

    checkpoint_path = root_dir / f"checkpoint-epoch-{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    if is_best:
        best_model_path = root_dir / "best_epe.pth"
        torch.save(checkpoint, best_model_path)

        if run: #NOTE for the time being only upload best-epe checkpoints to wandb
            # Create an artifact
            artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                type='model',
                metadata={'epoch': epoch+1, 'epe': epe}
            )
            # Add the checkpoint file to the artifact
            artifact.add_file(str(checkpoint_path))
            
            # Define aliases for easy reference
            aliases = ['latest']
            if is_best:
                aliases.append('best')
                
            # Log the artifact to W&B
            run.log_artifact(artifact, aliases=aliases)

def save_checkpoint_general(root_dir: str, model, epoch: int, metric_name: str, metric_value: float, is_best: bool, run=None, args=None):
    """Save checkpoint after every epoch and copy best checkpoints to additional directory."""

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
                'state_dict': model.state_dict(),
                'epoch': epoch+1,
                metric_name: metric_value,
            }
    
    if args is not None and hasattr(args, "run_name"):
        run_name = args.run_name
        checkpoint["run_name"] = run_name
        checkpoint_path = root_dir / f"ckpt_{run_name}_{epoch+1}.pth"
    else:
        checkpoint_path = root_dir / f"checkpoint-epoch-{epoch+1}.pth"

    torch.save(checkpoint, checkpoint_path)
    if is_best:
        checkpoint_path = root_dir / f"ckpt_{run_name}_{metric_name}_best.pth"
        torch.save(checkpoint, checkpoint_path)
        if run: #NOTE for the time being only upload best-epe checkpoints to wandb
            # Create an artifact
            artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                type='model',
                metadata={'epoch': epoch+1, metric_name: metric_value}
            )
            # Add the checkpoint file to the artifact
            artifact.add_file(str(checkpoint_path))
            
            # Define aliases for easy reference
            aliases = ['latest']
            if is_best:
                aliases.append('best')
                
            # Log the artifact to W&B
            run.log_artifact(artifact, aliases=aliases)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads checkpoint to resume training."""

    if checkpoint_path is None:
        raise ValueError(f"--load_ckpt_path is {checkpoint_path}. Failed to load checkpoint.")
    
    if not Path(checkpoint_path).exists():
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    return model, optimizer, checkpoint['epoch'], checkpoint['epe']

def save_checkpoint_metric(root_dir, model, global_step, args):
    """Save checkpoint after every epoch and copy best checkpoints to additional directory."""

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
                'state_dict': model.state_dict(),
                'train_dataset': "SceneFlow",
                'global_step': global_step,
            }
    checkpoint_path = root_dir / f"{args.wandb_run_name}_{global_step}.pth"
    torch.save(checkpoint, checkpoint_path)
    
def log_train_metrics(metrics, global_step, lr, args, run):
    averages = metrics.mean()
    metrics.reset()
    averages = reduce_metrics(averages, args.device)
    averages = format_log_dict(averages)
    # We do not want to log metrics to console every time.
    if args.rank == 0 and (global_step % args.val_frequency == 0):
        print(f"--- Training {global_step}/{args.total_steps} ---")
        for key, value in averages.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

    if run is not None:
        # Log training metrics immediately
        log_dict = {**averages}
        if len(lr) == 1:
            log_dict["Learning Rate"] = lr[0]
        elif len(lr) == 3:
            log_dict["Learning Rate Feature Extraction"] = lr[0]
            log_dict["Learning Rate Cost Filtering"] = lr[1]
            log_dict["Learning Rate Disparity Refinement"] = lr[2]

        run.log(log_dict, step=global_step)

def log_val_metrics(dataset_name, metrics, global_step, args, run):
    averages = metrics.mean()
    metrics.reset()
    averages = reduce_metrics(averages, args.device)
    averages = format_log_dict(averages)
    if args.rank == 0:
        print(f"--- Validation {dataset_name} ---")
        for key, value in averages.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

    if run is not None:
        # Log training metrics immediately
        log_dict = {**averages}
        run.log(log_dict, step=global_step)

    return averages

def log_val_metrics_save_ckpt(model, optimizer, dataset_name, metrics, global_step, args, run):
    averages = metrics.mean()
    metrics.reset()
    averages = reduce_metrics(averages, args.device)
    averages = format_log_dict(averages)
    if args.rank == 0:
        print(f"--- Validation {dataset_name} ---")
        for key, value in averages.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

    if run is not None:
        # Log training metrics immediately
        log_dict = {**averages}
        run.log(log_dict, step=global_step)

    if args.rank == 0:
        if dataset_name == "SceneFlow":
            avg_epe = averages[f"{dataset_name} Validation EPE (<192)"]
            if avg_epe < args.sceneflow_best_epe:
                save_checkpoint_metric(args.save_ckpt_path, model, optimizer, "SceneFlow", global_step, "EPE", avg_epe, True, run)
                print(f"New best SceneFlow Validation EPE: {avg_epe:.4f}, former best Validation EPE: {args.sceneflow_best_epe:.4f}. Checkpoint saved to dir {args.save_ckpt_path}.")
                args.sceneflow_best_epe = avg_epe
            else:
                print(f"Current SceneFlow Validation EPE: {avg_epe:.4f}, best Validation EPE: {args.sceneflow_best_epe:.4f}.")
            # Logic for "Regular" checkpoint (every val_frequency)
            save_checkpoint_metric(args.save_ckpt_path, model, optimizer, "SceneFlow", global_step, "EPE", avg_epe, False, run)
            print(f"Saving regular checkpoint to dir {args.save_ckpt_path}.")
                
        if dataset_name == "Kitti 2012":
            avg_d1 = averages[f"Kitti 2012 Validation D1 Occ (%)"]
            if avg_d1 < args.kitti2012_best_d1:
                save_checkpoint_metric(args.save_ckpt_path, model, optimizer, "Kitti 2012", global_step, "d1", avg_d1, True, run)
                print(f"New best Kitti 2012 Validation D1 Occ Error: {avg_d1:.4f} (%), former best Validation D1 Occ Error: {args.kitti2012_best_d1:.4f}. Checkpoint saved to dir {args.save_ckpt_path}.")
                args.kitti2012_best_d1 = avg_d1
            else:
                print(f"Current Kitti 2012 Validation D1 Occ Error: {avg_d1:.4f}, best Validation D1 Occ Error: {args.kitti2012_best_d1:.4f}.")

        if dataset_name == "Kitti 2015":
            avg_d1 = averages[f"Kitti 2015 Validation D1 Occ (%)"]
            if avg_d1 < args.kitti2015_best_d1:
                save_checkpoint_metric(args.save_ckpt_path, model, optimizer, "Kitti 2015", global_step, "d1", avg_d1, True, run)
                print(f"New best Kitti 2015 Occ Validation D1 Error: {avg_d1:.4f}, former best Occ Validation D1 Error: {args.kitti2015_best_d1:.4f}. Checkpoint saved to dir {args.save_ckpt_path}.")
                args.kitti2015_best_d1 = avg_d1
            else:
                print(f"Current Kitti 2015 Occ Validation D1 Error: {avg_d1:.4f}, best Occ Validation D1 Error: {args.kitti2015_best_d1:.4f}.")
        
        print(f"--- Validation {dataset_name} Complete ---")
        