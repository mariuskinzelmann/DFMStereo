# External Imports
import os
import sys
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import uuid
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# Internal Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.dataset import *
from dataset.transform import *
from utils.loss import *
from utils.distillation_tools import *
from utils.experiment import *
from utils.metrics import *
from utils.validation import *
from utils.optimizer import *

# Students:
from models.DFMStereo.core.dfmstereo_large import init_dfmstereo_large
from models.DFMStereo.core.dfmstereo_medium import init_dfmstereo_medium
from models.DFMStereo.core.dfmstereo_small import init_dfmstereo_small

# Teachers:
from models.FoundationStereo.core.foundation_stereo import init_foundation_stereo
from models.FastFoundationStereo.core.foundation_stereo import init_fast_foundation_stereo
from models.FoundationStereo.core.submodule import disparity_regression

from torch.amp import GradScaler

def setup_ddp_slurm(rank, world_size):
    """Initialise the DDP process group using Slurm environment variables."""
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp_slurm():
    dist.destroy_process_group()

def setup_wandb(args):
    """Initialise Weights and Biases for logging information."""

    #if args.wandb_run_name is not None:
    #    base_name = f"{args.wandb_run_name}"
    #    api = wandb.Api()
    #    project_path = f"{args.entity}/{args.project}" if hasattr(args, "entity") and hasattr(args, "project") else "Distilling-Foundation-Stereo/Distilling-Foundation-Stereo"

    #    existing_runs = [
    #        run.name for run in api.runs(
    #            project_path,
    #            filters={"display_name": {"$regex": f"^{base_name}_"}}
    #        )
    #    ]

    #    counter = 0
    #    run_name = f"{base_name}_{counter:02d}"    
    #    while run_name in existing_runs:
    #        counter += 1
    #        run_name = f"{base_name}_{counter:02d}"
    #else:

    run_name = f"training_{uuid.uuid4().hex[:4]}"

    run = wandb.init(
        project="project_name",
        entity="project_name",
        name=run_name,
        group=args.wandb_group if args.wandb_group else f"training",
        job_type="Knowledge Distillation" if args and hasattr(args, "being_taught") else "Training",
        config = {
            "Model Architecture": {
            "Model": args.model,
            "Feature Backbone": 'edgenext',
            "corr_groups": args.corr_groups,
            "concat_groups": args.concat_groups,
            "maxdisp": args.maxdisp,

            # Update Block / GRU
            "GRU": {
                "Num GRU Layers": args.n_gru_layers,
                "Downsample Factor": args.n_downsample,
                "Hidden Dimension": args.hidden_dim,
                "context_dim": args.context_dim,
            },
            },

            "Mixed Precision": {
                "use_mixed_precision": args.mixed_precision,
                "precision_dtype": args.precision_dtype,
            },

            "Knowledge Distillation": {
                "cost_volume_loss": 'mse',
                "agg_cost_volume_loss": 'mse',
                "feature_loss": 'mse',
                "logit_loss": 'hinton_logit_distillation',
                "refined_disparity_loss": 'smoothl1',
                

                "Being Taught": args.being_taught,
                "Teacher": args.teacher,

                "factor_transfer": args.factor_transfer,
                "feature_paraphraser_ckpt": args.feature_paraphraser_ckpt,
                "features": args.features,
                "feature_weights": args.feature_weights,

                "cost_volume": args.cost_volume,
                "cost_volume_weight": args.cost_volume_weight,
                "agg_cost_volume": args.agg_cost_volume,
                "agg_cost_volume_weight": args.agg_cost_volume_weight,

                "init_disp_logits": args.init_disp_logits,
                "logits_hard_kl_div": args.logits_hard_kl_div,
                "logits_soft_kl_div": args.logits_soft_kl_div,
                "hard_kl_div_weight": args.hard_kl_div_weight,
                "soft_kl_div_weight": args.soft_kl_div_weight,
                "temperature": args.temperature,

                "update_block_iters": args.update_block_iters,
                "refined_disp": args.refined_disp,
                "refined_disp_weight": args.refined_disp_weight,
            },


            "Foundation Stereo": {
                "Checkpoint": "23-51-11",
                "Max Disparity": 416,
                "Valid Iters": args.foundation_stereo_valid_iters if args and hasattr(args, "foundation_stereo_valid_iters") else 32,
            },

            "Loss Functions": {
                "Model Loss Function": args.model_loss_fn,
           },

            "Dataset": {
                "Dataset": args.dataset,
            },

            "Training Settings": {
                "Batch Size": args.batch_size,
                "Patch Size": args.patch_size,
                "Train Iters": args.train_iters,
            },

            "Validation Settings": {
                "val_frequency": args.val_frequency,
                "val_batch_size": args.val_batch_size,
                "valid_iters": args.valid_iters,
            },

            "Optimizer & LR Scheduler": {
                "LR Scheduler": "OneCycleLR",
                "Anneal Strategy": "Cosine",
                "pct_start": args.pct_start,
                "div_factor": args.div_factor,
                "final_div_factor": args.final_div_factor,                
                "Total Steps": args.total_steps,
                "Early Stop": args.early_stop,
                "Learning Rate": args.learning_rate,
                "Weight Decay": args.weight_decay,
                "edge_next_weight_decay": args.edge_next_weight_decay,
                "edge_next_learning_rate": args.edge_next_learning_rate,
            },

            "Data Augmentations": {
                'gamma': args.img_gamma,
                'saturation_range': args.saturation_range,
                'spatial_scale': True,
                'min_scale': args.spatial_scale[0],
                'max_scale': args.spatial_scale[1],
                'yjitter': not args.noyjitter
            },

        }
    )
    
    return run


def train(args, run=None):
    """"The main training loop."""

    # --- Data Augmentations for Training ---
    aug_params = {
        'crop_size': args.patch_size,
        'gamma': args.img_gamma,
        'saturation_range': args.saturation_range,
        'spatial_scale': True,
        'min_scale': args.spatial_scale[0],
        'max_scale': args.spatial_scale[1],
        'yjitter': not args.noyjitter
        }

    # --- Data Loading ---
    if args.dataset == "ALL":
        sceneflow = SceneFlow(args.dataset_path_sceneflow_train, aug_params, mode ='training')
        fsd = FSD(args.dataset_path_fsd, aug_params, mode='training')
        train_dataset = ConcatDataset([sceneflow, fsd])
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    elif args.dataset == "SceneFlow":
        train_dataset = SceneFlow(args.dataset_path_sceneflow_train, aug_params, mode ='training')
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    elif args.dataset == "FSD":
        train_dataset = FSD(args.dataset_path_fsd, aug_params, mode='training')
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    if args.validate_sceneflow:
        val_sceneflow = SceneFlow(args.dataset_path_sceneflow_test, aug_params=None, mode ='testing')
        val_sampler_sceneflow = DistributedSampler(dataset=val_sceneflow, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_sceneflow = DataLoader(dataset=val_sceneflow, batch_size=4, shuffle=False, sampler=val_sampler_sceneflow, num_workers=1, pin_memory=True, drop_last=False)
        args.sceneflow_best_epe = float('inf')

    if args.validate_kitti2012:
        kitti12 = KITTI12(args.dataset_path_kitti2012, aug_params=None, mode='training', occ=False)
        val_sampler_kitti2012 = DistributedSampler(kitti12, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_kitti2012 = DataLoader(kitti12, batch_size=1, shuffle=False, sampler=val_sampler_kitti2012, num_workers=1, pin_memory=True, drop_last=False)
        args.kitti2012_best_d1 = float('inf')

    if args.validate_kitti2015:
        kitti15 = KITTI15(args.dataset_path_kitti2015, aug_params=None, mode='training', occ=False)
        val_sampler_kitti2015 = DistributedSampler(kitti15, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_kitti2015 = DataLoader(kitti15, batch_size=1, shuffle=False, sampler=val_sampler_kitti2015, num_workers=1, pin_memory=True, drop_last=False)
        args.kitti2015_best_d1 = float('inf')
    
    if args.validate_middlebury_q:
        middlebury_q = Middlebury(args.dataset_path_middlebury, aug_params=None, split='MiddEval3', mode='training', resolution='Q', load_mask=True)
        val_sampler_middlebury_q = DistributedSampler(middlebury_q, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_middlebury_q = DataLoader(middlebury_q, batch_size=1, shuffle=False, sampler=val_sampler_middlebury_q, num_workers=1, pin_memory=True, drop_last=False)
        args.middlebury_q_best_bp3 = float('inf')

    if args.validate_middlebury_h:
        middlebury_h = Middlebury(args.dataset_path_middlebury, aug_params=None, split='MiddEval3', mode='training', resolution='H', load_mask=True)
        val_sampler_middlebury_h = DistributedSampler(middlebury_h, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_middlebury_h = DataLoader(middlebury_h, batch_size=1, shuffle=False, sampler=val_sampler_middlebury_h, num_workers=1, pin_memory=True, drop_last=False)
        args.middlebury_h_best_bp3 = float('inf')

    if args.validate_eth3d:
        eth3d = ETH3D(args.dataset_path_eth3d, aug_params=None, mode='training', load_mask=True)
        val_sampler_eth3d = DistributedSampler(eth3d, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_eth3d = DataLoader(eth3d, batch_size=1, shuffle=False, sampler=val_sampler_eth3d, num_workers=1, pin_memory=True, drop_last=False)
        args.eth3d_best_bp3 = float('inf')

    if args.validate_boosterq:
        boosterq = BoosterQ(args.dataset_path_boosterq, aug_params=None, mode='train', setup='balanced', resolution='Q', load_mask=True)
        val_sampler_boosterq = DistributedSampler(boosterq, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader_boosterq = DataLoader(boosterq, batch_size=1, shuffle=False, sampler=val_sampler_boosterq, num_workers=1, pin_memory=True, drop_last=False)
        args.boosterq_best_epe = float('inf')

    # --- Model Initialisation ---
    if args.model == 'dfmstereo_large':
        model = init_dfmstereo_large(args, eval=False)
    elif args.model == 'dfmstereo_medium':
        model = init_dfmstereo_medium(args, eval=False)
    elif args.model == 'dfmstereo_small':
        model = init_dfmstereo_small(args, eval=False)

    for p in model.parameters():
        p.requires_grad = True

    if args.world_size > 1:
        print(f"world_size: {args.world_size} > 1")
        print(f"Wrapping BatchNorm in SyncBatchNorm.")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(module=model, device_ids=[args.local_rank])
    model.module.train()
    model.module.freeze_bn()
    optimizer, lr_scheduler = fetch_optimizer_scheduler(model.module, args)
    
    # --- Teacher Initialisation ---
    if args.being_taught:
        if args.teacher == "foundation_stereo":
            teacher = init_foundation_stereo(args, use_gpu=True)
            if args.features and args.factor_transfer:
                print(f"Loading Feature Paraphraser Weights from {args.feature_paraphraser_ckpt}...")
                ckpt_path = Path(__file__).parent.parent.parent / "models" / "FoundationStereo" / "pretrained" / "23-51-11" / "feature_paraphraser" / args.feature_paraphraser_ckpt
                paraphraser_ckpt = torch.load(ckpt_path, map_location=args.device)
                teacher.feature_paraphraser.load_state_dict(paraphraser_ckpt['state_dict'])
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
                
        if args.teacher == "fast_foundation_stereo":
            teacher = init_fast_foundation_stereo(args)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
    else:
        teacher = None

    epoch = 0
    global_step = 0
    train_metrics = AverageMetricsDict()
    scaler = GradScaler("cuda", enabled=args.mixed_precision)

    # --- Training Loop ---
    if args.rank == 0:
        print(f"--- Starting Training for {args.total_steps} iterations ---")

    keep_training = True
    
    while keep_training:
        train_sampler.set_epoch(epoch)
        dist.barrier()

        for batch_idx, batch in enumerate(train_loader):
            # --- Training Step ---
            torch.cuda.synchronize()
            start_time = time.time()
            train_step_metrics = train_batch(model, teacher, batch, optimizer, lr_scheduler, scaler, args)
            torch.cuda.synchronize()
            end_time = time.time()

            # --- Timing Logic ---
            train_step_time_seconds = end_time - start_time
            train_step_time_ms = int(train_step_time_seconds * 1000)
            train_step_metrics["train_step_time_ms"] = train_step_time_ms

            train_metrics.update(train_step_metrics)
            del batch, train_step_metrics

            global_step += 1

            stop_training = (
                global_step >= args.total_steps or
                (args.early_stop is not None and global_step >= args.early_stop)
                )

            # --- Logging ----
            if global_step % args.log_frequency == 0:
                dist.barrier()
                log_train_metrics(train_metrics, global_step, [optimizer.param_groups[0]['lr']], args, run)

            # --- Validation ---
            should_validate = (
                global_step % args.val_frequency == 0 or
                stop_training
            )
            if should_validate:
                model.eval()
                if args.rank == 0:
                    print(f"--- Pausing Training ---")
                    print(f"--- Current Step: {global_step}/{args.total_steps}---")
                dist.barrier()
                
                if args.validate_sceneflow:
                    validate_sceneflow(model, val_loader_sceneflow, global_step, args, run)
                if args.validate_kitti2012:
                    validate_kitti2012(model, val_loader_kitti2012, global_step, args, run)
                if args.validate_kitti2015:
                    validate_kitti2015(model, val_loader_kitti2015, global_step, args, run)
                if args.validate_middlebury_q:
                    validate_middlebury_q(model, val_loader_middlebury_q, global_step, args, run)
                if args.validate_middlebury_h:
                    validate_middlebury_h(model, val_loader_middlebury_h, global_step, args, run)
                if args.validate_eth3d:
                    validate_eth3d(model, val_loader_eth3d, global_step, args, run)
                if args.validate_boosterq:
                    validate_boosterq(model, val_loader_boosterq, global_step, args, run)

                
                if args.rank == 0:
                    print(f"Saving Checkpoint at step {global_step} to '{args.save_ckpt_path}'.")
                save_checkpoint_metric(args.save_ckpt_path, model, global_step, args)

                dist.barrier()
                torch.cuda.empty_cache()

                model.train()
                model.module.freeze_bn() if isinstance(model, DDP) else model.freeze_bn()
                if args.rank == 0:
                    print(f"--- Resuming Training ---")
            
            if stop_training:
                if args.rank == 0:
                    print(f"Stopping training early at iteration {global_step}.")
                keep_training = False
                break
        epoch += 1

def train_batch(model, teacher, batch, optimizer, lr_scheduler, scaler, args):
    """Processes a single batch of training data."""

    left_image = batch['left_image'].to(args.device)
    right_image = batch['right_image'].to(args.device)
    disparity_gt = batch['disparity'].to(args.device)

    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        student_outputs = model(left_image, right_image, iters=args.train_iters, being_taught=args.being_taught)
        preds = student_outputs["init_disp_up"]
        iter_preds = student_outputs["disp_preds"]
        train_dict = {}

        model_loss = igev_sequence_loss_geo_vol(preds, iter_preds, disparity_gt, max_disp=args.maxdisp)
        
        train_dict["model_loss"] = model_loss.item()
        
        total_loss = model_loss

        if args.being_taught and teacher is not None:
            # Use Student Disparity Estimate for Update Block Distillation
            
            s_init_disp = student_outputs["init_disp"].detach() if args.use_student_init_disp else None
            
            # Teacher Inference:
            with torch.no_grad():
                if args.teacher == "foundation_stereo":
                    teacher_outputs = teacher(
                        left_image,
                        right_image,
                        iters=args.foundation_stereo_valid_iters,
                        test_mode=True,
                        teach_mode=True,
                        s_init_disp=s_init_disp,
                        )
                    
                elif args.teacher == "fast_foundation_stereo":
                    teacher_outputs = teacher(
                        left_image,
                        right_image,
                        iters=args.foundation_stereo_valid_iters,
                        test_mode=True,
                        teach_mode=True,
                        )
                
                
            if args.features:
                res = ['4x', '8x', '16x', '32x']
                weights = args.feature_weights
                for r, weight in zip(res, weights):
                    s_feat_left = student_outputs[f"feat_{r}_left"]
                    s_feat_right = student_outputs[f"feat_{r}_right"]
                    t_feat_left = teacher_outputs[f"feat_{r}_left"]
                    t_feat_right = teacher_outputs[f"feat_{r}_right"]

                    assert s_feat_left.shape == t_feat_left.shape, [s_feat_left.shape, t_feat_left.shape]

                    loss_left = F.mse_loss(s_feat_left, t_feat_left, reduction='mean')
                    loss_right = F.mse_loss(s_feat_right, t_feat_right, reduction='mean')
                    loss = 0.5 * (loss_left + loss_right)
                    train_dict[f"feat_loss_{r}"] = loss.item()
                    loss = weight * loss
                    train_dict[f"feat_loss_{r}_x_{weight}"] = loss.item()
                    total_loss = total_loss + loss

            if args.init_disp_logits:
                s_logits = student_outputs.pop("init_disp_logits")
                t_logits = teacher_outputs.pop("init_disp_logits").detach()

                with torch.no_grad():
                    t_probs = F.softmax(t_logits, dim=1)
                    t_init_disp = disparity_regression(t_probs, teacher.args.max_disp//4)
                    distillation_mask = (t_init_disp < model.module.args.max_disp//4).bool().squeeze(1) # [B, H//4, W//4]
                    del t_probs, t_init_disp

                t_logits = t_logits[:,0:48,:,:] # Truncation: [B, 48, H//4, W//4]

                # This code was used to additionally supervise the effect of 
                # distillation on the initial disparity prediciton. Uncomment if needed.
                #
                #with torch.no_grad():
                #    # --- Initial Disparity Estimation EPE (Student/Teacher) ---
                #    s_prob = F.softmax(s_logits, dim=1)
                #    t_prob = F.softmax(t_logits, dim=1)
                #
                #    s_init_disp = disparity_regression(s_prob, maxdisp=48)
                #    t_init_disp = disparity_regression(t_prob, maxdisp=48)
                #
                #    s_init_disp = s_init_disp.squeeze(1)
                #    t_init_disp = t_init_disp.squeeze(1)
                #
                #    for range_min, range_max in [(0,16), (16,32), (32,48)]:
                #        range_mask = distillation_mask & (t_init_disp >= range_min) & (t_init_disp < range_max)
                #        epe_range = (s_init_disp - t_init_disp).abs()[range_mask].mean()
                #        train_dict[f"init_disp_epe_{range_min}_{range_max}"] = epe_range.item()
                #
                #    init_disp_epe = (s_init_disp - t_init_disp).abs()
                #    train_dict[f"init_disp_epe"] = init_disp_epe[distillation_mask].mean().item()
                #    del s_prob, t_prob, s_init_disp, t_init_disp

                # Hard Targets:
                if args.logits_hard_kl_div:
                    s_log_prob = F.log_softmax(s_logits, dim=1)
                    t_log_prob = F.log_softmax(t_logits, dim=1)
                    kl_div_hard = F.kl_div(s_log_prob, t_log_prob, reduction='none', log_target=True)
                    kl_div_hard = torch.sum(kl_div_hard, dim=1)
                    kl_div_hard = kl_div_hard[distillation_mask].mean() if not args.teacher == "fast_foundation_stereo" else kl_div_hard.mean()
                    train_dict["logits_hard_kl_div"] = kl_div_hard.item()
                    kl_div_hard = kl_div_hard * args.hard_kl_div_weight
                    train_dict["logits_hard_kl_div_scaled"] = kl_div_hard.item()
                    total_loss = total_loss + kl_div_hard

                # Soft Targets:
                if args.logits_soft_kl_div:
                    s_log_prob = F.log_softmax(s_logits / args.temperature, dim=1)
                    t_log_prob = F.log_softmax(t_logits / args.temperature, dim=1)
                    kl_div_soft = F.kl_div(s_log_prob, t_log_prob, reduction='none', log_target=True)
                    kl_div_soft = torch.sum(kl_div_soft, dim=1)
                    kl_div_soft = kl_div_soft[distillation_mask].mean() * (args.temperature ** 2) if not args.teacher == "fast_foundation_stereo" else kl_div_soft.mean() * (args.temperature ** 2)
                    train_dict["logits_soft_kl_div"] = kl_div_soft.item()
                    kl_div_soft = kl_div_soft * args.soft_kl_div_weight
                    train_dict["logits_soft_kl_div_scaled"] = kl_div_soft.item()
                    total_loss = total_loss + kl_div_soft

            if args.cost_volume:
                s_cv = student_outputs.pop("cost_volume") # Student Shape: [B, 28, 48, H//4, W//4]
                t_cv = teacher_outputs.pop("cost_volume").detach() # Teacher Shape: [B, 28, 48, H//4, W//4] (Truncated)
                
                assert s_cv.shape == t_cv.shape, [s_cv.shape, t_cv.shape]

                # --- MSE Loss ---
                mse_loss_cv = F.mse_loss(s_cv, t_cv, reduction='none')
                B, C, D, H, W = mse_loss_cv.shape
                if not args.teacher == "fast_foundation_stereo":
                    mask_cv = distillation_mask.view(B, 1, 1, H, W).expand_as(mse_loss_cv) # Distillation Mask
                    mse_loss_cv = mse_loss_cv[mask_cv].mean() 
                else:
                    mse_loss_cv = mse_loss_cv.mean()

                train_dict["mse_loss_cv"] = mse_loss_cv.item()
                mse_loss_cv = mse_loss_cv * args.cost_volume_weight
                train_dict[f"mse_loss_cv_x_{args.cost_volume_weight}"] = mse_loss_cv.item()

                # --- Training Loss Contribution ---
                total_loss = total_loss + mse_loss_cv
                    
            if args.agg_cost_volume:
                
                # --- Final aggregated Cost Volume ---
                s_agg_cost_volume = student_outputs["agg_cost_volume"]
                t_agg_cost_volume = teacher_outputs["agg_cost_volume"].detach() # Teacher Shape: [B, 28, 48, H//4, W//4] (Truncated)

                if args.teacher == 'foundation_stereo':
                    agg_cost_volume_loss = F.mse_loss(s_agg_cost_volume, t_agg_cost_volume, reduction='none')
                    B, C, D, H, W = agg_cost_volume_loss.shape
                    mask_agg_cv = distillation_mask.view(B, 1, 1, H, W).expand_as(agg_cost_volume_loss) # Distillation Mask
                    agg_cost_volume_loss = agg_cost_volume_loss[mask_agg_cv].mean()
                elif args.teacher == 'fast_foundation_stereo':
                    agg_cost_volume_loss = F.mse_loss(s_agg_cost_volume, t_agg_cost_volume, reduction='mean')
                    
                train_dict["agg_cost_volume"] = agg_cost_volume_loss.item()
                agg_cost_volume_loss = agg_cost_volume_loss * args.agg_cost_volume_weight
                train_dict[f"agg_cost_volume_x_{args.agg_cost_volume_weight}"] = agg_cost_volume_loss.item()

                total_loss = total_loss + agg_cost_volume_loss

            if len(args.update_block_iters) > 0:
                for itr in args.update_block_iters:
                    
                    distillation_mask = None
                    
                    if args.refined_disp:
                        s_refined_disp = student_outputs[f"iter_{itr}_refined_disp"]
                        t_refined_disp = teacher_outputs[f"iter_{itr}_refined_disp"]

                        if not args.teacher == "fast_foundation_stereo":
                            distillation_mask = t_refined_disp < 48.0
                            loss = F.smooth_l1_loss(s_refined_disp[distillation_mask], t_refined_disp[distillation_mask], reduction='mean', beta=0.25)
                        else:
                            loss = F.smooth_l1_loss(s_refined_disp, t_refined_disp, reduction='mean', beta=0.25)

                        train_dict[f"iter_{itr}_refined_disp_loss"] = loss.item()
                        loss = loss * args.refined_disp_weight
                        train_dict[f"iter_{itr}_refined_disp_loss_x_{args.refined_disp_weight}"] = loss.item()

                        total_loss = total_loss + loss

    train_dict["total_loss"] = total_loss.item()

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    prev_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()

    # Only step OneCycleLR if optimizer.step() really happened
    if scaler.get_scale() >= prev_scale:
        lr_scheduler.step()

    return train_dict
 
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge Distillation for Stereo Vision.')
    # --- Model Choice ---
    parser.add_argument('--model', type=str, default='dfmstereo_large', choices=['dfmstereo_large', 'dfmstereo_medium', 'dfmstereo_small'])
    parser.add_argument('--dfmstereo_ckpt', type=str, default=None)

    # --- Mixed Precision Settings ---
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')

    parser.add_argument('--maxdisp', type=int, default=192)

    # --- Knowledge Distillation ---
    parser.add_argument('--being_taught', action='store_true', help='Set flag to use Knowledge Distillation.')
    parser.add_argument('--teacher', type=str, default='foundation_stereo', choices=['foundation_stereo', 'fast_foundation_stereo'])
    parser.add_argument('--foundation_stereo_valid_iters', type=int, default=8, help='Amount of Validation Iterations for FoundationStereo Teacher.')
    
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--feature_weights', type=float, nargs=4, default=[4.0, 1.0, 0.5, 0.25])
    parser.add_argument('--factor_transfer', action='store_true')
    parser.add_argument('--feature_paraphraser_ckpt', type=str, default='conv_3x3_conv_1x1_conv_3x3_120000.pth', help='CKPT File of Feature Paraphraser.')

    parser.add_argument('--distillation_mask_cv', action='store_true')

    parser.add_argument('--cost_volume', action='store_true', help='Set flag to use Knowledge Distillation.')
    parser.add_argument('--cost_volume_weight', type=float, default=1.0)

    parser.add_argument('--agg_cost_volume', action='store_true', help='Set flag to use Knowledge Distillation.')
    parser.add_argument('--agg_cost_volume_weight', type=float, default=1.0)

    parser.add_argument('--init_disp_logits', action='store_true', help='Set flag to distill the Logits for Initial Disparity Regression.')
    parser.add_argument('--logits_hard_kl_div', action='store_true')
    parser.add_argument('--logits_soft_kl_div', action='store_true')
    parser.add_argument('--hard_kl_div_weight', type=float, default=1.0)
    parser.add_argument('--soft_kl_div_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=4.0, help='Softmax Temperature.')

    parser.add_argument('--update_block_iters', type=int, nargs='+', default=[])
    parser.add_argument('--refined_disp', action='store_true')
    parser.add_argument('--refined_disp_weight', type=float, default=1.0)
    parser.add_argument('--disparity_refinement_mask', action='store_true')

    # --- Datasets ---
    parser.add_argument('--dataset', type=str, default='SceneFlow', choices=["SceneFlow", "KITTI", "ALL", "FSD"], help='Choose training dataset.')
    parser.add_argument('--dataset_path_sceneflow_train', type=str, default=None, help='Path to train dataset.')
    parser.add_argument('--dataset_path_sceneflow_test', type=str, default=None, help='Path to test dataset.')
    parser.add_argument('--dataset_path_fsd', type=str, default=None, help='Path to FSD Dataset.')
    parser.add_argument('--validate_sceneflow', action='store_true')
    parser.add_argument('--validate_kitti2012', action='store_true')
    parser.add_argument('--validate_kitti2015', action='store_true')
    parser.add_argument('--kitti_version', type=str, default='2012')
    parser.add_argument('--validate_middlebury_q', action='store_true')
    parser.add_argument('--validate_middlebury_h', action='store_true')
    parser.add_argument('--validate_eth3d', action='store_true')
    parser.add_argument('--validate_boosterq', action='store_true')
    parser.add_argument('--dataset_path_kitti2012', type=str, default=None, help='Path to KITTI2012.')
    parser.add_argument('--dataset_path_kitti2015', type=str, default=None, help='Path to KITTI2012.')
    parser.add_argument('--dataset_path_middlebury', type=str, default=None, help='Path to Middlebury.')
    parser.add_argument('--dataset_path_eth3d', type=str, default=None, help='Path to ETH3D.')
    parser.add_argument('--dataset_path_boosterq', type=str, default=None, help='Path to BoosterQ.')
    
    # --- Data Augmentation ---
    parser.add_argument('--patch_size', nargs=2, type=int, default=[256, 736], help='Height and width of image patches for training.')
    parser.add_argument('--img_gamma', type=float, nargs='+', default=[1,1,1,1], help="Gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='Color saturation')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8], help='Rescale Images by 2^(spatial_scale).')
    parser.add_argument('--noyjitter', action='store_true', help='Turn off Jitter for perfect Rectification.')

    # --- Training Settings ---
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--train_iters', type=int, default=8, help='ConvGRU Iterations for Training.')
    parser.add_argument('--model_loss_fn', type=str, default='SequenceLoss', help='Model Loss Function used during Training.')
    
    # --- Validation Settings ---
    parser.add_argument('--val_frequency', type=int, default=5000, help='Validation Frequency.')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Validation Batch Size. Relevant for SceneFlow.')
    parser.add_argument('--valid_iters', type=int, default=8, help='ConvGRU Iterations for Validation.')

    # --- Optimizer and Learning Rate Schedule ---
    parser.add_argument('--total_steps', type=int, default=200000, help='Total amount of training iterations.')
    parser.add_argument('--early_stop', type=int, default=None, help='Stop training early at iteration x.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning Rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW Optimizer.')
    parser.add_argument('--edge_next_learning_rate', type=float, default=2e-4, help='Learning Rate for EdgeNext Feature Backbone.')
    parser.add_argument('--edge_next_weight_decay', type=float, default=1e-4, help='Weight Decay for EdgeNext Feature Backbone.')
    parser.add_argument('--div_factor', type=int, default=100, help='OnceCycleLR div_factor.')
    parser.add_argument('--final_div_factor', type=int, default=1000, help='OnceCycleLR final_div_factor.')
    parser.add_argument('--pct_start', type=float, default=0.01, help='Percentage of Total Steps used to warm-up LR.')

    # --- WandB and Checkpoint Saving ---
    parser.add_argument('--wandb', action='store_true', help='Logging metrics using wandb.')
    parser.add_argument('--wandb_group', type=str, default='Training', help='Assigns a group name to this run in wandb.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Name of the training run for easier identificaiton.')
    parser.add_argument('--save_ckpt_path', type=str, default=None, help='Save location for checkpoints.')
    parser.add_argument('--log_frequency', type=int, default=100, help='Logging Frequency for Training.')

    args = parser.parse_args()

    assert args.patch_size[0] % 32 == 0, f"Patch height must be divisible by 32, results in Conv error otherwise." 
    assert args.patch_size[1] % 32 == 0, f"Patch width must be divisible by 32, results in Conv error otherwise."
    
    try:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank 
    except KeyError:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size

    # --- DDP Setup ---
    setup_ddp_slurm(rank, world_size)

    # --- Setup Device ---
    DEVICE = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(DEVICE) # setting the default device for each process
    torch.cuda.reset_peak_memory_stats(DEVICE)
    args.device = DEVICE

    # --- WandB Setup ---
    # Uncomment this to use wandb
    #run = setup_wandb(args) if args.rank == 0 and args.wandb else None # Only initialise wandb on rank 0
    run = None

    # --- Training ---
    train(args, run)
    
    dist.barrier()

    if run is not None:
        run.finish()

    cleanup_ddp_slurm()
