# External Imports
import os
import sys
import time
from torch.nn.parallel import DistributedDataParallel as DDP

# Internal Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.dataset import *
from dataset.transform import *
from utils.loss import *
from utils.distillation_tools import *
from utils.experiment import *
from utils.metrics import *
from models.DFMStereo.core.utils.utils import InputPadder

def validate_sceneflow(model, val_loader, global_step, args, run):
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image = val_batch['left_image'].to(args.device)
            right_image = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)

            mask = (disparity_gt >= 0.5) & (disparity_gt < 192)
            mask_disp_416 = (disparity_gt >= 0.5) & (disparity_gt < 416)

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            epe = EPE(prediction, disparity_gt, mask)
            epe_disp_416 = EPE(prediction, disparity_gt, mask_disp_416)

            val_step_metrics = {
                f"SceneFlow Validation EPE (<192)": epe.item(),
                f"SceneFlow Validation EPE (<416)": epe_disp_416.item(),
                f"SceneFlow Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"SceneFlow Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("SceneFlow", val_metrics, global_step, args, run)


def validate_kitti2012(model, val_loader, global_step, args, run):
    
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image = val_batch['left_image'].to(args.device)
            right_image = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)
            #valid = val_batch['valid'].to(args.device).bool()

            mask = (disparity_gt >= 0.5) & (disparity_gt < 192) #& valid

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            bp1_occ = Threshold(prediction, disparity_gt, mask, threshold=1.0)
            bp2_occ = Threshold(prediction, disparity_gt, mask, threshold=2.0)
            bp3_occ = Threshold(prediction, disparity_gt, mask, threshold=3.0)
            d1_occ = D1(prediction, disparity_gt, mask)

            val_step_metrics = {
                f"Kitti 2012 Validation BP-1 Noc (%)": bp1_occ.item(),
                f"Kitti 2012 Validation BP-2 Noc (%)": bp2_occ.item(),
                f"Kitti 2012 Validation BP-3 Noc (%)": bp3_occ.item(),
                f"Kitti 2012 Validation D1 Noc (%)": d1_occ.item(),
                f"Kitti 2012 Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"Kitti 2012 Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("Kitti 2012", val_metrics, global_step, args, run)


def validate_kitti2015(model, val_loader, global_step, args, run):
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image = val_batch['left_image'].to(args.device)
            right_image = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)
            #valid = val_batch['valid'].to(args.device).bool()

            mask = (disparity_gt >= 0.5) & (disparity_gt < 192) #& valid

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            d1_occ = D1(prediction, disparity_gt, mask)
            bp1_occ = Threshold(prediction, disparity_gt, mask, threshold=1.0)
            bp2_occ = Threshold(prediction, disparity_gt, mask, threshold=2.0)
            bp3_occ = Threshold(prediction, disparity_gt, mask, threshold=3.0)

            val_step_metrics = {
                f"Kitti 2015 Validation BP-1 Noc (%)": bp1_occ.item(),
                f"Kitti 2015 Validation BP-2 Noc (%)": bp2_occ.item(),
                f"Kitti 2015 Validation BP-3 Noc (%)": bp3_occ.item(),
                f"Kitti 2015 Validation D1 Noc (%)": d1_occ.item(),
                f"Kitti 2015 Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"Kitti 2015 Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("Kitti 2015", val_metrics, global_step, args, run)


def validate_middlebury_q(model, val_loader, global_step, args, run):
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image   = val_batch['left_image'].to(args.device)
            right_image  = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)
            mask_occ     = val_batch['mask_occ'].to(args.device).bool()

            # Combine disparity validity with non-occluded pixel mask
            mask = (disparity_gt >= 0.5) & (disparity_gt < args.maxdisp) & mask_occ

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            bp1 = Threshold(prediction, disparity_gt, mask, threshold=1.0)
            bp2 = Threshold(prediction, disparity_gt, mask, threshold=2.0)
            bp3 = Threshold(prediction, disparity_gt, mask, threshold=3.0)

            val_step_metrics = {
                f"Middlebury Q Validation BP-1 (%)": bp1.item(),
                f"Middlebury Q Validation BP-2 (%)": bp2.item(),
                f"Middlebury Q Validation BP-3 (%)": bp3.item(),
                f"Middlebury Q Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"Middlebury Q Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("Middlebury Q", val_metrics, global_step, args, run)


def validate_middlebury_h(model, val_loader, global_step, args, run):
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image   = val_batch['left_image'].to(args.device)
            right_image  = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)
            mask_occ     = val_batch['mask_occ'].to(args.device).bool()

            # Combine disparity validity with non-occluded pixel mask
            mask = (disparity_gt >= 0.5) & (disparity_gt < args.maxdisp) & mask_occ

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            bp1 = Threshold(prediction, disparity_gt, mask, threshold=1.0)
            bp2 = Threshold(prediction, disparity_gt, mask, threshold=2.0)
            bp3 = Threshold(prediction, disparity_gt, mask, threshold=3.0)

            val_step_metrics = {
                f"Middlebury H Validation BP-1 (%)": bp1.item(),
                f"Middlebury H Validation BP-2 (%)": bp2.item(),
                f"Middlebury H Validation BP-3 (%)": bp3.item(),
                f"Middlebury H Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"Middlebury H Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("Middlebury H", val_metrics, global_step, args, run)
    

def validate_eth3d(model, val_loader, global_step, args, run):
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image   = val_batch['left_image'].to(args.device)
            right_image  = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)
            mask_occ     = val_batch['mask_occ'].to(args.device).bool()

            # Combine disparity validity with non-occluded pixel mask
            mask = (disparity_gt >= 0.5) & (disparity_gt < args.maxdisp) & mask_occ

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            bp1 = Threshold(prediction, disparity_gt, mask, threshold=1.0)
            bp2 = Threshold(prediction, disparity_gt, mask, threshold=2.0)
            bp3 = Threshold(prediction, disparity_gt, mask, threshold=3.0)

            val_step_metrics = {
                f"ETH3D Validation BP-1 (%)": bp1.item(),
                f"ETH3D Validation BP-2 (%)": bp2.item(),
                f"ETH3D Validation BP-3 (%)": bp3.item(),
                f"ETH3D Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"ETH3D Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("ETH3D", val_metrics, global_step, args, run)
    

def validate_boosterq(model, val_loader, global_step, args, run):
    unwrapped_model = model.module if isinstance(model, DDP) else model
    val_metrics = AverageMetricsDict()

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_loader):
            left_image   = val_batch['left_image'].to(args.device)
            right_image  = val_batch['right_image'].to(args.device)
            disparity_gt = val_batch['disparity'].to(args.device)
            mask_occ     = val_batch['mask_occ'].to(args.device).bool()

            # Combine disparity validity with non-occluded pixel mask
            mask = (disparity_gt >= 0.5) & (disparity_gt < args.maxdisp) & mask_occ

            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)

            torch.cuda.synchronize()
            start_time = time.time()
            prediction = unwrapped_model(left_image, right_image, iters=args.valid_iters, test_mode=True)
            torch.cuda.synchronize()
            inference_time_ms = int((time.time() - start_time) * 1000)

            prediction = padder.unpad(prediction)
            prediction = prediction.squeeze(1)
            assert prediction.shape == mask.shape == disparity_gt.shape, [prediction.shape, mask.shape, disparity_gt.shape]

            bp2 = Threshold(prediction, disparity_gt, mask, threshold=2.0)
            bp4 = Threshold(prediction, disparity_gt, mask, threshold=4.0)
            bp6 = Threshold(prediction, disparity_gt, mask, threshold=6.0)
            bp8 = Threshold(prediction, disparity_gt, mask, threshold=8.0)
            epe = EPE(prediction, disparity_gt, mask)

            val_step_metrics = {
                f"BoosterQ Validation BP-2 (%)": bp2.item(),
                f"BoosterQ Validation BP-4 (%)": bp4.item(),
                f"BoosterQ Validation BP-6 (%)": bp6.item(),
                f"BoosterQ Validation BP-8 (%)": bp8.item(),
                f"BoosterQ Validation EPE (px)": epe.item(),
                f"BoosterQ Inference Time (ms)": inference_time_ms,
            }

            #if (val_batch_idx + 1) <= 10:
            #    del val_step_metrics[f"BoosterQ Inference Time (ms)"]

            val_metrics.update(val_step_metrics)
            del val_batch, val_step_metrics

    log_val_metrics("BoosterQ", val_metrics, global_step, args, run)
