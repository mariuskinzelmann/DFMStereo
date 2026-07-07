"""
Script for realtime demo on Intel RealSense models. Requires RealSense to be installed.
Supported Versions for Ubuntu:
- Ubuntu 20/22/24 LTS
1) Install Intel Realsense from: https://dev.realsenseai.com/installation/linux-ubuntu-installation-from-source/
2) Install conda environment using 'environment.yml'.
3) Activate dfmstereo environment.
4) Run Script.
"""


import numpy as np
import cv2
import pyrealsense2 as rs
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.DFMStereo.core.dfmstereo_large import init_dfmstereo_large
from models.DFMStereo.core.dfmstereo_medium import init_dfmstereo_medium
from models.DFMStereo.core.dfmstereo_small import init_dfmstereo_small

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Realtime Demo.')
    parser.add_argument('--mixed_precision', default=True, type=bool)
    parser.add_argument('--precision_dtype', default='float16', type=str)

    parser.add_argument('--dfmstereo_ckpt', default='million_scale', type=str)
    parser.add_argument('--valid_iters', default=8, type=int)

    parser.add_argument('--record', default=True, type=bool)
    parser.add_argument('--compile', default=True, type=bool)
    args = parser.parse_args()

    #args.optimise_volume_build = True

    model = init_dfmstereo_large(args, eval=True).eval().cuda()
    if args.compile:
        model  = torch.compile(model, mode="max-autotune-no-cudagraphs")

    width, height, fps = 640, 480, 15

    COLOR_MAP=cv2.COLORMAP_TURBO

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # left
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # right

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    intr = left_profile.get_intrinsics()
    fx = intr.fx
    extr = left_profile.get_extrinsics_to(right_profile)
    baseline_m = abs(extr.translation[0])  # ~0.095 m on the D455

    print(f"fx = {fx:.2f} px, baseline = {baseline_m * 1000:.1f} mm")

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_left = cv2.VideoWriter('left_ir.mp4',   fourcc, fps, (width, height), isColor=False)
        out_depth = cv2.VideoWriter('depth.mp4',    fourcc, fps, (width, height), isColor=True)
        out_combined = cv2.VideoWriter('combined.mp4', fourcc, fps, (width * 2, height), isColor=True)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            if not left_frame or not right_frame:
                continue

            left_img = np.asanyarray(left_frame.get_data(), dtype=np.uint8)
            right_img = np.asanyarray(right_frame.get_data(), dtype=np.uint8)
            
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)

            left_img = torch.from_numpy(left_img).permute(2, 0, 1).float().unsqueeze(0).cuda()
            right_img = torch.from_numpy(right_img).permute(2, 0, 1).float().unsqueeze(0).cuda()
            
            with torch.inference_mode():
                disparity = model(left_img, right_img, iters=args.valid_iters, test_mode=True, being_taught=False)

            # disparity (px) -> depth (m): depth = fx * baseline / disparity
            
            left_img = left_img.squeeze(0).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
            
            disparity = disparity.squeeze(0).squeeze(0).cpu().numpy()

            with np.errstate(divide="ignore", invalid="ignore"):
                depth_m = np.where(disparity > 0, (fx * baseline_m) / disparity, 0)

            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.clip(0, 255).astype(np.uint8), COLOR_MAP)[...,::-1]
            #depth_vis = np.uint8(depth_vis)
            if args.record:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                out_left.write(left_gray)
                out_depth.write(depth_vis)  # depth_vis is RGB, writer expects BGR
                out_combined.write(np.hstack([cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR), depth_vis]))

            cv2.imshow("Left View", left_img)
            cv2.imshow("Custom Depth", depth_vis)
            if cv2.waitKey(1) == 27:  # Esc to quit
                break
    finally:
        out_left.release()
        out_depth.release()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()