#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2

def render_video(model_path, iteration, views, gaussians, pipeline, background):
    # render_path_spiral
    # render_path_spherical
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)

    # Render input views - n_frames = n_views
    for idx, view in enumerate(views[:-4]):

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        rendering = render_pkg["render"]

        img_path = os.path.join(render_path, 'imgs')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        torchvision.utils.save_image(rendering, os.path.join(img_path, '{0:05d}'.format(idx) + ".png"))

    # Generate video
    fps = 30
    output_video_path = os.path.join(render_path, '3DGS_video.mp4')
    # Get a list of image file names and sort them
    images = sorted([img for img in os.listdir(img_path) if img.endswith(("png", "jpg", "jpeg"))])
    if not images:
        print("No images found in the folder.")
        return
    # Read the first image to determine the frame size
    first_image_path = os.path.join(img_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height)
    # Creating a Video Write Object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using the MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    # Write each image to the video in order
    for image in images:
        cur_img_path = os.path.join(img_path, image)
        frame = cv2.imread(cur_img_path)
        if frame is None:
            print(f"Failed to read image: {cur_img_path}")
            continue
        video_writer.write(frame)
    # Release of resources
    video_writer.release()
    print(f"Video saved to {output_video_path}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [0.6,0.6,0.6] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args))