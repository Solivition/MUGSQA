"""Sweep one Gaussian-Splatting reconstruction method over the MUGSQA grid.

This driver is method-agnostic. Drop it into the root of any patched upstream
GS repository (3DGS / Mip-Splatting / LightGaussian / Scaffold-GS / Octree-GS /
EfficientGaussian) and point it at the dataset produced by `render/`.

For every source object under `--dataset_dir`, it iterates the Cartesian
product of `--res x --views x --distance`, calls the upstream training script
on each distortion folder, and (optionally) renders an evaluation video.

Example
-------
    # 3DGS, GT point cloud
    python train_all.py \
        --dataset_dir /path/to/MUGSQA_SOURCE \
        --method 3dgs --train_script train.py \
        --gtpc --generate_video --port 6009

    # LightGaussian, random init, with explicit --gpu forwarded to the
    # upstream train script
    python train_all.py \
        --dataset_dir /path/to/MUGSQA_SOURCE \
        --method lgs --train_script train_densify_prune.py \
        --gpu 3 --generate_video --port 6010
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=str, required=True,
                   help='Root with one subfolder per source object (output of `render/`).')
    p.add_argument('--train_script', type=str, default='train.py',
                   help="Upstream training script in the current repo "
                        "(e.g. 'train.py', 'train_eval.py', 'train_densify_prune.py').")
    p.add_argument('--render_script', type=str, default='render_video.py',
                   help='Per-method video rendering script (provided in the MUGSQA patch).')
    p.add_argument('--method', type=str, required=True,
                   help="Tag used in the output folder name "
                        "(e.g. '3dgs', 'mip-splatting', 'lgs', 'scaffold-gs', 'octree-gs').")
    p.add_argument('--res', nargs='+', type=int, default=[1080, 720, 480],
                   help='Input view resolutions to sweep.')
    p.add_argument('--views', nargs='+', type=int, default=[72, 36, 9],
                   help='Number of input views to sweep.')
    p.add_argument('--distance', nargs='+', type=int, default=[5, 2, 1],
                   help='Camera-to-object distances to sweep.')
    p.add_argument('--gtpc', action='store_true',
                   help='Initialize from the ground-truth point cloud (otherwise: random).')
    p.add_argument('--generate_video', action='store_true',
                   help='Render an evaluation video after each successful training run.')
    p.add_argument('--port', type=int, default=6009,
                   help='GUI port forwarded to the upstream training script.')
    p.add_argument('--gpu', type=int, default=None,
                   help='If set, forwarded as `--gpu N` to the upstream training script. '
                        'Some methods (LightGaussian / Scaffold-GS / Octree-GS) require it.')
    return p.parse_args()


def run(cmd):
    print('>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f'   command exited with code {rc}', file=sys.stderr)
    return rc


def train_one_setting(args, res, views, distance):
    pc_suffix = 'gtpc' if args.gtpc else 'rndpc'
    out_tag = f'{args.method}_{pc_suffix}'
    setting_tag = f'{res}res_{views}views_{distance}distance'

    for sample in sorted(os.listdir(args.dataset_dir)):
        sample_root = os.path.join(args.dataset_dir, sample)
        dist_dir = os.path.join(sample_root, 'distortion', setting_tag)
        if not os.path.isdir(dist_dir):
            continue

        render_dir = os.path.join(dist_dir, 'render')
        out_dir = os.path.join(dist_dir, out_tag)
        os.makedirs(out_dir, exist_ok=True)

        # 1. Reconstruction
        done_marker = os.path.join(out_dir, 'point_cloud', 'iteration_30000')
        if os.path.exists(done_marker):
            print(f'  [skip-train] {sample}/{setting_tag}')
        else:
            cmd = ['python', args.train_script,
                   '-s', render_dir,
                   '-m', out_dir,
                   '--port', str(args.port)]
            if args.gtpc:
                cmd.append('--gtpc')
            if args.gpu is not None:
                cmd += ['--gpu', str(args.gpu)]
            run(cmd)

        # 2. Evaluation video
        if args.generate_video:
            if os.path.exists(os.path.join(out_dir, 'video')):
                print(f'  [skip-video] {sample}/{setting_tag}')
                continue
            ref_render = os.path.join(sample_root, 'calibrated', 'render')
            cmd = ['python', args.render_script,
                   '-s', ref_render,
                   '-m', out_dir,
                   '--white_background']
            run(cmd)


def main():
    args = parse_args()
    total = len(args.res) * len(args.views) * len(args.distance)
    idx = 1
    for r in args.res:
        for v in args.views:
            for d in args.distance:
                print(f'=== Setting {idx}/{total}: res={r} views={v} distance={d} ===')
                train_one_setting(args, r, v, d)
                idx += 1
    print('Finished.')


if __name__ == '__main__':
    main()
