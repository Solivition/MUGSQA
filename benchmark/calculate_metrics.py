"""Compute a per-video IQA/VQA metric over a MUGSQA subset.

Assumes the released layout:

    <dataset_dir>/
        reference/<object>_*.mp4
        main/<object>_*/<object>_*.mp4         # or additional/
        ...

For each distorted video, the matching reference is looked up by the leading
`<object>` token of the filename, the metric is run over all decoded frames
in batches, and the mean score is appended to:

    <output_dir>/<metric>_<subset>.csv      (columns: video_name, score, lower_better)

FID is handled separately because pyiqa's FID expects directories of frames,
not videos; in that case the caller is responsible for pre-extracting frames
into `<video>.mp4`-stripped sibling folders.
"""

import argparse
import csv
import io
import os

import cv2
import numpy as np
import pyiqa
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=str, required=True,
                   help='Root directory containing reference/ and the chosen subset.')
    p.add_argument('--subset', type=str, default='main', choices=['main', 'additional'],
                   help='Which MUGSQA split to score.')
    p.add_argument('--output_dir', type=str, default='benchmark/all_frames',
                   help='Where to write <metric>_<subset>.csv.')
    p.add_argument('--metric', type=str, default='psnr',
                   help='Any metric name supported by pyiqa '
                        '(psnr, ssim, ms_ssim, cw_ssim, fsim, vif, gmsd, nlpd, vsi, '
                        'lpips, niqe, piqe, fid, ...).')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=180,
                   help='Frames per forward pass.')
    return p.parse_args()


def imread2tensor(img):
    """PIL.Image -> CHW float tensor in [0, 1]."""
    return TF.to_tensor(img.convert('RGB'))


def read_video_frames(path):
    """Decode an entire video into a list of RGB ndarrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def score_video_pair(ref_path, dist_path, model, device, batch_size):
    ref_frames = read_video_frames(ref_path)
    dist_frames = read_video_frames(dist_path)
    if not ref_frames or not dist_frames:
        raise ValueError(f"Empty video: {ref_path if not ref_frames else dist_path}")

    n = min(len(ref_frames), len(dist_frames))
    ref = torch.stack([imread2tensor(Image.fromarray(f)) for f in ref_frames[:n]])
    dist = torch.stack([imread2tensor(Image.fromarray(f)) for f in dist_frames[:n]])
    ref, dist = ref.to(device), dist.to(device)

    scores = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = model(dist[i:i + batch_size], ref[i:i + batch_size])
            if isinstance(batch, torch.Tensor):
                batch = batch.cpu().numpy()
            scores.extend(np.asarray(batch).flatten().tolist())
    return float(np.mean(scores)) if scores else None


def score_fid_pair(ref_dir, dist_dir, model):
    """FID expects directories of pre-extracted frames, not videos."""
    with torch.no_grad():
        score = model(dist_dir, ref_dir)
    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()
    return float(np.asarray(score).flatten()[0])


def find_video_pairs(dataset_dir, subset, is_fid):
    """Return list of (ref_path, dist_path, dist_basename) tuples.

    For FID, paths point at pre-extracted frame directories instead of .mp4 files;
    the frame dir for a video `foo.mp4` is expected to live next to it as `foo/`
    (or, for references, under `reference/imgs/<object>/`).
    """
    ref_dir = os.path.join(dataset_dir, 'reference')
    dist_root = os.path.join(dataset_dir, subset)
    video_exts = ('.mp4', '.avi', '.mov')

    refs = {}
    for f in os.listdir(ref_dir):
        if not f.lower().endswith(video_exts):
            continue
        obj_key = os.path.splitext(f)[0].split('_')[0]
        if is_fid:
            frames_dir = os.path.join(ref_dir, 'imgs', obj_key)
            if os.path.isdir(frames_dir):
                refs[obj_key] = frames_dir
        else:
            refs[obj_key] = os.path.join(ref_dir, f)

    pairs = []
    for root, _, files in os.walk(dist_root):
        for f in files:
            if not f.lower().endswith(video_exts):
                continue
            obj_key = os.path.splitext(f)[0].split('_')[0]
            if obj_key not in refs:
                continue
            video_path = os.path.join(root, f)
            if is_fid:
                frames_dir = os.path.join(root, os.path.splitext(f)[0])
                if not os.path.isdir(frames_dir):
                    print(f"  skipping {video_path}: no extracted frames dir")
                    continue
                pairs.append((refs[obj_key], frames_dir, f))
            else:
                pairs.append((refs[obj_key], video_path, f))
    return pairs


def main():
    args = parse_args()
    is_fid = args.metric.lower() == 'fid'

    pairs = find_video_pairs(args.dataset_dir, args.subset, is_fid)
    print(f"Matched {len(pairs)} distorted videos under {args.subset}/")

    model = pyiqa.create_metric(args.metric, device=args.device)
    lower_better = bool(model.lower_better)

    results = []
    for ref, dist, video_name in tqdm(pairs, desc=f"{args.metric}/{args.subset}"):
        try:
            score = (score_fid_pair(ref, dist, model) if is_fid
                     else score_video_pair(ref, dist, model, args.device, args.batch_size))
        except Exception as exc:
            print(f"  error on {video_name}: {exc}")
            continue
        results.append({'video_name': video_name, 'score': score,
                        'lower_better': lower_better})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("No results; nothing written.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.metric}_{args.subset}.csv')
    with open(out_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['video_name', 'score', 'lower_better'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == '__main__':
    main()
