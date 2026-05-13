"""5-fold fine-tuning of a pyiqa metric on a MUGSQA subset.

Expected layout:

    <dataset_dir>/
        reference/<object>_*.mp4
        main/<object>_*/<object>_*.mp4         # or additional/
        ...

The MOS file is the released `mos_main.xlsx` / `mos_additional.xlsx`. It must
contain at minimum a `video_name` column (matching distorted .mp4 filenames)
and an `MOS` column.

For each fold the best weights (by validation PLCC) are written to
    <weights_dir>/<metric>/<metric>_fold<i>_best.pth
and the average correlations across folds to
    <output_dir>/average_results_<subset>.csv
"""

import argparse
import csv
import logging
import os

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

VIDEO_EXTS = ('.mp4', '.avi', '.mov')

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('finetune')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=str, required=True,
                   help='Root directory containing reference/ and the chosen subset.')
    p.add_argument('--subset', type=str, default='main', choices=['main', 'additional'])
    p.add_argument('--mos_xlsx', type=str, required=True,
                   help='Released MOS Excel file (mos_main.xlsx or mos_additional.xlsx).')
    p.add_argument('--output_dir', type=str, default='benchmark/finetune')
    p.add_argument('--weights_dir', type=str, default='benchmark/weights')
    p.add_argument('--metrics', type=str, nargs='+', default=['dbcnn'])
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--num_frames', type=int, default=1,
                   help='Frames sampled per video (1 = first frame only).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_mos_table(xlsx_path):
    """Read mos_main.xlsx / mos_additional.xlsx -> dict[video_name -> MOS]."""
    df = pd.read_excel(xlsx_path)
    if 'video_name' not in df.columns or 'MOS' not in df.columns:
        raise ValueError(
            f"{xlsx_path} must contain 'video_name' and 'MOS' columns "
            f"(got {list(df.columns)})")
    df = df.groupby('video_name', as_index=False)['MOS'].mean()
    return dict(zip(df['video_name'], df['MOS'].astype(float)))


class MUGSQAVideoDataset(Dataset):
    """Pairs each distorted MUGSQA video with its reference and MOS score."""

    def __init__(self, dataset_dir, subset, mos_table, num_frames=1):
        self.num_frames = num_frames

        ref_dir = os.path.join(dataset_dir, 'reference')
        dist_root = os.path.join(dataset_dir, subset)

        refs = {}
        for f in os.listdir(ref_dir):
            if f.lower().endswith(VIDEO_EXTS):
                obj = os.path.splitext(f)[0].split('_')[0]
                refs[obj] = os.path.join(ref_dir, f)

        self.samples = []  # (ref_path, dist_path, video_name, mos)
        for root, _, files in os.walk(dist_root):
            for f in files:
                if not f.lower().endswith(VIDEO_EXTS):
                    continue
                if f not in mos_table:
                    continue
                obj = os.path.splitext(f)[0].split('_')[0]
                ref = refs.get(obj)
                if ref is None:
                    continue
                self.samples.append((ref, os.path.join(root, f), f, mos_table[f]))

        log.info("Loaded %d videos for subset=%s", len(self.samples), subset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref_path, dist_path, _, mos = self.samples[idx]
        return {
            'ref_frames':  self._sample_frames(ref_path),
            'dist_frames': self._sample_frames(dist_path),
            'mos': torch.tensor(mos, dtype=torch.float32),
        }

    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Empty video: {video_path}")

        if self.num_frames == 1:
            indices = [0]
        else:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()

        out = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise ValueError(f"Cannot read frame {i} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(TF.to_tensor(Image.fromarray(frame)))
        cap.release()
        return torch.stack(out, dim=0)  # [T, C, H, W]


def correlations(y_true, y_pred):
    plcc, _ = pearsonr(y_true, y_pred)
    srcc, _ = spearmanr(y_true, y_pred)
    krcc, _ = kendalltau(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return plcc, srcc, krcc, rmse


def run_epoch(model, loader, optimizer, device, lower_better):
    """Run one training epoch; returns mean loss."""
    model.train()
    criterion = nn.MSELoss()
    losses = []
    for batch in loader:
        ref_frames = batch['ref_frames'].to(device)
        dist_frames = batch['dist_frames'].to(device)
        mos = batch['mos'].to(device)

        scores = []
        for i in range(ref_frames.size(0)):
            s = model(dist_frames[i], ref_frames[i])
            scores.append(s)
        pred = torch.stack(scores)
        if lower_better:
            pred = -pred

        optimizer.zero_grad()
        loss = criterion(pred, mos)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('inf')


def run_validation(model, loader, device, lower_better):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            ref_frames = batch['ref_frames'].to(device)
            dist_frames = batch['dist_frames'].to(device)
            mos = batch['mos'].cpu().numpy()
            for i in range(ref_frames.size(0)):
                s = model(dist_frames[i], ref_frames[i])
                if lower_better:
                    s = -s
                preds.append(s.cpu().item())
            truths.extend(mos.tolist())
    if not preds:
        return None
    return correlations(truths, preds)


def finetune_one_fold(metric_name, train_loader, val_loader, device, args, fold_idx):
    log.info("=== %s fold %d/%d ===", metric_name, fold_idx, args.n_folds)
    iqa = pyiqa.create_metric(metric_name, device=device, as_loss=True)
    lower_better = bool(iqa.lower_better)

    trainable = [p for p in iqa.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=args.lr)

    weights_dir = os.path.join(args.weights_dir, metric_name)
    os.makedirs(weights_dir, exist_ok=True)

    best = {'plcc': -float('inf'), 'srcc': 0.0, 'krcc': 0.0, 'rmse': float('inf')}
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(iqa, train_loader, optimizer, device, lower_better)
        result = run_validation(iqa, val_loader, device, lower_better)
        if result is None:
            log.warning("fold %d epoch %d: validation produced no scores", fold_idx, epoch)
            continue
        plcc, srcc, krcc, rmse = result
        log.info("fold %d epoch %d  loss=%.4f  PLCC=%.4f SRCC=%.4f KRCC=%.4f RMSE=%.4f",
                 fold_idx, epoch, train_loss, plcc, srcc, krcc, rmse)

        if plcc > best['plcc']:
            best.update(plcc=plcc, srcc=srcc, krcc=krcc, rmse=rmse)
            ckpt_path = os.path.join(weights_dir, f"{metric_name}_fold{fold_idx}_best.pth")
            torch.save({'state_dict': iqa.state_dict(),
                        'epoch': epoch, **best}, ckpt_path)
            log.info("  saved %s", ckpt_path)

    return best


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    mos_table = load_mos_table(args.mos_xlsx)
    log.info("Loaded %d MOS rows from %s", len(mos_table), args.mos_xlsx)

    dataset = MUGSQAVideoDataset(args.dataset_dir, args.subset, mos_table, args.num_frames)
    if len(dataset) == 0:
        raise SystemExit("No videos matched between dataset and MOS file.")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(kf.split(range(len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    avg_rows = []
    for metric in args.metrics:
        fold_results = []
        for k, (train_idx, val_idx) in enumerate(splits, start=1):
            train_loader = DataLoader(Subset(dataset, train_idx),
                                      batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(Subset(dataset, val_idx),
                                    batch_size=args.batch_size, shuffle=False)
            fold_results.append(
                finetune_one_fold(metric, train_loader, val_loader,
                                  args.device, args, fold_idx=k))

        avg = {k: float(np.mean([r[k] for r in fold_results]))
               for k in ('plcc', 'srcc', 'krcc', 'rmse')}
        log.info("%s average  PLCC=%.4f SRCC=%.4f KRCC=%.4f RMSE=%.4f",
                 metric, avg['plcc'], avg['srcc'], avg['krcc'], avg['rmse'])
        avg_rows.append({'metric': metric, 'avg_plcc': avg['plcc'],
                         'avg_srcc': avg['srcc'], 'avg_krcc': avg['krcc'],
                         'avg_rmse': avg['rmse']})

    avg_csv = os.path.join(args.output_dir, f"average_results_{args.subset}.csv")
    with open(avg_csv, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['metric', 'avg_plcc', 'avg_srcc',
                                                 'avg_krcc', 'avg_rmse'])
        writer.writeheader()
        writer.writerows(avg_rows)
    log.info("Wrote %s", avg_csv)


if __name__ == '__main__':
    main()
