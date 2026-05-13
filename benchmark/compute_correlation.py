"""Compute PLCC / SRCC / KRCC / RMSE between a metric CSV and the released MOS xlsx.

Inputs:
    --mos_xlsx     released mos_main.xlsx or mos_additional.xlsx
                   (must contain video_name + MOS columns)
    --metric_csv   output of calculate_metrics.py or per-fold predictions
                   (must contain video_name + score columns; optional lower_better)

Videos are matched by exact `video_name`. Before computing PLCC / RMSE the metric
scores can be passed through a 4- or 5-parameter logistic, per VQEG conventions.
"""

import argparse
from pathlib import Path

import pandas as pd

from correlation_coefficient import (
    calculate_krcc,
    calculate_plcc,
    calculate_rmse,
    calculate_srcc,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--mos_xlsx', type=Path, required=True,
                   help='Released MOS Excel file (mos_main.xlsx or mos_additional.xlsx).')
    p.add_argument('--metric_csv', type=Path, required=True,
                   help='Per-video metric scores CSV (video_name, score[, lower_better]).')
    p.add_argument('--fit_scale', type=str, default='logistic_4params',
                   choices=['logistic_4params', 'logistic_5params', 'none'],
                   help='Logistic curve used before PLCC / RMSE (VQEG-style).')
    return p.parse_args()


def load_mos(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    if 'video_name' not in df.columns or 'MOS' not in df.columns:
        raise ValueError(
            f"{xlsx_path} must contain 'video_name' and 'MOS' columns "
            f"(got {list(df.columns)})")
    return df.groupby('video_name', as_index=False)['MOS'].mean()


def load_metric(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'video_name' not in df.columns or 'score' not in df.columns:
        raise ValueError(
            f"{csv_path} must contain 'video_name' and 'score' columns "
            f"(got {list(df.columns)})")
    if 'lower_better' not in df.columns:
        df['lower_better'] = False
    else:
        df['lower_better'] = df['lower_better'].astype(bool)
    return df[['video_name', 'score', 'lower_better']]


def main():
    args = parse_args()
    mos_df = load_mos(args.mos_xlsx)
    metric_df = load_metric(args.metric_csv)

    merged = metric_df.merge(mos_df, on='video_name', how='inner')
    if merged.empty:
        raise SystemExit("No overlapping videos between MOS file and metric CSV.")

    lb = merged['lower_better']
    if lb.any() and not lb.all():
        raise ValueError("Mixed lower_better values within the metric CSV.")
    x = merged['score'].to_numpy()
    if lb.iloc[0]:
        x = -x
    y = merged['MOS'].to_numpy()

    fit_scale = None if args.fit_scale == 'none' else args.fit_scale

    results = {
        'PLCC': calculate_plcc(x, y, fit_scale),
        'SRCC': calculate_srcc(x, y),
        'KRCC': calculate_krcc(x, y),
        'RMSE': calculate_rmse(x, y, fit_scale),
    }

    print(f"Matched {len(merged)} videos "
          f"({len(mos_df)} MOS rows, {len(metric_df)} metric rows).")
    for key, val in results.items():
        print(f"  {key}: {'None' if val is None else f'{val:.6f}'}")


if __name__ == '__main__':
    main()
