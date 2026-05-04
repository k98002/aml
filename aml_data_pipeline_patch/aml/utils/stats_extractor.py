"""Extract per-laundering-type statistics from dataset index."""

import json
import os
import pandas as pd
import numpy as np
from s3_utils import ensure_dataset_files

def extract_stats_by_type(index_path: str, output_path: str):
    """
    Load dataset index CSV, compute statistics grouped by laundering_type.

    Computes mean, std, min, max for: n, m_unique, cycles, depth, max_out

    Args:
        index_path: Path to patterns_sorted.csv
        output_path: Path to write patterns_stats_by_type.json
    """
    # Load index
    df = pd.read_csv(index_path)

    # Filter to laundering patterns only
    df_launder = df[df['is_laundering'] == 1].copy()

    stats = {}

    for launder_type in sorted(df_launder['laundering_type'].unique()):
        type_df = df_launder[df_launder['laundering_type'] == launder_type]

        stats[launder_type] = {
            'count': int(type_df.shape[0]),
            'n': {
                'mean': float(type_df['n'].mean()),
                'std': float(type_df['n'].std()),
                'min': int(type_df['n'].min()),
                'max': int(type_df['n'].max()),
            },
            'm_unique': {
                'mean': float(type_df['m_unique'].mean()),
                'std': float(type_df['m_unique'].std()),
                'min': int(type_df['m_unique'].min()),
                'max': int(type_df['m_unique'].max()),
            },
            'cycles': {
                'mean': float(type_df['cycles'].mean()),
                'std': float(type_df['cycles'].std()),
                'min': int(type_df['cycles'].min()),
                'max': int(type_df['cycles'].max()),
            },
            'depth': {
                'mean': float(type_df['depth'].mean()),
                'std': float(type_df['depth'].std()),
                'min': int(type_df['depth'].min()),
                'max': int(type_df['depth'].max()),
            },
            'max_out': {
                'mean': float(type_df['max_out'].mean()),
                'std': float(type_df['max_out'].std()),
                'min': int(type_df['max_out'].min()),
                'max': int(type_df['max_out'].max()),
            },
        }

    # Also compute global stats for all laundering patterns
    stats['_global'] = {
        'count': int(df_launder.shape[0]),
        'n': {
            'mean': float(df_launder['n'].mean()),
            'std': float(df_launder['n'].std()),
            'min': int(df_launder['n'].min()),
            'max': int(df_launder['n'].max()),
        },
        'm_unique': {
            'mean': float(df_launder['m_unique'].mean()),
            'std': float(df_launder['m_unique'].std()),
            'min': int(df_launder['m_unique'].min()),
            'max': int(df_launder['m_unique'].max()),
        },
        'cycles': {
            'mean': float(df_launder['cycles'].mean()),
            'std': float(df_launder['cycles'].std()),
            'min': int(df_launder['cycles'].min()),
            'max': int(df_launder['cycles'].max()),
        },
        'depth': {
            'mean': float(df_launder['depth'].mean()),
            'std': float(df_launder['depth'].std()),
            'min': int(df_launder['depth'].min()),
            'max': int(df_launder['depth'].max()),
        },
        'max_out': {
            'mean': float(df_launder['max_out'].mean()),
            'std': float(df_launder['max_out'].std()),
            'min': int(df_launder['max_out'].min()),
            'max': int(df_launder['max_out'].max()),
        },
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Extracted stats for {len(stats) - 1} laundering types + global stats")
    print(f"Saved to {output_path}")

    return stats


if __name__ == '__main__':
    # Default to src/data/ directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    index_path = os.path.join(data_dir, 'patterns_sorted.csv')
    output_path = os.path.join(data_dir, 'patterns_stats_by_type.json')

    # Ensure patterns_sorted.csv exists (download from S3 if needed)
    if not os.path.exists(index_path):
        print("=" * 70)
        print("patterns_sorted.csv not found locally. Attempting to download from S3...")
        print("=" * 70)
        ensure_dataset_files(data_dir, verbose=True)

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"patterns_sorted.csv not found at {index_path}.\n"
            f"Run extract_sort.py first or ensure AWS S3 access is configured."
        )

    extract_stats_by_type(index_path, output_path)
