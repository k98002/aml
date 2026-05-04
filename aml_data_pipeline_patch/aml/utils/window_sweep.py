"""
Run multiple temporal extraction windows and summarize graph-size diagnostics.

This helps choose `--window-size` empirically instead of picking a fixed number
of transactions per window. It repeatedly calls data_pipeline.build into separate
subdirectories and writes a compact CSV summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from data_pipeline import build  # noqa: E402


def _q(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else 0.0


def _summarize(out_dir: Path, window: str) -> Dict[str, object]:
    idx_path = out_dir / "patterns_unique.csv"
    summary_path = out_dir / "data_pipeline_summary.json"
    df = pd.read_csv(idx_path)
    df_l = df[df["is_laundering"].astype(int) == 1]
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    row: Dict[str, object] = {
        "window_size": window,
        "unique_patterns": len(df),
        "laundering_patterns": len(df_l),
        "normal_patterns": int((df["is_laundering"].astype(int) == 0).sum()),
        "train_laundering": summary["splits"].get("train_laundering", 0),
        "dropped_max_nodes": summary["counters"].get("dropped_max_nodes", 0),
        "dropped_max_edges": summary["counters"].get("dropped_max_edges", 0),
        "large_bucket_flushes": summary["counters"].get("large_bucket_flushes", 0),
    }
    for prefix, part in (("all", df), ("laundering", df_l)):
        for metric in ("n", "m_unique", "tx_count", "duration_seconds"):
            row[f"{prefix}_{metric}_q50"] = _q(part[metric], 0.50) if metric in part else 0.0
            row[f"{prefix}_{metric}_q90"] = _q(part[metric], 0.90) if metric in part else 0.0
            row[f"{prefix}_{metric}_q95"] = _q(part[metric], 0.95) if metric in part else 0.0
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep AML extraction window sizes.")
    p.add_argument("--csv-path", default="data/SAML-D.csv")
    p.add_argument("--base-out-dir", default="data/window_sweep")
    p.add_argument("--windows", default="1D,3D,7D,14D")
    p.add_argument("--session-gap", default="24h")
    p.add_argument("--max-nodes", type=int, default=49)
    p.add_argument("--max-unique-edges", type=int, default=95)
    p.add_argument("--max-tx", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-laundering-types", default="")
    p.add_argument("--holdout-laundering-types", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_out_dir)
    base.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for window in [w.strip() for w in args.windows.split(",") if w.strip()]:
        out_dir = base / f"window_{window.replace('/', '_')}"
        ns = SimpleNamespace(
            csv_path=args.csv_path,
            out_dir=str(out_dir),
            chunksize=args.chunksize,
            window_size=window,
            session_gap=args.session_gap,
            bucket_mode="window",
            filter_laundering=-1,
            min_nodes=3,
            min_tx=2,
            min_unique_edges=2,
            max_nodes=args.max_nodes,
            max_unique_edges=args.max_unique_edges,
            max_tx=args.max_tx,
            max_bucket_rows=300_000,
            val_frac=0.10,
            test_frac=0.10,
            split_seed=args.split_seed,
            train_laundering_types=args.train_laundering_types,
            holdout_laundering_types=args.holdout_laundering_types,
            log_every_chunks=10,
        )
        print("=" * 72)
        print(f"Building window {window} -> {out_dir}")
        print("=" * 72)
        build(ns)
        rows.append(_summarize(out_dir, window))

    summary = pd.DataFrame(rows)
    out_csv = base / "window_sweep_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote sweep summary: {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
