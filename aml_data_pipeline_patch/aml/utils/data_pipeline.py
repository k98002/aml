"""
Build a leakage-safe, unique-pattern AML graph dataset for the PPO/GCPN project.

This replaces the older `extract_sort.py` behaviour that physically oversampled
laundering rows. The new pipeline:

1. extracts temporal connected components as unique graph patterns;
2. keeps transaction attributes in aggregated edge metadata when available;
3. applies node/edge/transaction caps compatible with the generator;
4. splits by unique pattern_id before any sampling;
5. writes train-only statistics for reward calibration;
6. leaves class/typology balancing to the online sampler.

Typical use from the project root:

    python utils/data_pipeline.py build \
        --csv-path data/SAML-D.csv \
        --out-dir data \
        --window-size 7D \
        --session-gap 24h \
        --max-nodes 49 \
        --max-unique-edges 95

Then train with:

    python train_aml.py index_path=data/patterns_train.csv \
        stats_path=data/patterns_stats_train_by_type.json \
        sampling_mode=typology_complexity_balanced
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.utils import UnionFind


# -----------------------------------------------------------------------------
# Column inference
# -----------------------------------------------------------------------------

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "sender": ("Sender_account", "Sender", "sender", "sender_account", "orig_acct", "from_account"),
    "receiver": ("Receiver_account", "Receiver", "receiver", "receiver_account", "bene_acct", "to_account"),
    "date": ("Date", "date"),
    "time": ("Time", "time"),
    "timestamp": ("Timestamp", "timestamp", "ts", "datetime", "DateTime", "date_time"),
    "is_laundering": ("Is_laundering", "is_laundering", "SAR", "label", "Label"),
    "laundering_type": ("Laundering_type", "laundering_type", "Typology", "typology", "type"),
    "amount": ("Amount", "amount", "transaction_amount", "Transaction_amount"),
    "payment_currency": ("Payment_currency", "payment_currency", "Currency", "currency"),
    "received_currency": ("Received_currency", "received_currency"),
    "payment_type": ("Payment_type", "payment_type", "Transaction_type", "transaction_type", "channel"),
    "sender_country": (
        "Sender_bank_location",
        "sender_bank_location",
        "Sender_country",
        "sender_country",
        "from_country",
    ),
    "receiver_country": (
        "Receiver_bank_location",
        "receiver_bank_location",
        "Receiver_country",
        "receiver_country",
        "to_country",
    ),
}

REQUIRED_KEYS = ("sender", "receiver", "is_laundering")


@dataclass(frozen=True)
class TxRow:
    tx_id: int
    ts: pd.Timestamp
    sender: str
    receiver: str
    is_laundering: int
    laundering_type: str
    amount: Optional[float] = None
    payment_currency: Optional[str] = None
    received_currency: Optional[str] = None
    payment_type: Optional[str] = None
    sender_country: Optional[str] = None
    receiver_country: Optional[str] = None
    time_bin: Optional[pd.Timestamp] = None


@dataclass
class PipelineCounters:
    emitted: int = 0
    dropped_empty: int = 0
    dropped_min_nodes: int = 0
    dropped_min_tx: int = 0
    dropped_min_edges: int = 0
    dropped_max_nodes: int = 0
    dropped_max_edges: int = 0
    dropped_max_tx: int = 0
    large_bucket_flushes: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _safe_str(x: object) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _safe_float(x: object) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    try:
        val = float(str(x).replace(",", ""))
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def _parse_label(x: object) -> int:
    if x is None or pd.isna(x):
        return 0
    if isinstance(x, (int, np.integer, bool)):
        return int(x)
    if isinstance(x, float):
        return int(x > 0)
    s = str(x).strip().lower()
    return int(s in {"1", "true", "t", "yes", "y", "laundering", "sar", "suspicious"})


def _parse_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_columns(columns: Iterable[str]) -> Dict[str, Optional[str]]:
    available = set(columns)
    resolved: Dict[str, Optional[str]] = {}
    for key, aliases in COLUMN_ALIASES.items():
        resolved[key] = next((alias for alias in aliases if alias in available), None)
    missing = [key for key in REQUIRED_KEYS if resolved.get(key) is None]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns {missing}. Available columns: {sorted(available)}"
        )
    if resolved.get("timestamp") is None and (resolved.get("date") is None or resolved.get("time") is None):
        raise ValueError("Input CSV needs either a timestamp column or both Date and Time columns.")
    if resolved.get("laundering_type") is None:
        print("Warning: laundering_type column not found; using 'unknown'/'normal'.")
    return resolved


def _window_key(ts: pd.Timestamp, window_size: str) -> Optional[pd.Timestamp]:
    if window_size.lower() in {"none", "null", "off"}:
        return None
    return ts.floor(window_size)


# -----------------------------------------------------------------------------
# Graph metrics and pattern building
# -----------------------------------------------------------------------------


def _cyclomatic_number_undirected(n: int, undirected_edge_count: int) -> int:
    return max(0, undirected_edge_count - n + 1)


def _depth_via_scc_condensation(dg: nx.DiGraph) -> int:
    if dg.number_of_nodes() <= 1 or dg.number_of_edges() == 0:
        return 0
    condensation = nx.condensation(dg)
    return nx.dag_longest_path_length(condensation) if condensation.number_of_edges() > 0 else 0


def _compute_complexity(node_ids: Sequence[str], edge_keys: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    n = len(node_ids)
    edges = list(edge_keys)
    dg = nx.DiGraph()
    dg.add_nodes_from(node_ids)
    dg.add_edges_from(edges)

    max_out = max((d for _, d in dg.out_degree()), default=0)
    undirected_edges = {tuple(sorted((u, v))) for (u, v) in edges}
    cycles = _cyclomatic_number_undirected(n=n, undirected_edge_count=len(undirected_edges))
    depth = _depth_via_scc_condensation(dg)

    return {
        "n": int(n),
        "m_unique": int(len(edges)),
        "cycles": int(cycles),
        "depth": int(depth),
        "max_out": int(max_out),
    }


def _union_find_components(rows: Sequence[TxRow]) -> List[List[TxRow]]:
    uf = UnionFind()
    for r in rows:
        uf.union(r.sender, r.receiver)

    groups: Dict[str, List[TxRow]] = defaultdict(list)
    for r in rows:
        groups[uf[r.sender]].append(r)
    return list(groups.values())


def _session_split(rows: Sequence[TxRow], gap: Optional[pd.Timedelta]) -> List[List[TxRow]]:
    if gap is None:
        return [list(rows)]

    rows_sorted = sorted(rows, key=lambda r: r.ts)
    sessions: List[List[TxRow]] = []
    cur: List[TxRow] = []
    last_ts: Optional[pd.Timestamp] = None

    for r in rows_sorted:
        if last_ts is not None and (r.ts - last_ts) > gap and cur:
            sessions.append(cur)
            cur = []
        cur.append(r)
        last_ts = r.ts

    if cur:
        sessions.append(cur)
    return sessions


def _edge_attr_from_rows(rows: Sequence[TxRow]) -> Dict[str, object]:
    amounts = [r.amount for r in rows if r.amount is not None]
    first_ts = min(r.ts for r in rows)
    last_ts = max(r.ts for r in rows)
    pay_cur = Counter(r.payment_currency for r in rows if r.payment_currency)
    rec_cur = Counter(r.received_currency for r in rows if r.received_currency)
    pay_type = Counter(r.payment_type for r in rows if r.payment_type)
    cross_border = [
        int(r.sender_country != r.receiver_country)
        for r in rows
        if r.sender_country is not None and r.receiver_country is not None
    ]

    if amounts:
        amount_sum = float(np.sum(amounts))
        amount_mean = float(np.mean(amounts))
        amount_min = float(np.min(amounts))
        amount_max = float(np.max(amounts))
    else:
        amount_sum = amount_mean = amount_min = amount_max = None

    return {
        "tx_count": int(len(rows)),
        "amount_sum": amount_sum,
        "amount_mean": amount_mean,
        "amount_min": amount_min,
        "amount_max": amount_max,
        "first_ts": str(first_ts),
        "last_ts": str(last_ts),
        "duration_seconds": float((last_ts - first_ts).total_seconds()),
        "payment_currencies": dict(pay_cur.most_common()),
        "received_currencies": dict(rec_cur.most_common()),
        "payment_types": dict(pay_type.most_common()),
        "cross_border_ratio": float(np.mean(cross_border)) if cross_border else None,
    }


def _pattern_label(rows: Sequence[TxRow]) -> Tuple[int, str, Dict[str, int]]:
    is_laundering = int(any(r.is_laundering == 1 for r in rows))
    if is_laundering:
        counts = Counter(r.laundering_type for r in rows if r.is_laundering == 1)
        typ = counts.most_common(1)[0][0] if counts else "unknown"
    else:
        counts = Counter({"normal": len(rows)})
        typ = "normal"
    return is_laundering, str(typ), {str(k): int(v) for k, v in counts.items()}


def _build_pattern_instance(
    rows: Sequence[TxRow],
    pattern_id: str,
    args: argparse.Namespace,
    counters: PipelineCounters,
) -> Optional[Dict[str, object]]:
    if not rows:
        counters.dropped_empty += 1
        return None

    node_ids = sorted({r.sender for r in rows}.union({r.receiver for r in rows}))
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    edge_rows: Dict[Tuple[str, str], List[TxRow]] = defaultdict(list)
    for r in rows:
        edge_rows[(r.sender, r.receiver)].append(r)

    compx = _compute_complexity(node_ids=node_ids, edge_keys=edge_rows.keys())
    tx_count = len(rows)

    if compx["n"] < args.min_nodes:
        counters.dropped_min_nodes += 1
        return None
    if tx_count < args.min_tx:
        counters.dropped_min_tx += 1
        return None
    if compx["m_unique"] < args.min_unique_edges:
        counters.dropped_min_edges += 1
        return None
    if args.max_nodes > 0 and compx["n"] > args.max_nodes:
        counters.dropped_max_nodes += 1
        return None
    if args.max_unique_edges > 0 and compx["m_unique"] > args.max_unique_edges:
        counters.dropped_max_edges += 1
        return None
    if args.max_tx > 0 and tx_count > args.max_tx:
        counters.dropped_max_tx += 1
        return None

    sorted_edge_keys = sorted(edge_rows.keys())
    edge_index: List[List[int]] = []
    edge_weight: List[int] = []
    edge_attrs: List[Dict[str, object]] = []
    for u, v in sorted_edge_keys:
        erows = edge_rows[(u, v)]
        edge_index.append([node_map[u], node_map[v]])
        edge_weight.append(len(erows))
        edge_attrs.append(_edge_attr_from_rows(erows))

    start_ts = min(r.ts for r in rows)
    end_ts = max(r.ts for r in rows)
    amounts = [r.amount for r in rows if r.amount is not None]
    amount_total = float(np.sum(amounts)) if amounts else None
    amount_mean = float(np.mean(amounts)) if amounts else None
    is_laundering, laundering_type, type_counts = _pattern_label(rows)

    time_bins = sorted({str(r.time_bin) for r in rows if r.time_bin is not None})
    window_id = time_bins[0] if len(time_bins) == 1 else "multi" if time_bins else "none"

    counters.emitted += 1
    return {
        "pattern_id": pattern_id,
        "laundering_type": laundering_type,
        "laundering_type_counts": type_counts,
        "is_laundering": int(is_laundering),
        "start_ts": str(start_ts),
        "end_ts": str(end_ts),
        "duration_seconds": float((end_ts - start_ts).total_seconds()),
        "tx_ids": [int(r.tx_id) for r in rows],
        "tx_count": int(tx_count),
        "amount_total": amount_total,
        "amount_mean": amount_mean,
        "node_ids": node_ids,
        "node_map": node_map,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_attrs": edge_attrs,
        "complexity": compx,
        "window_id": window_id,
    }


def _write_pattern(
    rec: Dict[str, object],
    fout_jsonl,
    offsets: Dict[str, int],
    index_rows: List[Dict[str, object]],
) -> None:
    offsets[str(rec["pattern_id"])] = fout_jsonl.tell()
    fout_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    c = rec["complexity"]
    index_rows.append(
        {
            "pattern_id": rec["pattern_id"],
            "laundering_type": rec["laundering_type"],
            "is_laundering": rec["is_laundering"],
            "n": c["n"],
            "m_unique": c["m_unique"],
            "cycles": c["cycles"],
            "depth": c["depth"],
            "max_out": c["max_out"],
            "tx_count": rec["tx_count"],
            "duration_seconds": rec["duration_seconds"],
            "amount_total": rec["amount_total"],
            "amount_mean": rec["amount_mean"],
            "start_ts": rec["start_ts"],
            "end_ts": rec["end_ts"],
            "window_id": rec["window_id"],
        }
    )


def _extract_patterns_from_bucket(
    bucket_rows: List[TxRow],
    pattern_counter: int,
    fout_jsonl,
    offsets: Dict[str, int],
    index_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    counters: PipelineCounters,
) -> int:
    if not bucket_rows:
        return pattern_counter

    session_gap = None if args.session_gap.lower() in {"none", "off", "null"} else pd.Timedelta(args.session_gap)

    for comp_rows in _union_find_components(bucket_rows):
        for session_rows in _session_split(comp_rows, session_gap):
            # Splitting by time can break connectivity; rebuild components after session split.
            for sub_rows in _union_find_components(session_rows):
                pattern_id = f"P{pattern_counter:07d}"
                pattern_counter += 1
                rec = _build_pattern_instance(sub_rows, pattern_id, args, counters)
                if rec is not None:
                    _write_pattern(rec, fout_jsonl, offsets, index_rows)

    return pattern_counter


# -----------------------------------------------------------------------------
# Statistics and splits
# -----------------------------------------------------------------------------


def _metric_stats(s: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q50": 0.0, "q90": 0.0, "q95": 0.0}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q50": float(clean.quantile(0.50)),
        "q90": float(clean.quantile(0.90)),
        "q95": float(clean.quantile(0.95)),
    }


def compute_stats_by_type(index_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Compute laundering-only stats from a unique train split."""
    metrics = [
        "n",
        "m_unique",
        "cycles",
        "depth",
        "max_out",
        "tx_count",
        "duration_seconds",
        "amount_total",
        "amount_mean",
    ]
    df_l = index_df[index_df["is_laundering"].astype(int) == 1].copy()
    stats: Dict[str, Dict[str, object]] = {}

    if df_l.empty:
        stats["_global"] = {"count": 0, **{m: _metric_stats(pd.Series(dtype=float)) for m in metrics}}
        return stats

    for typ, part in df_l.groupby("laundering_type", sort=True):
        stats[str(typ)] = {"count": int(len(part))}
        for metric in metrics:
            if metric in part.columns:
                stats[str(typ)][metric] = _metric_stats(part[metric])

    stats["_global"] = {"count": int(len(df_l))}
    for metric in metrics:
        if metric in df_l.columns:
            stats["_global"][metric] = _metric_stats(df_l[metric])
    return stats


def write_stats(index_df: pd.DataFrame, out_dir: Path) -> None:
    stats = compute_stats_by_type(index_df)
    stats_path = out_dir / "patterns_stats_train_by_type.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Backward-compatible flattened global stats used by older reward code/scripts.
    g = stats.get("_global", {})
    flat = {}
    mapping = {
        "n": "n",
        "m_unique": "m",
        "cycles": "cycles",
        "depth": "depth",
        "max_out": "max_out",
    }
    for src, dst in mapping.items():
        if src in g:
            flat[f"{dst}_mean"] = g[src]["mean"]
            flat[f"{dst}_std"] = g[src]["std"]
            flat[f"{dst}_min"] = g[src]["min"]
            flat[f"{dst}_max"] = g[src]["max"]
    with open(out_dir / "dataset_stats_train.json", "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)

    # Legacy filename so existing config overrides do not break immediately.
    with open(out_dir / "patterns_stats_by_type.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _strata(row: pd.Series) -> str:
    if int(row["is_laundering"]) == 1:
        return "L::" + str(row["laundering_type"])
    return "N::normal"


def split_unique_index(index_df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> Dict[str, int]:
    """Split unique pattern rows, with optional typology holdout/filtering."""
    rng = np.random.default_rng(args.split_seed)
    df = index_df.drop_duplicates("pattern_id", keep="first").copy()
    df["is_laundering"] = df["is_laundering"].astype(int)
    df["laundering_type"] = df["laundering_type"].fillna("unknown").astype(str)

    train_types = set(_parse_list(args.train_laundering_types))
    holdout_types = set(_parse_list(args.holdout_laundering_types))

    if train_types:
        train_allowed_mask = (df["is_laundering"] == 0) | (df["laundering_type"].isin(train_types))
    else:
        train_allowed_mask = pd.Series(True, index=df.index)

    holdout_mask = (df["is_laundering"] == 1) & (df["laundering_type"].isin(holdout_types))
    holdout_df = df[holdout_mask].copy()
    pool = df[~holdout_mask].copy()

    # Stratified random split by label+typology. Tiny strata are assigned train-first.
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    pool["_strata"] = pool.apply(_strata, axis=1)
    for _, part in pool.groupby("_strata", sort=True):
        part = part.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        n = len(part)
        n_test = int(round(n * args.test_frac))
        n_val = int(round(n * args.val_frac))
        if n >= 3:
            n_test = max(n_test, 1)
            n_val = max(n_val, 1)
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
        else:
            n_test = 0
            n_val = 0
        test_parts.append(part.iloc[:n_test])
        val_parts.append(part.iloc[n_test : n_test + n_val])
        train_parts.append(part.iloc[n_test + n_val :])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)

    # Put holdout laundering typologies into test only.
    if not holdout_df.empty:
        test_df = pd.concat([test_df, holdout_df], ignore_index=True)

    # Filter only the training split if current phase should use topology-representable labels.
    if train_types:
        before = len(train_df)
        train_df = train_df[(train_df["is_laundering"] == 0) | (train_df["laundering_type"].isin(train_types))].copy()
        print(f"Filtered train split to requested laundering types: {before:,} -> {len(train_df):,}")

    sort_cols = ["is_laundering", "laundering_type", "n", "m_unique", "cycles", "depth", "max_out", "tx_count"]
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part = part.drop(columns=[c for c in ["_strata"] if c in part.columns], errors="ignore")
        part = part.sort_values(sort_cols, ascending=True).reset_index(drop=True)
        part.to_csv(out_dir / f"patterns_{name}.csv", index=False)

    train_for_stats = train_df.drop(columns=[c for c in ["_strata"] if c in train_df.columns], errors="ignore")
    write_stats(train_for_stats, out_dir)

    return {
        "unique": int(len(df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "holdout_laundering": int(len(holdout_df)),
        "train_laundering": int((train_df["is_laundering"].astype(int) == 1).sum()) if not train_df.empty else 0,
        "train_normal": int((train_df["is_laundering"].astype(int) == 0).sum()) if not train_df.empty else 0,
    }


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------


def _read_rows_from_chunk(chunk: pd.DataFrame, cols: Dict[str, Optional[str]], offset_tx: int, args: argparse.Namespace) -> Tuple[List[TxRow], int]:
    original_len = len(chunk)
    if cols.get("timestamp"):
        ts = pd.to_datetime(chunk[cols["timestamp"]], errors="coerce")
    else:
        ts = pd.to_datetime(
            chunk[cols["date"]].astype(str) + " " + chunk[cols["time"]].astype(str),
            errors="coerce",
        )
    chunk = chunk.copy()
    chunk["_ts"] = ts
    chunk = chunk.dropna(subset=["_ts", cols["sender"], cols["receiver"], cols["is_laundering"]]).reset_index(drop=True)

    rows: List[TxRow] = []
    for local_i, row in chunk.iterrows():
        tx_id = int(offset_tx + local_i)
        timestamp = row["_ts"]
        time_bin = _window_key(timestamp, args.window_size)
        is_l = _parse_label(row[cols["is_laundering"]])
        raw_type = _safe_str(row[cols["laundering_type"]]) if cols.get("laundering_type") else None
        laundering_type = raw_type if is_l and raw_type else ("normal" if not is_l else "unknown")

        rows.append(
            TxRow(
                tx_id=tx_id,
                ts=timestamp,
                sender=str(row[cols["sender"]]),
                receiver=str(row[cols["receiver"]]),
                is_laundering=is_l,
                laundering_type=str(laundering_type),
                amount=_safe_float(row[cols["amount"]]) if cols.get("amount") else None,
                payment_currency=_safe_str(row[cols["payment_currency"]]) if cols.get("payment_currency") else None,
                received_currency=_safe_str(row[cols["received_currency"]]) if cols.get("received_currency") else None,
                payment_type=_safe_str(row[cols["payment_type"]]) if cols.get("payment_type") else None,
                sender_country=_safe_str(row[cols["sender_country"]]) if cols.get("sender_country") else None,
                receiver_country=_safe_str(row[cols["receiver_country"]]) if cols.get("receiver_country") else None,
                time_bin=time_bin,
            )
        )
    return rows, offset_tx + original_len


def _bucket_key(row: TxRow, mode: str) -> Tuple[object, ...]:
    # All modes start with temporal bucket, so normal and laundering use the same time logic.
    base = (str(row.time_bin) if row.time_bin is not None else "none",)
    if mode == "window":
        return base
    if mode == "window_label":
        return base + (row.is_laundering,)
    if mode == "window_type":
        return base + (row.laundering_type,)
    raise ValueError(f"Unknown bucket mode: {mode}")


def build(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    sample = pd.read_csv(csv_path, nrows=100)
    cols = _resolve_columns(sample.columns)
    print("Resolved columns:")
    for k, v in cols.items():
        print(f"  {k}: {v}")

    dtype = {}
    for key in ["sender", "receiver", "laundering_type", "payment_currency", "received_currency", "payment_type", "sender_country", "receiver_country"]:
        if cols.get(key):
            dtype[cols[key]] = str

    out_jsonl = out_dir / "patterns.jsonl"
    out_offsets = out_dir / "patterns_offsets.json"
    out_unique = out_dir / "patterns_unique.csv"
    out_unsorted = out_dir / "patterns_index_unsorted.csv"
    out_sorted_legacy = out_dir / "patterns_sorted.csv"

    buckets: Dict[Tuple[object, ...], List[TxRow]] = defaultdict(list)
    bucket_sizes: Dict[Tuple[object, ...], int] = defaultdict(int)
    offsets: Dict[str, int] = {}
    index_rows: List[Dict[str, object]] = []
    counters = PipelineCounters()
    pattern_counter = 0
    offset_tx = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        reader = pd.read_csv(csv_path, chunksize=args.chunksize, dtype=dtype, low_memory=False)
        for chunk_no, chunk in enumerate(reader, start=1):
            rows, offset_tx = _read_rows_from_chunk(chunk, cols, offset_tx, args)
            if args.filter_laundering in {0, 1}:
                rows = [r for r in rows if r.is_laundering == args.filter_laundering]

            for row in rows:
                key = _bucket_key(row, args.bucket_mode)
                buckets[key].append(row)
                bucket_sizes[key] += 1

                if args.max_bucket_rows > 0 and bucket_sizes[key] >= args.max_bucket_rows:
                    counters.large_bucket_flushes += 1
                    pattern_counter = _extract_patterns_from_bucket(
                        buckets[key], pattern_counter, fout, offsets, index_rows, args, counters
                    )
                    buckets[key].clear()
                    bucket_sizes[key] = 0

            if chunk_no % args.log_every_chunks == 0:
                print(
                    f"chunk={chunk_no:,} emitted={counters.emitted:,} "
                    f"open_buckets={len(buckets):,} rows_seen={offset_tx:,}"
                )

        for key, rows in list(buckets.items()):
            if rows:
                pattern_counter = _extract_patterns_from_bucket(
                    rows, pattern_counter, fout, offsets, index_rows, args, counters
                )

    with open(out_offsets, "w", encoding="utf-8") as f:
        json.dump(offsets, f)

    index_df = pd.DataFrame(index_rows)
    if index_df.empty:
        raise RuntimeError("No valid patterns were emitted. Relax min/cap settings or inspect input columns.")

    sort_cols = ["n", "m_unique", "cycles", "depth", "max_out", "tx_count"]
    index_df = index_df.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    index_df.to_csv(out_unique, index=False)
    index_df.to_csv(out_unsorted, index=False)
    # Legacy name intentionally contains unique rows now, not oversampled rows.
    index_df.to_csv(out_sorted_legacy, index=False)

    split_summary = split_unique_index(index_df, args, out_dir)

    summary = {
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "window_size": args.window_size,
        "session_gap": args.session_gap,
        "bucket_mode": args.bucket_mode,
        "min_nodes": args.min_nodes,
        "min_tx": args.min_tx,
        "min_unique_edges": args.min_unique_edges,
        "max_nodes": args.max_nodes,
        "max_unique_edges": args.max_unique_edges,
        "max_tx": args.max_tx,
        "counters": counters.as_dict(),
        "splits": split_summary,
        "columns": cols,
        "laundering_type_counts_unique": index_df[index_df["is_laundering"] == 1]["laundering_type"].value_counts().to_dict(),
        "label_counts_unique": index_df["is_laundering"].value_counts().to_dict(),
    }
    with open(out_dir / "data_pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("DATA PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Unique patterns: {split_summary['unique']:,}")
    print(f"Train/val/test: {split_summary['train']:,} / {split_summary['val']:,} / {split_summary['test']:,}")
    print(f"Train laundering/normal: {split_summary['train_laundering']:,} / {split_summary['train_normal']:,}")
    print(f"Dropped/counters: {counters.as_dict()}")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_unique}")
    print(f"Wrote: {out_dir / 'patterns_train.csv'}")
    print(f"Wrote: {out_dir / 'patterns_stats_train_by_type.json'}")
    print("No physical oversampling was applied. Balance is handled by PatternSampler during training.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build AML graph patterns with unique sampling-safe splits.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Extract unique graph patterns and train/val/test splits.")
    b.add_argument("--csv-path", default="data/SAML-D.csv")
    b.add_argument("--out-dir", default="data")
    b.add_argument("--chunksize", type=int, default=200_000)
    b.add_argument("--window-size", default="7D", help="Pandas offset alias, e.g. 1D, 3D, 7D, 14D; or none.")
    b.add_argument("--session-gap", default="24h", help="Split connected components after inactivity gap; or none.")
    b.add_argument("--bucket-mode", default="window", choices=["window", "window_label", "window_type"])
    b.add_argument("--filter-laundering", type=int, choices=[0, 1], default=-1)

    b.add_argument("--min-nodes", type=int, default=3)
    b.add_argument("--min-tx", type=int, default=2)
    b.add_argument("--min-unique-edges", type=int, default=2)
    b.add_argument("--max-nodes", type=int, default=49, help="0 disables cap. Use max_nodes-1 from config.")
    b.add_argument("--max-unique-edges", type=int, default=95, help="0 disables cap. Use max_action safety margin.")
    b.add_argument("--max-tx", type=int, default=0, help="0 disables cap.")
    b.add_argument("--max-bucket-rows", type=int, default=300_000)

    b.add_argument("--val-frac", type=float, default=0.10)
    b.add_argument("--test-frac", type=float, default=0.10)
    b.add_argument("--split-seed", type=int, default=42)
    b.add_argument(
        "--train-laundering-types",
        default="",
        help="Comma-separated allow-list for laundering types in TRAIN only. Use for topology-representable Phase 1 types.",
    )
    b.add_argument(
        "--holdout-laundering-types",
        default="",
        help="Comma-separated laundering types forced into test for unseen-typology evaluation.",
    )
    b.add_argument("--log-every-chunks", type=int, default=10)
    b.set_defaults(func=build)
    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
