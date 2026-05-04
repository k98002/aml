"""
Pattern sampling utilities for AML graph-generation training.

This sampler fixes the previous data imbalance approach by sampling online from
unique pattern rows instead of physically duplicating laundering rows in the CSV.
It supports typology-balanced and typology+complexity-balanced sampling for the
expert-imitation phase used by the PPO/GCPN training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


NORMAL_TYPE = "__normal__"


@dataclass
class SamplerDiagnostics:
    """Lightweight counters useful for checking coverage during training."""

    total_samples: int = 0
    duplicate_pattern_rows_removed: int = 0


class PatternSampler:
    """
    Online sampler over a unique pattern index.

    Parameters
    ----------
    index_df:
        DataFrame with at least pattern_id, is_laundering, laundering_type,
        n, m_unique, cycles, depth, max_out, tx_count.
    mode:
        One of:
        - unique_uniform
        - label_balanced
        - typology_balanced
        - typology_complexity_balanced
    use_laundering_only:
        If true, normal rows are removed before sampling.
    include_laundering_types:
        Optional allow-list for laundering types. Empty/None means keep all.
    exclude_laundering_types:
        Optional deny-list for laundering types.
    complexity_bins:
        Number of within-typology size/complexity bins for balanced sampling.
    seed:
        Random seed.

    Notes
    -----
    The class stores original DataFrame row indices, so the environment can load
    patterns by its existing `load_pattern(idx)` method.
    """

    VALID_MODES = {
        "unique_uniform",
        "label_balanced",
        "typology_balanced",
        "typology_complexity_balanced",
        # Backward-compatible aliases.
        "uniform",
        "balanced_per_type",
    }

    def __init__(
        self,
        index_df: pd.DataFrame,
        mode: str = "typology_complexity_balanced",
        use_laundering_only: bool = True,
        include_laundering_types: Optional[Sequence[str]] = None,
        exclude_laundering_types: Optional[Sequence[str]] = None,
        complexity_bins: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown sampling mode: {mode}. Valid modes: {sorted(self.VALID_MODES)}")

        self.mode = self._normalise_mode(mode)
        self.rng = np.random.default_rng(seed)
        self.complexity_bins = max(int(complexity_bins), 1)
        self.diagnostics = SamplerDiagnostics()

        df = index_df.copy()
        if "pattern_id" not in df.columns:
            raise ValueError("index_df must contain a pattern_id column")
        if "is_laundering" not in df.columns:
            raise ValueError("index_df must contain an is_laundering column")

        before = len(df)
        df = df.drop_duplicates(subset=["pattern_id"], keep="first").copy()
        self.diagnostics.duplicate_pattern_rows_removed = before - len(df)

        if "laundering_type" not in df.columns:
            df["laundering_type"] = np.where(df["is_laundering"].astype(int) == 1, "unknown", "normal")
        df["laundering_type"] = df["laundering_type"].fillna("unknown").astype(str)
        df["is_laundering"] = df["is_laundering"].astype(int)

        include_set = {str(x) for x in include_laundering_types or [] if str(x).strip()}
        exclude_set = {str(x) for x in exclude_laundering_types or [] if str(x).strip()}

        if use_laundering_only:
            df = df[df["is_laundering"] == 1].copy()

        if include_set:
            keep = (df["is_laundering"] == 0) | (df["laundering_type"].isin(include_set))
            df = df[keep].copy()

        if exclude_set:
            drop = (df["is_laundering"] == 1) & (df["laundering_type"].isin(exclude_set))
            df = df[~drop].copy()

        if df.empty:
            raise ValueError(
                "No patterns left after sampler filtering. Check use_laundering_only and typology filters."
            )

        self.df = df
        self.indices = df.index.to_numpy()
        self._prepare_complexity()
        self._prepare_groups()

    @staticmethod
    def _normalise_mode(mode: str) -> str:
        if mode == "uniform":
            return "unique_uniform"
        if mode == "balanced_per_type":
            return "typology_balanced"
        return mode

    def _prepare_complexity(self) -> None:
        required = ["n", "m_unique", "cycles", "depth", "max_out", "tx_count"]
        for col in required:
            if col not in self.df.columns:
                self.df[col] = 0
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)

        # Smooth scalar used only for curriculum/binning, not as reward.
        self.df["_complexity_score"] = (
            self.df["n"]
            + self.df["m_unique"]
            + 0.5 * self.df["cycles"]
            + 0.5 * self.df["depth"]
            + 0.25 * self.df["max_out"]
            + 0.05 * self.df["tx_count"]
        )

    def _type_key(self, row: pd.Series) -> str:
        if int(row["is_laundering"]) == 1:
            return str(row["laundering_type"])
        return NORMAL_TYPE

    def _prepare_groups(self) -> None:
        self.type_to_indices: Dict[str, np.ndarray] = {}
        self.type_bin_to_indices: Dict[str, Dict[int, np.ndarray]] = {}
        self.label_to_indices: Dict[int, np.ndarray] = {}

        tmp = self.df.copy()
        tmp["_type_key"] = tmp.apply(self._type_key, axis=1)

        for label, part in tmp.groupby("is_laundering", sort=True):
            self.label_to_indices[int(label)] = part.index.to_numpy()

        for typ, part in tmp.groupby("_type_key", sort=True):
            part = part.sort_values("_complexity_score")
            idxs = part.index.to_numpy()
            self.type_to_indices[typ] = idxs
            self.type_bin_to_indices[typ] = self._make_bins(part)

        self.types = sorted(self.type_to_indices.keys())
        self.labels = sorted(self.label_to_indices.keys())

    def _make_bins(self, part: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Create stable within-group complexity bins."""
        n = len(part)
        if n == 0:
            return {}
        if self.complexity_bins <= 1 or n < self.complexity_bins:
            return {0: part.index.to_numpy()}

        ordered = part.sort_values("_complexity_score")
        bins: Dict[int, np.ndarray] = {}
        # Rank-split is more stable than qcut for tiny/duplicate groups.
        chunks = np.array_split(ordered.index.to_numpy(), self.complexity_bins)
        for b, arr in enumerate(chunks):
            if len(arr) > 0:
                bins[b] = arr
        return bins

    def __len__(self) -> int:
        return len(self.df)

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "num_rows": int(len(self.df)),
            "num_types": int(len(self.types)),
            "types": self.types,
            "labels": self.labels,
            "duplicates_removed": int(self.diagnostics.duplicate_pattern_rows_removed),
        }

    def sample_index(self, curriculum: bool = False, level_total: int = 6, level: int = 0) -> int:
        """Return one DataFrame row index compatible with env.load_pattern(idx)."""
        self.diagnostics.total_samples += 1

        if self.mode == "unique_uniform":
            return int(self._sample_from_indices(self.indices, curriculum, level_total, level))

        if self.mode == "label_balanced":
            label = int(self.rng.choice(self.labels))
            return int(self._sample_from_indices(self.label_to_indices[label], curriculum, level_total, level))

        if self.mode == "typology_balanced":
            typ = str(self.rng.choice(self.types))
            return int(self._sample_from_indices(self.type_to_indices[typ], curriculum, level_total, level))

        if self.mode == "typology_complexity_balanced":
            typ = str(self.rng.choice(self.types))
            bins = self.type_bin_to_indices[typ]
            available_bins = sorted(bins.keys())
            if curriculum and len(available_bins) > 1:
                # Curriculum chooses easier-to-harder bins over levels.
                frac = min(max((level + 1) / float(max(level_total, 1)), 0.0), 1.0)
                max_bin_pos = max(0, int(np.ceil(frac * len(available_bins))) - 1)
                candidate_bins = available_bins[: max_bin_pos + 1]
            else:
                candidate_bins = available_bins
            b = int(self.rng.choice(candidate_bins))
            return int(self.rng.choice(bins[b]))

        # Defensive fallback.
        return int(self.rng.choice(self.indices))

    def _sample_from_indices(
        self,
        idxs: Iterable[int],
        curriculum: bool,
        level_total: int,
        level: int,
    ) -> int:
        arr = np.asarray(list(idxs), dtype=int)
        if len(arr) == 0:
            raise RuntimeError("Cannot sample from an empty index group")
        if not curriculum or len(arr) <= 1:
            return int(self.rng.choice(arr))

        # Sort by complexity and sample from the current curriculum slice.
        part = self.df.loc[arr].sort_values("_complexity_score")
        ordered = part.index.to_numpy()
        level_total = max(int(level_total), 1)
        level = min(max(int(level), 0), level_total - 1)
        start = int(np.floor(level / float(level_total) * len(ordered)))
        end = int(np.floor((level + 1) / float(level_total) * len(ordered)))
        end = max(end, start + 1)
        return int(self.rng.choice(ordered[start:end]))
