"""
Local fallback for dataset availability checks.

The original project imported s3_utils, but the uploaded zip did not include it.
This fallback intentionally does not attempt any remote download; it simply checks
that expected files exist and prints clear instructions otherwise.
"""

from pathlib import Path
from typing import Iterable


DATASET_FILES = (
    "patterns.jsonl",
    "patterns_offsets.json",
    "patterns_train.csv",
    "patterns_stats_train_by_type.json",
)

LEGACY_DATASET_FILES = (
    "patterns.jsonl",
    "patterns_offsets.json",
    "patterns_sorted.csv",
    "patterns_stats_by_type.json",
)


def ensure_saml_d_csv(out_dir: str, verbose: bool = True) -> bool:
    path = Path(out_dir) / "SAML-D.csv"
    ok = path.exists()
    if verbose and not ok:
        print(f"SAML-D.csv not found at {path}. Place the Kaggle CSV there or pass --csv-path to data_pipeline.py.")
    return ok


def _all_exist(base: Path, names: Iterable[str]) -> bool:
    return all((base / name).exists() for name in names)


def ensure_dataset_files(data_dir: str, verbose: bool = True) -> bool:
    base = Path(data_dir)
    ok_new = _all_exist(base, DATASET_FILES)
    ok_legacy = _all_exist(base, LEGACY_DATASET_FILES)
    if ok_new or ok_legacy:
        return True
    if verbose:
        print(f"Dataset files not found under {base}.")
        print("Run: python utils/data_pipeline.py build --csv-path data/SAML-D.csv --out-dir data")
    return False
