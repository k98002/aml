"""
Backward-compatible entry point for the AML data pipeline.

The previous implementation repeated laundering rows with LAUNDERING_MULTIPLIER.
That is intentionally removed. This wrapper delegates to data_pipeline.py and
keeps the old command name working:

    python utils/extract_sort.py

Recommended explicit command:

    python utils/data_pipeline.py build --csv-path data/SAML-D.csv --out-dir data
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support both `python utils/extract_sort.py` from project root and direct use
# from inside utils/.
UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from data_pipeline import main as _pipeline_main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend([
            "build",
            "--csv-path", str(UTILS_DIR.parent / "data" / "SAML-D.csv"),
            "--out-dir", str(UTILS_DIR.parent / "data"),
        ])
    _pipeline_main()
