# AML data pipeline patch

This patch removes physical oversampling and makes the training data split-safe.

## What changed

- `patterns_unique.csv` is one row per extracted graph pattern.
- `patterns_train.csv`, `patterns_val.csv`, and `patterns_test.csv` are split by unique `pattern_id` before any sampling.
- `patterns_stats_train_by_type.json` is computed from the train split only.
- Balancing happens online through `PatternSampler`, not by repeating rows.
- Optional transaction attributes are stored in `edge_attrs` when the SAML-D CSV contains them: amount, currencies, payment type, sender/receiver locations.
- The environment now preserves `edge_weight` during expert imitation instead of silently treating every edge as weight 1.

## Build data

From the `aml/` project root:

```bash
python utils/data_pipeline.py build \
  --csv-path data/SAML-D.csv \
  --out-dir data \
  --window-size 7D \
  --session-gap 24h \
  --max-nodes 49 \
  --max-unique-edges 95
```

`--window-size` should be swept experimentally, for example `1D`, `3D`, `7D`, `14D`.
Pick the smallest window that keeps enough laundering components while keeping `q90/q95(n)` and `q90/q95(m_unique)` inside the model budget.

## Optional Phase-1 typology filtering

For topology-only Phase 1, avoid training on typologies that require amount/time/invoice context.
Once you know the exact label names in your CSV, pass an allow-list:

```bash
python utils/data_pipeline.py build \
  --csv-path data/SAML-D.csv \
  --out-dir data \
  --train-laundering-types "fan_in,fan_out,chain,cycle"
```

The full unique index still exists in `patterns_unique.csv`; only the training split is filtered.

## Train

The patched config already points to:

```yaml
index_path: data/patterns_train.csv
stats_path: data/patterns_stats_train_by_type.json
sampling_mode: typology_complexity_balanced
```

Run:

```bash
python train_aml.py
```

## Output files

- `patterns.jsonl`: graph records with topology plus optional edge attributes.
- `patterns_offsets.json`: random-access offsets for the environment.
- `patterns_unique.csv`: all unique extracted patterns, no oversampling.
- `patterns_train.csv`: train split used by generator/detector training.
- `patterns_val.csv`: validation split.
- `patterns_test.csv`: held-out evaluation split.
- `patterns_stats_train_by_type.json`: train-only laundering stats for reward calibration.
- `data_pipeline_summary.json`: extraction counters, split counts, label/type counts.
