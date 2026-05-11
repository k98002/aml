# AML Phase 1.5 Method and Runbook

## 0. Current smoke-test verdict

The latest short run is healthy enough to move to a larger controlled run.

Observed from the test log:

```text
Dataset files found:
  patterns.jsonl
  patterns_offsets.json
  patterns_train.csv
  patterns_stats_train.json
  patterns_metadata.json

Environment:
  Loaded attributed metadata
  Loaded 2642 training patterns
  Built balanced sampling lookup for 17 laundering types
  Loaded train statistics from patterns_stats_train.json
  Reward mode: phase15_target_attr
  Curriculum: True, 6 levels

Training:
  Curriculum advanced 0 -> 1
  Iter 10: Reward 0.71 | Len 15.1 | V/T/A 0.99/0.55/0.73 | H_amt/H_time 2.71/2.02
  Iter 20: Reward 0.85 | Len 48.4 | V/T/A 1.00/0.82/0.80 | H_amt/H_time 2.73/2.01
  Iter 30: Reward 0.62 | Len 17.4 | V/T/A 0.98/0.58/0.71 | H_amt/H_time 2.69/1.97
  Status: ok
```

Interpretation:

```text
Good signs:
  files load correctly
  attributed metadata is loaded
  train-only stats are loaded
  sampler is active
  reward mode is the new target/attribute reward
  curriculum advanced beyond Level 0
  validity is high
  target reward is nonzero
  attribute reward is nonzero
  amount/time entropy is not collapsed

Not yet proven:
  long-run convergence
  generated graph novelty
  held-out typology generalisation
  detector robustness
  stable behaviour after higher curriculum levels
```

Conclusion:

```text
Ready for a medium larger run.
Not yet ready to claim final AML behaviour generation.
```

Run a medium job first, then inspect diagnostics before a full-scale job.

---

## 1. What the method is now

The current method is no longer the original topology-only PPO generator.

It is now:

```text
leakage-safe attributed transaction-pattern extraction
+ unique-pattern online sampling
+ target-conditioned attributed graph generation
+ PPO/GCPN-style sequential transaction construction
+ attribute-aware terminal reward
+ curriculum and diagnostics
```

The generator learns a policy over partial transaction graphs.

At each step, it decides:

```text
op:        ADD_TX or STOP
src:       existing node or NEW_NODE
dst:       existing node or NEW_NODE
amount:    discrete amount bin
time:      discrete delta-time bin
currency:  categorical currency id
payment:   categorical payment type id
```

The policy is not directly asked to match five dataset statistics. It is rewarded for producing a valid, target-conditioned, attributed transaction graph with plausible AML-like topology and attribute behaviour.

---

## 2. Data extraction pipeline

### 2.1 Input

Raw SAML-D CSV.

Expected raw columns are inferred by aliases, including:

```text
sender account
receiver account
timestamp or date/time
is_laundering label
laundering_type
amount
currency
payment type / channel
sender / receiver country, if available
```

### 2.2 Extraction logic

The new pipeline extracts transaction graph patterns as temporal connected components.

Conceptually:

```python
def build_patterns(raw_csv):
    rows = read_raw_transactions(raw_csv)
    rows = parse_sender_receiver_label_time_amount_currency_type(rows)

    for temporal_window in group_by_time_window(rows):
        components = connected_components(temporal_window)

        for component in components:
            sessions = split_by_inactivity_gap(component)

            for session in sessions:
                subcomponents = connected_components(session)

                for subgraph_rows in subcomponents:
                    if not passes_min_caps(subgraph_rows):
                        continue
                    if exceeds_model_caps(subgraph_rows):
                        continue

                    pattern = build_pattern(subgraph_rows)
                    write_jsonl(pattern)
                    write_index_row(pattern)
```

Key design choices:

```text
Time window defines candidate activity period.
Connected component defines related accounts.
Session gap prevents unrelated activity from merging.
Node/edge/transaction caps keep patterns inside generator budget.
Pattern IDs are unique.
No physical laundering oversampling is applied.
```

### 2.3 Why time windows, not fixed transaction counts

The pipeline does not force every sample to contain exactly `K` transactions.

It uses:

```text
time window + connected component + session split + graph-size caps
```

because fixed transaction-count windows distort AML temporal behaviour.

Example:

```text
50 transactions in a high-activity account cluster may cover 3 minutes.
50 transactions in a low-activity cluster may cover 3 weeks.
```

The model needs coherent transaction behaviour, not arbitrary row chunks.

---

## 3. Data files and purpose

After preprocessing, the important files are:

```text
data/patterns.jsonl
data/patterns_offsets.json
data/patterns_unique.csv
data/patterns_train.csv
data/patterns_val.csv
data/patterns_test.csv
data/patterns_metadata.json
data/patterns_stats_train.json
data/patterns_stats_train_by_type.json
data/data_pipeline_summary.json
```

### 3.1 `patterns.jsonl`

Primary pattern store.

Each line is one extracted graph pattern.

Contains:

```text
pattern_id
is_laundering
laundering_type
node_ids / node_map
edge_index / edge_weight
transactions or edge_events
complexity metrics
start/end time
tx_count
amount totals
```

Example shape:

```json
{
  "pattern_id": "P0000123",
  "is_laundering": 1,
  "laundering_type": "fan_out",
  "node_ids": ["A", "B", "C"],
  "transactions": [
    {
      "src": 0,
      "dst": 1,
      "amount": 1200.0,
      "amount_bin": 6,
      "timestamp": "2023-01-01T10:05:00",
      "delta_t": 0.0,
      "delta_t_bin": 0,
      "currency_id": 0,
      "payment_type_id": 2
    },
    {
      "src": 1,
      "dst": 2,
      "amount": 1100.0,
      "amount_bin": 6,
      "timestamp": "2023-01-01T10:40:00",
      "delta_t": 2100.0,
      "delta_t_bin": 2,
      "currency_id": 0,
      "payment_type_id": 2
    }
  ],
  "complexity": {
    "n": 3,
    "m_unique": 2,
    "cycles": 0,
    "depth": 2,
    "max_out": 1
  }
}
```

The environment loads full pattern records from this file.

### 3.2 `patterns_offsets.json`

Fast lookup table.

Maps:

```text
pattern_id -> byte offset in patterns.jsonl
```

Used so the environment does not scan the whole JSONL file when loading one pattern.

Pseudo-code:

```python
def load_pattern(pattern_id):
    offset = offsets[pattern_id]
    file.seek(offset)
    return json.loads(file.readline())
```

### 3.3 `patterns_unique.csv`

One row per unique extracted pattern.

Used for:

```text
inspection
summary counts
manual debugging
window-size comparison
checking typology distribution
```

It should not contain duplicated physical oversampling rows.

### 3.4 `patterns_train.csv`

Training index.

Used by:

```text
PatternSampler
expert imitation
PPO environment reset
reward calibration linkage
```

Contains only pattern IDs from the train split.

No validation/test pattern IDs should appear here.

### 3.5 `patterns_val.csv`

Validation index.

Used for:

```text
reward diagnostics
model selection
checking generated-vs-real validation distribution
```

Not used for PPO training.

### 3.6 `patterns_test.csv`

Held-out evaluation index.

Used for:

```text
final evaluation
held-out typology checks
novelty / non-memorization tests
```

Not used for generator training.

### 3.7 `patterns_metadata.json`

Attribute vocabulary and bin metadata.

Contains train-fitted encodings:

```text
amount_bin_num
delta_t_bin_num
amount_bin_edges
amount_bin_centers
delta_t_bin_edges
delta_t_bin_centers
currency_vocab
payment_type_vocab
```

Used by the environment to map discrete action bins into numeric transaction attributes.

Example:

```python
def decode_action(action, metadata):
    amount = metadata["amount_bin_centers"][action.amount_bin]
    delta_t = metadata["delta_t_bin_centers"][action.delta_t_bin]
    currency = inv_vocab(metadata["currency_vocab"])[action.currency_id]
    payment_type = inv_vocab(metadata["payment_type_vocab"])[action.payment_type_id]
    return amount, delta_t, currency, payment_type
```

### 3.8 `patterns_stats_train.json`

Train-only global reward calibration statistics.

Used by reward realism terms.

Examples:

```text
node-count distribution
edge-count distribution
transaction-count distribution
amount distribution
time / duration distribution
currency/payment distribution, if included
```

Important rule:

```text
Reward calibration must use train-only stats.
```

Validation/test statistics should not leak into training.

### 3.9 `patterns_stats_train_by_type.json`

Train-only stats grouped by laundering type.

Used for:

```text
typology-specific realism comparison
curriculum / diagnostic reporting
typology-level sampling sanity checks
```

### 3.10 `data_pipeline_summary.json`

Audit file.

Contains:

```text
pipeline arguments
window size
session gap
caps
number of unique patterns
train/val/test counts
laundering type counts
dropped pattern counts
resolved raw CSV columns
```

Used to prove what preprocessing actually did.

---

## 4. Online sampler

The old approach duplicated laundering rows.

The new approach samples online from unique patterns.

### 4.1 Why

Old logic:

```text
690 suspicious patterns × 10 = 6900 rows
```

But information content remains:

```text
690 unique suspicious patterns
```

The new sampler avoids fake data expansion.

### 4.2 Sampler modes

Supported modes:

```text
unique_uniform
label_balanced
typology_balanced
typology_complexity_balanced
```

Recommended for current generator:

```text
typology_complexity_balanced
```

### 4.3 Sampler pseudo-code

```python
class PatternSampler:
    def __init__(index_df):
        df = drop_duplicate_pattern_ids(index_df)
        df = apply_train_typology_filters(df)
        groups = group_by_laundering_type(df)
        bins = split_each_type_by_complexity(groups)

    def sample_index(level):
        typology = uniform_sample(types)
        complexity_bin = sample_allowed_bin(typology, level)
        pattern_row = uniform_sample(patterns[typology][complexity_bin])
        return pattern_row.index
```

Effect:

```text
rare typologies get exposure
large/common typologies do not dominate automatically
curriculum can start with simpler patterns
no train/val/test leakage from duplicated rows
```

---

## 5. Environment and observation construction

The environment is now an attributed transaction graph MDP.

### 5.1 State

The state is the current partial transaction graph:

```text
G_t = generated transactions so far
```

The environment exposes:

```text
node features
edge/adjacency channels
global target vector
valid action masks
```

### 5.2 Node features

Node attributes are derived from generated transaction events.

Examples:

```text
in_degree
out_degree
in_tx_count
out_tx_count
log_in_amount
log_out_amount
balance_ratio
pass_through_ratio
time_span
recency
source_role_score
intermediary_role_score
sink_role_score
real_node_marker
new_node_marker
```

Example node feature computation:

```python
def node_features(node, transactions):
    incoming = [tx for tx in transactions if tx.dst == node]
    outgoing = [tx for tx in transactions if tx.src == node]

    in_amount = sum(tx.amount for tx in incoming)
    out_amount = sum(tx.amount for tx in outgoing)

    pass_through = min(in_amount, out_amount) / max(in_amount, out_amount, eps)

    return [
        len(unique_src(incoming)),
        len(unique_dst(outgoing)),
        len(incoming),
        len(outgoing),
        log1p(in_amount),
        log1p(out_amount),
        out_amount / max(in_amount, eps),
        pass_through,
        time_span(incoming + outgoing),
        recency(incoming + outgoing),
        source_score(incoming, outgoing),
        intermediary_score(incoming, outgoing),
        sink_score(incoming, outgoing),
        1.0,
        0.0,
    ]
```

### 5.3 Edge / adjacency channels

Edge channels aggregate transaction events between account pairs.

Examples:

```text
edge_exists
tx_count
log_amount_sum
log_amount_mean
recentness
currency distribution
payment-type distribution
```

Pseudo-code:

```python
def edge_channels(u, v, transactions):
    txs = [tx for tx in transactions if tx.src == u and tx.dst == v]
    if not txs:
        return zeros(edge_channel_dim)

    return [
        1.0,
        log1p(len(txs)),
        log1p(sum(tx.amount for tx in txs)),
        log1p(mean(tx.amount for tx in txs)),
        recentness(max(tx.timestamp for tx in txs)),
        currency_counts(txs),
        payment_type_counts(txs),
    ]
```

### 5.4 Global target vector

Each episode samples a target behaviour:

```text
chain_layering
fan_in_aggregation
fan_out_dispersal
cycle
pass_through_intermediary
```

The target becomes part of the observation:

```python
obs["global"] = concat(
    one_hot(target_behavior),
    [curriculum_level, target_budget, max_steps_remaining]
)
```

This prevents the same generic graph from scoring well for every episode.

---

## 6. Action space

Current action is all-discrete.

```text
a_t = [op, src, dst, amount_bin, delta_t_bin, currency_id, payment_type_id]
```

Meaning:

```text
op:
  0 = ADD_TX
  1 = STOP

src:
  existing node id or NEW_NODE

dst:
  existing node id or NEW_NODE

amount_bin:
  discrete amount interval fitted from train data

delta_t_bin:
  discrete time-gap interval fitted from train data

currency_id:
  categorical currency id

payment_type_id:
  categorical transaction/payment type id
```

### 6.1 Valid endpoint rules

Allowed:

```text
existing -> existing
existing -> NEW_NODE
NEW_NODE -> existing
```

Usually disallowed:

```text
NEW_NODE -> NEW_NODE
```

Reason:

```text
NEW_NODE -> NEW_NODE creates a disconnected edge unless special disconnected construction is supported.
```

### 6.2 Action execution

```python
def step(action):
    if action.op == STOP:
        done = True
        reward = terminal_reward(current_graph)
        return obs, reward, done, info

    src = resolve_endpoint(action.src)
    dst = resolve_endpoint(action.dst)

    if invalid(src, dst, action):
        reward = invalid_action_penalty
        return obs, reward, False, info

    tx = Transaction(
        src=src,
        dst=dst,
        amount=metadata.amount_bin_centers[action.amount_bin],
        delta_t=metadata.delta_t_bin_centers[action.delta_t_bin],
        currency_id=action.currency_id,
        payment_type_id=action.payment_type_id,
    )

    transactions.append(tx)
    update_graph_aggregates(tx)
    obs = build_observation(transactions)
    return obs, step_reward, False, info
```

---

## 7. Policy model

The policy consumes:

```text
node feature tensor
edge/adjacency tensor
global target vector
valid action masks
```

It outputs a masked multi-head categorical distribution.

### 7.1 Policy heads

```text
op head
src head
dst head
amount_bin head
delta_t_bin head
currency head
payment_type head
value head
```

### 7.2 Autoregressive structure

For STOP:

```text
log_prob(action) = log_prob(op=STOP)
```

For ADD_TX:

```text
log_prob(action) =
    log_prob(op=ADD_TX)
  + log_prob(src)
  + log_prob(dst | src)
  + log_prob(amount_bin)
  + log_prob(delta_t_bin)
  + log_prob(currency_id)
  + log_prob(payment_type_id)
```

Pseudo-code:

```python
def sample_action(obs):
    graph_emb = graph_encoder(obs.node, obs.adj)
    target_emb = global_encoder(obs.global)
    h = concat(graph_emb, target_emb)

    op = Categorical(mask(op_logits(h), obs.op_mask)).sample()

    if op == STOP:
        return Action(op=STOP), logp_op, entropy_op, value(h)

    src = Categorical(mask(src_logits(h), obs.src_mask)).sample()
    dst = Categorical(mask(dst_logits(h, src), obs.dst_mask[src])).sample()

    amount_bin = Categorical(amount_logits(h, src, dst)).sample()
    delta_t_bin = Categorical(time_logits(h, src, dst)).sample()
    currency_id = Categorical(currency_logits(h, src, dst)).sample()
    payment_type_id = Categorical(payment_logits(h, src, dst)).sample()

    logp = sum_component_log_probs()
    entropy = weighted_component_entropy()
    return action, logp, entropy, value(h)
```

---

## 8. Expert imitation phase

Before PPO, the model performs expert imitation from real extracted patterns.

Purpose:

```text
teach basic action syntax
teach STOP timing
teach endpoint selection
teach realistic amount/time/currency/payment choices
reduce random invalid exploration
```

### 8.1 Expert trace construction

```python
def sample_expert_example(pattern):
    txs = sort_by_timestamp(pattern.transactions)
    k = random_prefix_length(0, len(txs))

    obs = build_observation(txs[:k])

    if k == len(txs):
        target_action = Action(op=STOP)
    else:
        next_tx = txs[k]
        target_action = encode_transaction_as_action(next_tx, current_prefix=txs[:k])

    return obs, target_action
```

### 8.2 Expert loss masking

STOP examples train only the operation head.

ADD_TX examples train:

```text
op
src
dst
amount_bin
delta_t_bin
currency_id
payment_type_id
```

Pseudo-code:

```python
def expert_loss(batch):
    loss = 0
    for obs, action in batch:
        pred = policy.evaluate_action(obs, action)
        loss += CE(pred.op, action.op)

        if action.op == ADD_TX:
            loss += CE(pred.src, action.src)
            loss += CE(pred.dst, action.dst)
            loss += CE(pred.amount_bin, action.amount_bin)
            loss += CE(pred.delta_t_bin, action.delta_t_bin)
            loss += CE(pred.currency_id, action.currency_id)
            loss += CE(pred.payment_type_id, action.payment_type_id)

    return loss / len(batch)
```

---

## 9. PPO training loop

After expert imitation, PPO collects rollouts and updates the policy.

### 9.1 PPO rollout

```python
def collect_rollout(env, policy, T):
    obs = env.reset()
    buffer = []

    for t in range(T):
        action, logp, entropy, value = policy.sample_action(obs)
        next_obs, reward, done, info = env.step(action)

        buffer.append({
            "obs": obs,
            "action": action,
            "logp": logp,
            "value": value,
            "reward": reward,
            "done": done,
            "info": info,
        })

        obs = env.reset() if done else next_obs

    return buffer
```

### 9.2 PPO update

```python
def ppo_update(buffer):
    advantages = compute_gae(buffer.rewards, buffer.values, buffer.dones)
    returns = advantages + buffer.values

    for epoch in range(optim_epochs):
        for batch in minibatches(buffer):
            new_logp, entropy, value = policy.evaluate_action(batch.obs, batch.action)
            ratio = exp(new_logp - batch.old_logp)

            unclipped = ratio * batch.advantages
            clipped = clip(ratio, 1 - eps, 1 + eps) * batch.advantages
            policy_loss = -mean(min(unclipped, clipped))

            value_loss = mse(value, batch.returns)
            entropy_bonus = weighted_entropy(entropy)

            loss = policy_loss + vf_coef * value_loss - entropy_bonus
            optimizer.step(loss)
```

---

## 10. Reward function

Current reward mode:

```yaml
reward_type: phase15_target_attr
```

### 10.1 Reward structure

Terminal reward:

```python
if not hard_valid(G):
    R = -1.0
else:
    R = (
        w_target * r_target_aml(G, target)
      + w_attr   * r_attribute_behavior(G, target)
      + w_real   * r_realism(G)
      + w_novel  * r_novelty(G)
      + w_stop   * r_stop_quality(G)
      - w_degen  * r_anti_degenerate(G)
    )
```

Validity is mostly a gate.

It should not be a large positive reward source because then the agent can win by simply producing any valid connected graph.

### 10.2 Target-conditioned AML reward

Each target activates different topology and attribute terms.

```python
def r_target_aml(G, target):
    if target == "chain_layering":
        return mean([
            chain_depth_score(G),
            pass_through_score(G),
            temporal_order_score(G),
        ])

    if target == "fan_in_aggregation":
        return mean([
            fan_in_score(G),
            sink_role_score(G),
            amount_merge_score(G),
        ])

    if target == "fan_out_dispersal":
        return mean([
            fan_out_score(G),
            source_role_score(G),
            amount_split_score(G),
        ])

    if target == "cycle":
        return mean([
            cycle_score(G),
            amount_circulation_score(G),
            time_consistency_score(G),
        ])

    if target == "pass_through_intermediary":
        return mean([
            intermediary_score(G),
            pass_through_score(G),
            chain_continuation_score(G),
        ])
```

### 10.3 Attribute reward

Attribute reward checks that amount/time/currency/payment choices matter.

Examples:

```python
def pass_through_score(node):
    in_amt = node.in_amount
    out_amt = node.out_amount
    if in_amt <= 0 or out_amt <= 0:
        return 0.0
    return min(in_amt, out_amt) / max(in_amt, out_amt)


def amount_split_score(source):
    outgoing = source.out_transactions
    if len(outgoing) < 2:
        return 0.0
    diversity = normalized_entropy([tx.amount_bin for tx in outgoing])
    amount_plausibility = train_amount_bin_likelihood(outgoing)
    return 0.5 * diversity + 0.5 * amount_plausibility


def time_velocity_score(path):
    gaps = [tx.delta_t_bin for tx in path]
    ordered = all(g >= 0 for g in gaps)
    plausible = train_time_bin_likelihood(gaps)
    return ordered * plausible
```

### 10.4 Realism reward

Realism keeps generated graphs near train distribution.

It should answer:

```text
Does this graph look statistically plausible?
```

It should not answer:

```text
Is this laundering?
```

Examples:

```python
r_realism = mean([
    size_plausibility(G.n, train_stats.n),
    edge_plausibility(G.m, train_stats.m_unique),
    tx_count_plausibility(G.tx_count, train_stats.tx_count),
    amount_bin_likelihood(G.amount_bins, train_metadata),
    time_bin_likelihood(G.delta_t_bins, train_metadata),
])
```

### 10.5 Novelty reward

Current novelty is a proxy.

Recommended future stronger version:

```python
r_novelty = bounded_nearest_neighbor_distance(
    embedding(G),
    train_suspicious_embeddings
)
```

Good novelty means:

```text
not a duplicate
not wildly unrealistic
still close enough to AML-like behaviour
```

---

## 11. Curriculum

Current levels:

```text
Level 0: valid attributed graph completion
Level 1: target-conditioned topology behaviour
Level 2+: amount/time/currency/payment attributes enter more strongly
Level 3: typology-balanced generation difficulty
Level 4: novelty pressure
Level 5: future detector/adversarial pressure
```

Advancement logic:

```python
if (
    rolling_reward >= reward_threshold
    and rolling_validity >= validity_threshold
    and rolling_target >= target_threshold
    and rolling_attribute >= attribute_threshold
):
    level += 1
```

Your smoke run already advanced:

```text
Level 0 -> Level 1
```

That is materially better than the previous failure mode where Level stayed at 0.

---

## 12. Diagnostic interpretation

### 12.1 `V/T/A`

```text
V = validity component
T = target-conditioned behaviour component
A = attribute behaviour component
```

Interpretation:

```text
V high, T low, A low:
  model is only learning valid graphs

V high, T high, A low:
  model is topology-aware but attribute-light

V high, T high, A high:
  desired direction

V low:
  environment/action constraints are not being learned
```

### 12.2 `H_amt/H_time`

Amount/time entropy.

Interpretation:

```text
very high forever:
  amount/time heads may be irrelevant or undertrained

near zero too early:
  amount/time heads collapsed to one bin

moderate and changing:
  healthy exploration
```

Your smoke run:

```text
H_amt ≈ 2.69–2.73
H_time ≈ 1.97–2.02
```

This suggests no immediate collapse.

### 12.3 Length

Your smoke run had one long-episode spike:

```text
Iter 20 Len 48.4
```

Watch this in larger training.

Bad signs:

```text
Len stays near max_action
reward high despite very long graphs
STOP entropy collapses
```

---

## 13. Larger run commands

Run from the directory containing `train_aml.py` and the `data/` folder.

### 13.1 Medium run

```bash
python train_aml.py \
  device=cuda \
  fresh_csv=true \
  num_steps=100000 \
  timesteps_per_batch=1024 \
  optim_epochs=4 \
  optim_batchsize=128 \
  expert_end=200 \
  rl_start=1 \
  save_every=5000 \
  hydra.run.dir=. \
  hydra.output_subdir=null
```

CPU fallback:

```bash
python train_aml.py \
  device=cpu \
  fresh_csv=true \
  num_steps=50000 \
  timesteps_per_batch=512 \
  optim_epochs=2 \
  optim_batchsize=64 \
  expert_end=100 \
  rl_start=1 \
  save_every=5000 \
  hydra.run.dir=. \
  hydra.output_subdir=null
```

### 13.2 Full run after medium run passes

```bash
python train_aml.py \
  device=cuda \
  fresh_csv=true \
  num_steps=500000 \
  timesteps_per_batch=2048 \
  optim_epochs=4 \
  optim_batchsize=256 \
  expert_end=500 \
  rl_start=1 \
  save_every=10000 \
  hydra.run.dir=. \
  hydra.output_subdir=null
```

---

## 14. Larger run pass criteria

A larger run is healthy if:

```text
1. Level progresses beyond 1.
2. V stays high, usually > 0.95.
3. T improves or remains materially nonzero.
4. A improves or remains materially nonzero.
5. H_amt and H_time do not collapse immediately to 0.
6. Length does not stay pinned near max_action.
7. Generated CSV has nonzero r_amount_flow and r_time_velocity.
8. Attribute histograms are not single-bin collapsed.
9. Reward diagnostics show real-pattern replay above random and zero-attribute ablation.
```

A larger run is unhealthy if:

```text
reward high + T low + A low:
  still validity/topology shortcut

reward high + H_amt/H_time near max forever:
  attributes may be ignored

reward high + H_amt/H_time near 0 from early training:
  attribute mode collapse

Level 1 forever after many steps:
  target/attribute thresholds too hard or reward is not reachable

Len near max_action for most episodes:
  STOP policy is weak or reward incentivizes graph bloat
```

---

## 15. Post-run inspection commands

### 15.1 Inspect generated CSV

```bash
python - <<'PY'
import glob
import pandas as pd

paths = sorted(glob.glob("generated_graphs/*.csv"))
print("latest:", paths[-1])
df = pd.read_csv(paths[-1], comment="#")

cols = [
    "iteration", "level", "target_name", "reward",
    "r_valid", "r_target_aml", "r_attribute_behavior",
    "r_amount_flow", "r_time_velocity", "r_realism", "r_novelty",
    "target_match_rate", "len"
]
cols = [c for c in cols if c in df.columns]
print(df[cols].tail(30))

num_cols = [c for c in ["reward", "r_target_aml", "r_attribute_behavior", "r_amount_flow", "r_time_velocity", "target_match_rate"] if c in df.columns]
print(df.groupby("level")[num_cols].mean())
PY
```

### 15.2 Reward diagnostics

```bash
python utils/reward_diagnostics.py \
  --config config/train_aml.yaml \
  --episodes 300 \
  --real-limit 300
```

Expected:

```text
real-pattern replay > random policy
real-pattern replay > zero-attribute ablation
zero-attribute ablation should not score close to real replay
```

### 15.3 Split leakage check

```bash
python utils/verify_phase15_data.py --data-dir data
```

Expected:

```text
OK split leakage check
OK transaction attributes and metadata
```

---

## 16. What can be claimed now

Defensible Phase 1.5 claim:

```text
The system implements a leakage-safe attributed transaction-graph generation pipeline.
It trains a PPO/GCPN-style policy to sequentially construct target-conditioned transaction graphs using amount/time/currency/payment attributes.
The reward uses validity gating, target-conditioned AML topology priors, attribute-behaviour terms, train-data realism, novelty proxy, and stop-quality control.
```

Do not claim yet:

```text
The generator discovers truly new laundering typologies.
The generated graphs are fully realistic laundering cases.
The model proves detector robustness.
The generator is independent of known suspicious labels.
```

Next valid research claim after larger runs:

```text
The generator produces valid attributed graph variations under target-conditioned AML-like behaviours and can be used as a structured stress-test source for detector development.
```

---

## 17. Current phase diagram

```text
Raw SAML-D CSV
    ↓
Issue 1 data pipeline
    - temporal windows
    - connected components
    - session split
    - unique patterns
    - train/val/test split
    - train-only bins/vocabs/stats
    ↓
patterns.jsonl + offsets + train index + metadata + stats
    ↓
PatternSampler
    - typology-balanced
    - complexity-balanced
    - no physical oversampling
    ↓
TransactionGraphEnv
    - partial attributed graph state
    - node features from event aggregates
    - edge channels from transaction aggregates
    - global target behaviour
    ↓
Policy
    - graph encoder
    - global target encoder
    - masked multi-head categorical action
    ↓
Expert imitation
    - real pattern prefix → next transaction action
    ↓
PPO
    - rollout collection
    - terminal target/attribute reward
    - curriculum progression
    ↓
Generated attributed AML-like transaction graphs
    ↓
Diagnostics
    - V/T/A
    - entropy
    - histograms
    - random/zero-attribute/real replay comparison
```

---

## 18. Why this is still RL, not direct rule generation

The reward now evaluates final graph behaviour; it does not directly prescribe the construction sequence.

The policy must learn:

```text
which node to extend
whether to create source or destination
when to split
when to merge
when to chain
which amount bin to choose
which time gap to choose
when to stop
```

This remains a sequential decision problem:

```text
s_t = partial transaction graph

a_t = add attributed transaction or stop

R_T = validity + target behaviour + attribute behaviour + realism + novelty + stop quality
```

The purpose of the reward is to rank completed scenarios, not to hand the agent an explicit construction script.

---

## 19. Immediate next step

Run the medium job.

Primary watch fields:

```text
Level
Reward
Len
V/T/A
H_amt/H_time
r_amount_flow
r_time_velocity
target_match_rate
```

Proceed to the full run only if the medium run avoids:

```text
Level 1 forever
length pinned near max
attribute entropy collapse
attribute reward near zero
random/zero-attribute reward close to real replay reward
```
