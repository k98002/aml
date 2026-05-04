"""
Transaction Graph Environment for AML Pattern Generation
"""
import json
import numpy as np
import networkx as nx
import random
from pathlib import Path

try:
    from utils.pattern_sampler import PatternSampler
except ImportError:  # pragma: no cover - supports running from inside environment/
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
    from pattern_sampler import PatternSampler


class TransactionGraphEnv:
    """
    Environment for generating AML transaction graph patterns.
    - Loads patterns from JSONL
    - Edge type is transaction
    - Curriculum based on graph complexity (n, m, cycles, depth)
    """

    def __init__(self, config):
        """Initialize transaction graph environment."""
        self.config = config

        # Dataset paths
        self.dataset_path = Path(config.dataset_path)
        self.offsets_path = Path(config.offsets_path)

        # Load dataset index and offsets
        self._load_dataset()

        # Load dataset statistics for reward computation
        self._load_stats()

        # Graph parameters
        self.max_nodes = config.max_nodes
        self.min_nodes = config.min_nodes
        self.max_action = config.max_action
        self.min_action = config.min_action

        # Edge type (single type for transactions)
        self.edge_type_num = 1
        self.node_feature_dim = config.node_feature_dim  # Dummy features

        # Current state
        self.current_graph = None
        self.step_count = 0

        # Observation/action space (without batch dimension)
        self.observation_space = {
            'adj': np.zeros((self.edge_type_num, self.max_nodes, self.max_nodes)),
            'node': np.zeros((1, self.max_nodes, self.node_feature_dim))
        }
        self.action_space = {
            'shape': (4,),  # [node1, node2, edge_type, stop]
            'n': [self.max_nodes, self.max_nodes, self.edge_type_num, 2]
        }

    def _load_dataset(self):
        """Load unique dataset index, offsets, and online sampler."""
        import pandas as pd

        index_path = Path(self.config.index_path)
        self.dataset_index = pd.read_csv(index_path)

        if "pattern_id" not in self.dataset_index.columns:
            raise ValueError(f"Index file {index_path} must contain pattern_id")

        before = len(self.dataset_index)
        self.dataset_index = self.dataset_index.drop_duplicates(subset=["pattern_id"], keep="first").reset_index(drop=True)
        duplicates = before - len(self.dataset_index)
        if duplicates:
            print(f"Removed {duplicates} duplicate pattern rows from index. Use online sampling instead of physical oversampling.")

        # The sampler, not the CSV, is responsible for balancing. This keeps train/val/test leakage-safe.
        sampling_mode = getattr(
            self.config,
            "sampling_mode",
            getattr(self.config, "laundering_sampling", "unique_uniform"),
        )
        include_types = getattr(
            self.config,
            "representable_laundering_types",
            getattr(self.config, "include_laundering_types", []),
        )
        exclude_types = getattr(self.config, "exclude_laundering_types", [])
        complexity_bins = getattr(self.config, "complexity_bins", 3)
        seed = getattr(self.config, "seed", None)

        self.sampler = PatternSampler(
            self.dataset_index,
            mode=sampling_mode,
            use_laundering_only=bool(getattr(self.config, "use_laundering_only", False)),
            include_laundering_types=include_types,
            exclude_laundering_types=exclude_types,
            complexity_bins=complexity_bins,
            seed=seed,
        )

        # Kept for backwards compatibility with older code paths.
        self.dataset_len = len(self.sampler)
        self.type_to_indices = getattr(self.sampler, "type_to_indices", {})
        self.laundering_types = [t for t in getattr(self.sampler, "types", []) if t != "__normal__"]

        with open(self.offsets_path, 'r') as f:
            self.offsets = json.load(f)

        print(f"Loaded {len(self.dataset_index)} unique index rows from {index_path}")
        print(f"Sampler summary: {self.sampler.summary()}")

    def _load_stats(self):
        """Load dataset statistics for reward computation."""
        stats_path = Path(self.config.stats_path)
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded dataset statistics from {stats_path}")
        else:
            print(f"Warning: Stats file not found at {stats_path}")
            self.stats = {}

        # Print reward configuration for observability
        reward_type = getattr(self.config, 'reward_type', 'structural_smooth')
        target_stats_mode = getattr(self.config, 'target_stats_mode', 'global')
        print(f"Reward mode: {reward_type}")
        print(f"Target stats mode: {target_stats_mode}")

    def load_pattern(self, idx):
        """Load pattern by index from JSONL."""
        pattern_id = self.dataset_index.iloc[idx]['pattern_id']
        offset = self.offsets[pattern_id]

        with open(self.dataset_path, 'r') as f:
            f.seek(offset)
            pattern = json.loads(f.readline())

        return pattern

    def reset(self):
        """Reset environment to initial state with one seed node."""
        # Start with one seed node (node 0)
        self.current_graph = nx.DiGraph()
        self.current_graph.add_node(0)
        self.step_count = 0

        return self._get_observation()

    def _get_observation(self):
        """Convert current graph to observation format (without batch dimension).

        Node features (6D):
        - Real nodes (0 to n-1): [in_deg, out_deg, log_in, log_out, balance, is_new=1.0]
        - NEW_NODE placeholder (max_nodes-1): [0, 0, 0, 0, 0, is_new=1.0]
        - Invalid nodes (n to max_nodes-2): all zeros including is_new=0.0
        """
        n = self.current_graph.number_of_nodes()
        new_node_idx = self.max_nodes - 1

        # Initialize observation (shape matches observation_space)
        obs = {
            'adj': np.zeros((self.edge_type_num, self.max_nodes, self.max_nodes)),
            'node': np.zeros((1, self.max_nodes, self.node_feature_dim))
        }

        # Fill adjacency matrix
        node_list = list(self.current_graph.nodes())
        for i, u in enumerate(node_list):
            for j, v in enumerate(node_list):
                if self.current_graph.has_edge(u, v):
                    # Single edge type (index 0)
                    weight = self.current_graph[u][v].get('weight', 1)
                    obs['adj'][0, i, j] = weight

        # Compute node features (6 dimensions per node)
        for i, node in enumerate(node_list):
            in_deg = self.current_graph.in_degree(node)
            out_deg = self.current_graph.out_degree(node)

            # Compute total flow (weighted by edge weights)
            total_in = sum(self.current_graph[u][node].get('weight', 1)
                           for u in self.current_graph.predecessors(node))
            total_out = sum(self.current_graph[node][v].get('weight', 1)
                            for v in self.current_graph.successors(node))

            # Feature vector: [in_deg, out_deg, log_in, log_out, balance_ratio, is_new]
            # Normalize degrees by a reasonable maximum (10)
            feat_in_deg = min(in_deg / 10.0, 1.0)
            feat_out_deg = min(out_deg / 10.0, 1.0)

            # Log-scaled flow (assuming weight represents transaction count or amount proxy)
            feat_log_in = np.log10(total_in + 1) / 2.0  # Normalize by log10(100) ≈ 2
            feat_log_out = np.log10(total_out + 1) / 2.0

            # Balance ratio: out / (in + 1) to avoid division by zero
            feat_balance = total_out / (total_in + 1.0)
            feat_balance = min(feat_balance, 2.0) / 2.0  # Clip and normalize to [0, 1]

            # is_new = 1.0 for all real nodes (they are "existing" and selectable)
            feat_is_new = 1.0

            obs['node'][0, i, :] = np.array([
                feat_in_deg,
                feat_out_deg,
                feat_log_in,
                feat_log_out,
                feat_balance,
                feat_is_new,
            ])

        # NEW_NODE placeholder: [0, 0, 0, 0, 0, 1.0] (special token with is_new=1.0)
        obs['node'][0, new_node_idx, 5] = 1.0  # is_new marker for NEW_NODE

        return obs

    def step(self, action):
        """Take action in environment.

        Action: [node1, node2, edge_type, stop]
        - node1: existing node (0 to n-1)
        - node2: existing node (0 to n-1) OR NEW_NODE (max_nodes-1)
        - edge_type: edge type index (0 for single-type transaction graph)
        - stop: stop flag (0 or 1)

        If node2 == NEW_NODE, environment creates a single new node at index n
        and auto-rewrites the action to (node1, n).
        """
        node1, node2, edge_type, stop = int(action[0]), int(action[1]), int(action[2]), int(action[3])

        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        new_node_idx = self.max_nodes - 1

        # Check if stop action
        if stop == 1 and self.step_count >= self.min_action:
            done = True
            # Compute terminal reward (deliberate stop)
            reward = self._compute_terminal_reward(terminal_reason='stop')
            info['final_graph'] = self.current_graph.copy()
            info['stop_reason'] = 'stop'
            info['timed_out'] = False
            info.update(self._compute_graph_metrics())
            return self._get_observation(), reward, done, info

        # Check step limits
        if self.step_count >= self.max_action:
            done = True
            # Compute terminal reward (timeout - reached max steps)
            reward = self._compute_terminal_reward(terminal_reason='timeout')
            info['final_graph'] = self.current_graph.copy()
            info['stop_reason'] = 'timeout'
            info['timed_out'] = True
            info.update(self._compute_graph_metrics())
            return self._get_observation(), reward, done, info

        # Get current node count
        n_nodes = self.current_graph.number_of_nodes()

        # === VALIDATION: node1 must be an existing real node ===
        if not (isinstance(node1, (int, np.integer)) and 0 <= node1 < n_nodes):
            # Invalid: node1 is not an existing real node
            reward = -0.01
            return self._get_observation(), reward, done, info

        # === VALIDATION: reject node1 == NEW_NODE (safety check) ===
        if node1 == new_node_idx:
            reward = -0.01
            return self._get_observation(), reward, done, info

        # === PROCESS node2: NEW_NODE or existing node ===
        if node2 == new_node_idx:
            # NEW_NODE selected: create exactly one new node
            if n_nodes >= self.max_nodes - 1:
                # Graph is full; cannot create more nodes
                reward = -0.01
                return self._get_observation(), reward, done, info

            # Create new node at index n_nodes
            node2_processed = n_nodes
            self.current_graph.add_node(node2_processed)

        elif isinstance(node2, (int, np.integer)) and 0 <= node2 < n_nodes:
            # node2 is an existing real node
            node2_processed = node2

        else:
            # Invalid: node2 is neither NEW_NODE nor an existing real node
            reward = -0.01
            return self._get_observation(), reward, done, info

        # === ADD EDGE with processed indices ===
        if node1 == node2_processed:
            # Self-loop not allowed
            reward = -0.01
        elif self.current_graph.has_edge(node1, node2_processed):
            # Edge already exists (increment weight)
            self.current_graph[node1][node2_processed]['weight'] += 1
            reward = 0.005  # Reduced from 0.02 to prevent reward farming
        else:
            # Add new edge
            self.current_graph.add_edge(node1, node2_processed, weight=1)
            reward = 0.01  # Reduced from 0.05 to prevent reward farming

        return self._get_observation(), reward, done, info

    def _compute_terminal_reward(self, terminal_reason='stop'):
        """Compute reward for complete graph.

        Args:
            terminal_reason: 'stop' (deliberate) or 'timeout' (reached max_action)
        """
        if self.config.reward_type == 'structural':
            reward = self._structural_reward()
        elif self.config.reward_type == 'structural_smooth':
            reward = self._structural_reward_smooth()
        else:
            reward = 0.0

        # Apply timeout penalty if episode ended by timeout
        if terminal_reason == 'timeout':
            reward -= self.config.timeout_penalty

        return reward

    def _compute_graph_metrics(self):
        """Compute detailed graph metrics for observability."""
        g = self.current_graph
        n = g.number_of_nodes()
        m = g.number_of_edges()

        # Connectivity
        weakly_connected = nx.is_weakly_connected(g) if n > 0 else False
        num_components = nx.number_weakly_connected_components(g) if n > 0 else 0

        # Cycles
        try:
            cycles = len(list(nx.simple_cycles(g)))
        except:
            cycles = 0

        # Depth (diameter)
        try:
            if weakly_connected:
                ug = g.to_undirected()
                depth = nx.diameter(ug)
            else:
                depth = 0
        except:
            depth = 0

        # Max out-degree
        max_out = max(dict(g.out_degree()).values()) if n > 0 else 0

        return {
            'weakly_connected': weakly_connected,
            'num_components': num_components,
            'cycles': cycles,
            'depth': depth,
            'max_out': max_out
        }

    def _structural_reward(self):
        """Compute structural match reward."""
        n = self.current_graph.number_of_nodes()
        m = self.current_graph.number_of_edges()

        if n < self.min_nodes or m < 2:
            return -1.0  # Too small

        # Basic size reward
        if 5 <= n <= 20:
            reward = 1.0
        else:
            reward = 0.5

        # Connectivity bonus
        if nx.is_weakly_connected(self.current_graph):
            reward += 0.5

        return reward

    def _structural_reward_smooth(self):
        """Compute smooth structural match reward with exponential scoring."""
        g = self.current_graph
        n = g.number_of_nodes()
        m = g.number_of_edges()

        # Hard guardrails for invalid graphs
        if n < self.min_nodes or m < 2:
            return -1.0  # Too small, unviable graph

        # Size overflow penalty (graphs that are too large)
        if n > 30:  # Soft max threshold
            size_overflow = (n - 30) * self.config.size_overflow_penalty
        else:
            size_overflow = 0.0

        # Compute graph metrics
        try:
            cycles = len(list(nx.simple_cycles(g)))
        except:
            cycles = 0

        try:
            if nx.is_weakly_connected(g):
                # Depth is longest shortest path in undirected version
                ug = g.to_undirected()
                depth = nx.diameter(ug)
            else:
                depth = 0
        except:
            depth = 0

        max_out = max(dict(g.out_degree()).values()) if len(g.nodes()) > 0 else 0

        # Get target stats (global for now, per-type later)
        if self.config.target_stats_mode == 'global' and '_global' in self.stats:
            targets = self.stats['_global']
        else:
            # Fallback to reasonable defaults if stats not loaded
            targets = {
                'n': {'mean': 8.0, 'std': 3.0},
                'm_unique': {'mean': 7.0, 'std': 3.0},
                'cycles': {'mean': 0.5, 'std': 1.0},
                'depth': {'mean': 3.0, 'std': 2.0},
                'max_out': {'mean': 3.0, 'std': 2.0},
            }

        # Compute smooth match scores using exponential decay
        # Scale is 2x std to make scoring less sensitive
        scale_n = max(targets['n']['std'] * 2.0, 1.0)
        scale_m = max(targets['m_unique']['std'] * 2.0, 1.0)
        scale_cycles = max(targets['cycles']['std'] * 2.0, 0.5)
        scale_depth = max(targets['depth']['std'] * 2.0, 1.0)
        scale_max_out = max(targets['max_out']['std'] * 2.0, 1.0)

        score_n = np.exp(-abs(n - targets['n']['mean']) / scale_n)
        score_m = np.exp(-abs(m - targets['m_unique']['mean']) / scale_m)
        score_cycles = np.exp(-abs(cycles - targets['cycles']['mean']) / scale_cycles)
        score_depth = np.exp(-abs(depth - targets['depth']['mean']) / scale_depth)
        score_max_out = np.exp(-abs(max_out - targets['max_out']['mean']) / scale_max_out)

        # Weighted sum
        smooth_score = (
            self.config.reward_w_n * score_n +
            self.config.reward_w_m * score_m +
            self.config.reward_w_cycles * score_cycles +
            self.config.reward_w_depth * score_depth +
            self.config.reward_w_max_out * score_max_out
        )

        # Connectivity bonus/penalty
        if nx.is_weakly_connected(g):
            smooth_score += self.config.connected_bonus
        else:
            smooth_score -= self.config.disconnect_penalty

        # Apply size overflow penalty
        smooth_score -= size_overflow

        return smooth_score

    def get_expert(self, batch_size, is_final=False, curriculum=0, level_total=6, level=0):
        """
        Get expert demonstrations from dataset.

        Args:
            batch_size: Number of expert samples
            is_final: Whether to sample complete patterns
            curriculum: Whether to use curriculum learning
            level_total: Total curriculum levels
            level: Current curriculum level

        Returns:
            observations: Dict with 'adj' and 'node' (6D features with NEW_NODE)
            actions: Expert actions (batch_size, 4)
        """
        obs_batch = {
            'adj': np.zeros((batch_size, self.edge_type_num, self.max_nodes, self.max_nodes)),
            'node': np.zeros((batch_size, 1, self.max_nodes, self.node_feature_dim))
        }
        actions_batch = np.zeros((batch_size, 4))
        new_node_idx = self.max_nodes - 1

        for i in range(batch_size):
            # Sample pattern online from unique rows. Balance/curriculum are sampler concerns,
            # not physical duplicate rows in the CSV.
            idx = self.sampler.sample_index(
                curriculum=bool(curriculum),
                level_total=level_total,
                level=level,
            )

            # Load pattern
            pattern = self.load_pattern(idx)

            # Build graph from edge_index and preserve edge_weight as transaction-count flow proxy.
            full_graph = nx.DiGraph()
            full_graph.add_nodes_from(range(len(pattern['node_ids'])))
            edge_index = pattern.get('edge_index', [])
            edge_weight = pattern.get('edge_weight', [1] * len(edge_index))
            for e, w in zip(edge_index, edge_weight):
                u, v = int(e[0]), int(e[1])
                try:
                    weight = float(w)
                except Exception:
                    weight = 1.0
                full_graph.add_edge(u, v, weight=weight)

            edges = list(full_graph.edges())

            # Randomly decide to show stop action (25% of time)
            show_stop = random.random() < 0.25

            # Sample subgraph
            if show_stop or is_final or len(edges) <= 2:
                # Return complete graph + stop action
                subgraph_edges = edges
                is_stop = True
                next_edge = None
            else:
                # Sample random subgraph (remove some edges)
                num_edges_sub = random.randint(1, len(edges) - 1)
                subgraph_edges = random.sample(edges, k=num_edges_sub)

                # Find edge to add next
                remaining_edges = [e for e in edges if e not in subgraph_edges]
                next_edge = random.choice(remaining_edges)
                is_stop = False

            # Convert subgraph to observation and keep edge weights.
            subgraph = nx.DiGraph()
            for u, v in subgraph_edges:
                subgraph.add_edge(u, v, weight=full_graph[u][v].get('weight', 1.0))

            # Get nodes in current subgraph (don't include next_edge target yet)
            nodes_in_subgraph = set(subgraph.nodes())

            # Create sorted list and mapping from original IDs to indices
            nodes = sorted(list(nodes_in_subgraph))
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}

            # Fill observation using remapped indices
            for u_idx, u in enumerate(nodes):
                for v_idx, v in enumerate(nodes):
                    if subgraph.has_edge(u, v):
                        obs_batch['adj'][i, 0, u_idx, v_idx] = 1

            # Compute node features from subgraph (6D with is_new)
            for node_idx, node in enumerate(nodes):
                in_deg = subgraph.in_degree(node)
                out_deg = subgraph.out_degree(node)

                # Compute weighted flow
                total_in = sum(subgraph[u][node].get('weight', 1)
                               for u in subgraph.predecessors(node))
                total_out = sum(subgraph[node][v].get('weight', 1)
                                for v in subgraph.successors(node))

                # Same feature computation as _get_observation
                feat_in_deg = min(in_deg / 10.0, 1.0)
                feat_out_deg = min(out_deg / 10.0, 1.0)
                feat_log_in = np.log10(total_in + 1) / 2.0
                feat_log_out = np.log10(total_out + 1) / 2.0
                feat_balance = min(total_out / (total_in + 1.0), 2.0) / 2.0
                feat_is_new = 1.0  # is_new marker for real nodes

                obs_batch['node'][i, 0, node_idx, :] = np.array([
                    feat_in_deg,
                    feat_out_deg,
                    feat_log_in,
                    feat_log_out,
                    feat_balance,
                    feat_is_new
                ])

            # Add NEW_NODE placeholder (at max_nodes-1) with is_new=1.0
            obs_batch['node'][i, 0, new_node_idx, 5] = 1.0

            # Create action with remapped node indices
            if is_stop:
                # Stop action (any valid nodes)
                node1_idx = random.randint(0, max(1, len(nodes) - 1))
                node2_idx = random.randint(0, max(1, len(nodes) - 1))
                actions_batch[i] = [node1_idx, node2_idx, 0, 1]  # Stop
            else:
                # Remap next edge
                node1_orig, node2_orig = next_edge
                node1_idx = node_to_idx.get(node1_orig)
                node2_idx = node_to_idx.get(node2_orig)

                # If node1 not in subgraph, skip this sample (invalid)
                if node1_idx is None:
                    # Fallback: use stop action
                    actions_batch[i] = [0, 0, 0, 1]
                    continue

                # If node2 not in subgraph, use NEW_NODE token
                if node2_idx is None:
                    node2_idx = new_node_idx  # NEW_NODE placeholder

                actions_batch[i] = [node1_idx, node2_idx, 0, 0]  # Don't stop

        return obs_batch, actions_batch

    def seed(self, seed):
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
