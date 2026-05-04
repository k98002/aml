"""
GCN Policy Network for molecular graph generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .gcn import GCNStack

class GCNPolicy(nn.Module):
    """
    GCN-based policy network for molecular graph generation.

    The policy outputs:
    1. Stop action (binary)
    2. First node selection (categorical over nodes)
    3. Second node selection (categorical over nodes, conditioned on first node)
    4. Bond type selection (categorical over bond types)
    """

    def __init__(self, observation_space, action_space, atom_type_num, config):
        super(GCNPolicy, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.atom_type_num = atom_type_num
        self.config = config

        # Dimensions
        self.node_feature_dim = observation_space['node'].shape[2]  # d_n
        self.edge_dim = observation_space['adj'].shape[0]           # number of bond types
        self.emb_size = config.emb_size

        # Node embedding layer (projects atom features to embedding dimension)
        self.node_embedding = nn.Linear(self.node_feature_dim, 8, bias=False)

        # GCN stack
        self.gcn_stack = GCNStack(
            num_layers=config.layer_num_g,
            in_channels=8,
            hidden_channels=config.emb_size,
            edge_dim=self.edge_dim,
            aggregate=config.gcn_aggregate,
            has_residual=config.has_residual,
            has_concat=config.has_concat,
            bn=config.bn
        )

        # Output embedding dimension (may include graph embedding)
        if config.graph_emb:
            self.final_emb_dim = config.emb_size * 2
        else:
            self.final_emb_dim = config.emb_size

        # Stop prediction head
        self.stop_mlp1 = nn.Linear(config.emb_size, config.emb_size, bias=False)
        self.stop_mlp2 = nn.Linear(config.emb_size, 2)

        # First node selection head
        self.first_node_mlp1 = nn.Linear(self.final_emb_dim, config.emb_size)
        self.first_node_mlp2 = nn.Linear(config.emb_size, 1)

        # Second node selection head (conditioned on first node)
        self.second_node_mlp1 = nn.Linear(self.final_emb_dim * 2, config.emb_size)
        self.second_node_mlp2 = nn.Linear(config.emb_size, 1)

        # Edge (bond type) prediction head
        self.edge_mlp1 = nn.Linear(config.emb_size * 2, config.emb_size)
        self.edge_mlp2 = nn.Linear(config.emb_size, self.edge_dim)

        # Value function head
        self.value_mlp1 = nn.Linear(config.emb_size, config.emb_size, bias=False)
        self.value_mlp2 = nn.Linear(config.emb_size, 1)

        # Batch normalization layers if enabled
        if config.bn:
            self.node_emb_bn = nn.BatchNorm1d(8)
            self.stop_bn = nn.BatchNorm1d(config.emb_size)
            self.value_bn = nn.BatchNorm1d(config.emb_size)
        else:
            self.node_emb_bn = None
            self.stop_bn = None
            self.value_bn = None

    def forward(self, obs_adj, obs_node, deterministic=False, ac_real=None):
        """
        Forward pass through the policy network.

        Args:
            obs_adj: Adjacency matrices (batch_size, edge_dim, max_nodes, max_nodes)
            obs_node: Node features (batch_size, 1, max_nodes, node_feature_dim)
            deterministic: If True, return argmax actions instead of sampling
            ac_real: Ground truth actions for training (batch_size, 4), used for teacher forcing

        Returns:
            actions: Sampled or deterministic actions (batch_size, 4)
            log_probs: Log probabilities of actions (batch_size,)
            value: Value function estimates (batch_size, 1)
            entropy: Entropy of action distributions
            logits_dict: Dictionary of logits for all action components
            log_prob_dict: Dictionary of individual log probs (for expert training)
        """
        batch_size = obs_adj.shape[0]
        max_nodes = obs_adj.shape[2]

        # 1. Node embedding
        # obs_node: (batch_size, 1, max_nodes, node_feature_dim)
        obs_node_squeezed = obs_node.squeeze(1)  # (batch_size, max_nodes, node_feature_dim)
        node_emb_input = self.node_embedding(obs_node_squeezed)  # (batch_size, max_nodes, 8)

        if self.config.bn and self.node_emb_bn is not None:
            # Apply batch norm
            node_emb_input = node_emb_input.reshape(-1, 8)
            node_emb_input = self.node_emb_bn(node_emb_input)
            node_emb_input = node_emb_input.reshape(batch_size, max_nodes, 8)

        # Add back the middle dimension for GCN
        node_emb_input = node_emb_input.unsqueeze(1)  # (batch_size, 1, max_nodes, 8)

        # 2. GCN forward pass
        # Output: (batch_size, max_nodes, emb_size)
        emb_node = self.gcn_stack(obs_adj, node_emb_input)

        # 3. Calculate node masks (which nodes are valid)
        # Valid nodes have non-zero features (real nodes + NEW_NODE have is_new=1.0)
        node_mask = (obs_node.sum(dim=-1) > 0).squeeze(1)  # (batch_size, max_nodes)
        ob_len = node_mask.sum(dim=1)  # (batch_size,) number of valid nodes

        # Exclude NEW_NODE from first_node selection (prevent selecting placeholder as source)
        # NEW_NODE is at index max_nodes-1, so ob_len_first masks it out
        ob_len_first = ob_len - self.atom_type_num  # atom_type_num=1 for NEW_NODE

        # Create masks for first node selection (exclude last atom_type_num nodes)
        first_node_mask = torch.arange(max_nodes, device=obs_adj.device).unsqueeze(0) < ob_len_first.unsqueeze(1)

        # Mask null embeddings if enabled
        if self.config.mask_null:
            emb_node = emb_node * node_mask.unsqueeze(-1).float()

        # 4. Compute graph embedding (sum of all node embeddings)
        emb_graph = emb_node.sum(dim=1, keepdim=True)  # (batch_size, 1, emb_size)

        # Optionally concatenate graph embedding to each node
        if self.config.graph_emb:
            emb_graph_expanded = emb_graph.expand(-1, max_nodes, -1)  # (batch_size, max_nodes, emb_size)
            emb_node_with_graph = torch.cat([emb_node, emb_graph_expanded], dim=-1)  # (batch_size, max_nodes, emb_size*2)
        else:
            emb_node_with_graph = emb_node

        # 5. Stop logits
        stop_emb = F.relu(self.stop_mlp1(emb_node))  # (batch_size, max_nodes, emb_size)
        if self.config.bn and self.stop_bn is not None:
            stop_emb = stop_emb.reshape(-1, self.config.emb_size)
            stop_emb = self.stop_bn(stop_emb)
            stop_emb = stop_emb.reshape(batch_size, max_nodes, self.config.emb_size)

        stop_emb = stop_emb.sum(dim=1)  # (batch_size, emb_size)
        logits_stop = self.stop_mlp2(stop_emb)  # (batch_size, 2)

        # Apply stop shift (now 0.0 to remove bias - config updated)
        stop_shift = torch.tensor([0.0, 0.0], device=obs_adj.device)
        logits_stop = logits_stop + stop_shift

        # 6. First node logits
        logits_first = F.relu(self.first_node_mlp1(emb_node_with_graph))  # (batch_size, max_nodes, emb_size)
        logits_first = self.first_node_mlp2(logits_first).squeeze(-1)  # (batch_size, max_nodes)

        # Mask invalid nodes
        logits_first = logits_first.masked_fill(~first_node_mask, -1e10)

        # 7. Sample first node (or use ground truth for training)
        if ac_real is not None:
            ac_first = ac_real[:, 0].long()
        else:
            dist_first = Categorical(logits=logits_first)
            ac_first = dist_first.sample() if not deterministic else logits_first.argmax(dim=1)

        # Get embedding of first node
        emb_first = emb_node[torch.arange(batch_size), ac_first]  # (batch_size, emb_size)
        emb_first = emb_first.unsqueeze(1)  # (batch_size, 1, emb_size)

        # 8. Second node logits (conditioned on first node)
        # Concatenate first node embedding with all node embeddings
        emb_first_expanded = emb_first.expand(-1, max_nodes, -1)  # (batch_size, max_nodes, emb_size)

        if self.config.graph_emb:
            emb_cat = torch.cat([emb_first_expanded, emb_node_with_graph], dim=-1)
        else:
            emb_cat = torch.cat([emb_first_expanded, emb_node], dim=-1)

        logits_second = F.relu(self.second_node_mlp1(emb_cat))  # (batch_size, max_nodes, emb_size)
        logits_second = self.second_node_mlp2(logits_second).squeeze(-1)  # (batch_size, max_nodes)

        # Mask: cannot select first node, and must be valid
        second_node_mask = node_mask.clone()
        second_node_mask[torch.arange(batch_size), ac_first] = False
        logits_second = logits_second.masked_fill(~second_node_mask, -1e10)

        # 9. Sample second node (or use ground truth for training)
        if ac_real is not None:
            ac_second = ac_real[:, 1].long()
        else:
            dist_second = Categorical(logits=logits_second)
            ac_second = dist_second.sample() if not deterministic else logits_second.argmax(dim=1)

        # Get embedding of second node
        emb_second = emb_node[torch.arange(batch_size), ac_second]  # (batch_size, emb_size)
        emb_second = emb_second.unsqueeze(1)  # (batch_size, 1, emb_size)

        # 10. Edge type logits
        emb_edge_cat = torch.cat([emb_first, emb_second], dim=-1).squeeze(1)  # (batch_size, emb_size*2)
        logits_edge = F.relu(self.edge_mlp1(emb_edge_cat))  # (batch_size, emb_size)
        logits_edge = self.edge_mlp2(logits_edge)  # (batch_size, edge_dim)

        # 11. Sample edge type (or use ground truth for training)
        if ac_real is not None:
            ac_edge = ac_real[:, 2].long()
        else:
            dist_edge = Categorical(logits=logits_edge)
            ac_edge = dist_edge.sample() if not deterministic else logits_edge.argmax(dim=1)

        # 12. Sample stop action
        if ac_real is not None:
            ac_stop = ac_real[:, 3].long()
        else:
            dist_stop = Categorical(logits=logits_stop)
            ac_stop = dist_stop.sample() if not deterministic else logits_stop.argmax(dim=1)

        # 13. Compute value function
        value_emb = F.relu(self.value_mlp1(emb_node))  # (batch_size, max_nodes, emb_size)
        if self.config.bn and self.value_bn is not None:
            value_emb = value_emb.reshape(-1, self.config.emb_size)
            value_emb = self.value_bn(value_emb)
            value_emb = value_emb.reshape(batch_size, max_nodes, self.config.emb_size)

        value_emb = value_emb.max(dim=1)[0]  # (batch_size, emb_size)
        value = self.value_mlp2(value_emb)  # (batch_size, 1)

        # 14. Assemble actions
        actions = torch.stack([ac_first, ac_second, ac_edge, ac_stop], dim=1)  # (batch_size, 4)

        # 15. Compute log probabilities (needed for PPO)
        # For training, we need to compute log probs using the REAL second node logits
        # (conditioned on ground truth first node, not sampled first node)
        if ac_real is not None:
            # Recompute second node logits conditioned on ground truth first node
            ac_first_real = ac_real[:, 0].long()
            emb_first_real = emb_node[torch.arange(batch_size), ac_first_real].unsqueeze(1)
            emb_first_real_expanded = emb_first_real.expand(-1, max_nodes, -1)

            if self.config.graph_emb:
                emb_cat_real = torch.cat([emb_first_real_expanded, emb_node_with_graph], dim=-1)
            else:
                emb_cat_real = torch.cat([emb_first_real_expanded, emb_node], dim=-1)

            logits_second_real = F.relu(self.second_node_mlp1(emb_cat_real))
            logits_second_real = self.second_node_mlp2(logits_second_real).squeeze(-1)

            # Mask for real second node
            second_node_mask_real = node_mask.clone()
            second_node_mask_real[torch.arange(batch_size), ac_first_real] = False
            logits_second_real = logits_second_real.masked_fill(~second_node_mask_real, -1e10)

            # Recompute edge logits with real nodes
            ac_second_real = ac_real[:, 1].long()
            emb_second_real = emb_node[torch.arange(batch_size), ac_second_real].unsqueeze(1)
            emb_edge_cat_real = torch.cat([emb_first_real, emb_second_real], dim=-1).squeeze(1)
            logits_edge_real = F.relu(self.edge_mlp1(emb_edge_cat_real))
            logits_edge_real = self.edge_mlp2(logits_edge_real)

        else:
            logits_second_real = logits_second
            logits_edge_real = logits_edge

        # Compute log probabilities
        dist_first = Categorical(logits=logits_first)
        dist_second_real = Categorical(logits=logits_second_real)
        dist_edge_real = Categorical(logits=logits_edge_real)
        dist_stop = Categorical(logits=logits_stop)

        log_prob_first = dist_first.log_prob(actions[:, 0])
        log_prob_second = dist_second_real.log_prob(actions[:, 1])
        log_prob_edge = dist_edge_real.log_prob(actions[:, 2])
        log_prob_stop = dist_stop.log_prob(actions[:, 3])

        log_probs = log_prob_first + log_prob_second + log_prob_edge + log_prob_stop

        # Entropy for each action component
        entropy_first = dist_first.entropy()
        entropy_second = dist_second_real.entropy()
        entropy_edge = dist_edge_real.entropy()
        entropy_stop = dist_stop.entropy()
        entropy = entropy_first + entropy_second + entropy_edge + entropy_stop

        # Store logits for debugging/logging
        logits_dict = {
            'logits_first': logits_first,
            'logits_second': logits_second_real,
            'logits_edge': logits_edge_real,
            'logits_stop': logits_stop
        }

        # Store individual log_probs for masking in expert training
        log_prob_dict = {
            'first': log_prob_first,
            'second': log_prob_second,
            'edge': log_prob_edge,
            'stop': log_prob_stop,
        }

        return actions, log_probs, value, entropy, logits_dict, log_prob_dict

    def evaluate_actions(self, obs_adj, obs_node, actions):
        """
        Evaluate actions (compute log probs and values for given actions).
        Used during PPO updates.

        Args:
            obs_adj: Adjacency matrices
            obs_node: Node features
            actions: Actions to evaluate (batch_size, 4)

        Returns:
            log_probs: Log probabilities of actions
            value: Value estimates
            entropy: Entropy of action distributions
        """
        _, log_probs, value, entropy, _, _ = self.forward(obs_adj, obs_node, ac_real=actions)
        return log_probs, value, entropy