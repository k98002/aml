"""
Rollout buffer for storing trajectories and computing advantages
"""
import torch
import numpy as np


class RolloutBuffer:
    """
    Buffer for storing rollouts and computing advantages using GAE.

    Args:
        buffer_size: Maximum number of steps to store
        obs_shape_adj: Shape of adjacency observation
        obs_shape_node: Shape of node observation
        device: Device to store tensors on
    """

    def __init__(self, buffer_size, obs_shape_adj, obs_shape_node, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.reset()

        # Pre-allocate memory
        self.obs_adj = torch.zeros((buffer_size, *obs_shape_adj), dtype=torch.float32)
        self.obs_node = torch.zeros((buffer_size, *obs_shape_node), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, 4), dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)

    def reset(self):
        """Reset buffer"""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def add(self, obs_adj, obs_node, action, reward, value, log_prob, done):
        """
        Add a single timestep to the buffer.

        Args:
            obs_adj: Adjacency matrix observation
            obs_node: Node feature observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.obs_adj[self.ptr] = torch.as_tensor(obs_adj, dtype=torch.float32)
        self.obs_node[self.ptr] = torch.as_tensor(obs_node, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.bool)

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_value: Value estimate for the last state (for bootstrapping)
            gamma: Discount factor
            lam: GAE lambda parameter

        This implements TD(lambda) advantage estimation as described in:
        "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
        """
        last_gae_lam = 0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_value = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Compute returns
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get(self):
        """
        Get all data from buffer and reset.

        Returns:
            Dictionary with all buffer contents
        """
        assert self.ptr > 0, "Buffer is empty"

        # Normalize advantages
        adv_mean = self.advantages[:self.ptr].mean()
        adv_std = self.advantages[:self.ptr].std()
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - adv_mean) / (adv_std + 1e-8)

        data = {
            'obs_adj': self.obs_adj[:self.ptr].to(self.device),
            'obs_node': self.obs_node[:self.ptr].to(self.device),
            'actions': self.actions[:self.ptr].to(self.device),
            'rewards': self.rewards[:self.ptr].to(self.device),
            'values': self.values[:self.ptr].to(self.device),
            'log_probs': self.log_probs[:self.ptr].to(self.device),
            'dones': self.dones[:self.ptr].to(self.device),
            'advantages': self.advantages[:self.ptr].to(self.device),
            'returns': self.returns[:self.ptr].to(self.device),
        }

        self.reset()
        return data

    def get_batch_iterator(self, batch_size):
        """
        Create a generator for iterating over batches of data.

        Args:
            batch_size: Size of each batch

        Yields:
            Dictionary with batch data
        """
        indices = np.random.permutation(self.ptr)
        start_idx = 0

        while start_idx < self.ptr:
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield {
                'obs_adj': self.obs_adj[batch_indices].to(self.device),
                'obs_node': self.obs_node[batch_indices].to(self.device),
                'actions': self.actions[batch_indices].to(self.device),
                'advantages': self.advantages[batch_indices].to(self.device),
                'returns': self.returns[batch_indices].to(self.device),
                'old_log_probs': self.log_probs[batch_indices].to(self.device),
            }

            start_idx += batch_size
