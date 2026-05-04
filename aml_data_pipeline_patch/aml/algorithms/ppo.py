"""
Proximal Policy Optimization (PPO) algorithm for PyTorch GCPN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from .storage import RolloutBuffer


class PPO:
    """
    PPO algorithm implementation for GCPN.

    Args:
        policy: Policy network
        config: Configuration object
        device: Device to run on
    """

    def __init__(self, policy, config, device='cpu'):
        self.policy = policy
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=config.lr)

        # Statistics tracking
        self.total_steps = 0
        self.total_episodes = 0

    def collect_rollouts(self, env, n_steps):
        """
        Collect rollouts from the environment.

        Args:
            env: Environment instance
            n_steps: Number of steps to collect

        Returns:
            buffer: RolloutBuffer with collected data
            ep_info: List of episode information dictionaries
        """
        # Create buffer
        obs_shape_adj = env.observation_space['adj'].shape
        obs_shape_node = env.observation_space['node'].shape
        buffer = RolloutBuffer(n_steps, obs_shape_adj, obs_shape_node, device=self.device)

        # Episode statistics
        ep_info = []
        ep_rewards = deque(maxlen=100)
        ep_lengths = deque(maxlen=100)
        ep_lengths_valid = deque(maxlen=100)

        # Current episode tracking
        current_ep_reward = 0
        current_ep_reward_env = 0
        current_ep_length = 0
        current_ep_length_valid = 0

        # Reset environment
        obs = env.reset()
        obs_adj = obs['adj'][None, ...]  # Add batch dimension
        obs_node = obs['node'][None, ...]

        self.policy.eval()
        with torch.no_grad():
            for step in range(n_steps):
                # Convert observations to tensors
                obs_adj_tensor = torch.from_numpy(obs_adj).float().to(self.device)
                obs_node_tensor = torch.from_numpy(obs_node).float().to(self.device)

                # Get action from policy
                actions, log_probs, values, _, _, _ = self.policy(obs_adj_tensor, obs_node_tensor)

                # Convert to numpy for environment
                action = actions[0].cpu().numpy()
                value = values[0].item()
                log_prob = log_probs[0].item()

                # Take step in environment
                obs_next, reward, done, info = env.step(action[None, ...])

                # Store in buffer
                buffer.add(
                    obs_adj[0],  # Remove batch dimension
                    obs_node[0],
                    action,
                    reward,
                    value,
                    log_prob,
                    done
                )

                # Update episode statistics
                current_ep_reward += reward
                current_ep_length += 1

                # Track valid actions (actions with positive reward)
                if reward > 0:
                    current_ep_length_valid += 1
                    current_ep_reward_env += reward

                # Update observation
                obs_adj = obs_next['adj'][None, ...]
                obs_node = obs_next['node'][None, ...]

                self.total_steps += 1

                # Handle episode end
                if done:
                    # Record episode info
                    ep_info.append({
                        'reward': current_ep_reward,
                        'reward_env': current_ep_reward_env,
                        'length': current_ep_length,
                        'length_valid': current_ep_length_valid,
                        'info': info
                    })

                    ep_rewards.append(current_ep_reward)
                    ep_lengths.append(current_ep_length)
                    ep_lengths_valid.append(current_ep_length_valid)

                    # Reset episode tracking
                    current_ep_reward = 0
                    current_ep_reward_env = 0
                    current_ep_length = 0
                    current_ep_length_valid = 0
                    self.total_episodes += 1

                    # Reset environment
                    obs = env.reset()
                    obs_adj = obs['adj'][None, ...]
                    obs_node = obs['node'][None, ...]

            # Bootstrap value for last state (if not done)
            if not done:
                obs_adj_tensor = torch.from_numpy(obs_adj).float().to(self.device)
                obs_node_tensor = torch.from_numpy(obs_node).float().to(self.device)
                _, _, last_value, _, _, _ = self.policy(obs_adj_tensor, obs_node_tensor)
                last_value = last_value[0].item()
            else:
                last_value = 0.0

        # Compute advantages
        buffer.compute_advantages(last_value, gamma=self.config.gamma, lam=self.config.lam)

        return buffer, ep_info

    def update(self, buffer):
        """
        Update policy using PPO.

        Args:
            buffer: RolloutBuffer with collected data

        Returns:
            Dictionary with training statistics
        """
        self.policy.train()

        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
        }

        # Multiple epochs over the data
        for epoch in range(self.config.optim_epochs):
            # Iterate over batches
            for batch in buffer.get_batch_iterator(self.config.optim_batchsize):
                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch['obs_adj'],
                    batch['obs_node'],
                    batch['actions']
                )

                # Flatten values
                values = values.squeeze(-1)

                # Compute policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - batch['old_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.functional.mse_loss(values, batch['returns'])

                # Compute entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss + self.config.entcoeff * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                # Logging
                with torch.no_grad():
                    approx_kl = (batch['old_log_probs'] - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_param).float().mean().item()

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['approx_kl'].append(approx_kl)
                stats['clip_fraction'].append(clip_fraction)

        # Average statistics
        for key in stats:
            stats[key] = np.mean(stats[key])

        return stats

    def train_expert(self, env, batch_size, curriculum=0, level=0, level_total=6):
        """
        Expert imitation

        Args:
            env: Environment instance
            batch_size: Batch size for expert data
            curriculum: Whether to use curriculum learning
            level: Current curriculum level
            level_total: Total curriculum levels

        Returns:
            Dictionary with training statistics
        """
        self.policy.train()

        # Get expert data from environment
        expert_obs, expert_actions = env.get_expert(
            batch_size,
            curriculum=curriculum,
            level=level,
            level_total=level_total
        )

        # Convert to tensors
        obs_adj = torch.from_numpy(expert_obs['adj']).float().to(self.device)
        obs_node = torch.from_numpy(expert_obs['node']).float().to(self.device)
        actions = torch.from_numpy(expert_actions).long().to(self.device)

        # Forward pass
        _, log_probs, _, _, _, log_prob_dict = self.policy(obs_adj, obs_node, ac_real=actions)

        # Mask losses for stop actions (only train stop head)
        # When stop=1, don't train node1/node2/edge_type (they're dummy values)
        stop_mask = (actions[:, 3] == 1)  # Stop actions
        node_mask = (actions[:, 3] == 0)  # Edge addition actions

        # Track statistics for validation logging
        stats = {
            'expert_loss': 0.0,
            'num_stop': stop_mask.sum().item(),
            'num_node': node_mask.sum().item(),
            'loss_stop': 0.0,
            'loss_nodes': 0.0,
        }

        if stop_mask.sum() > 0 and node_mask.sum() > 0:
            # Mixed batch: stop and non-stop examples
            loss_stop = -log_prob_dict['stop'][stop_mask].mean()
            loss_nodes = -(
                log_prob_dict['first'][node_mask] +
                log_prob_dict['second'][node_mask] +
                log_prob_dict['edge'][node_mask]
            ).mean()
            loss_stop_head = -log_prob_dict['stop'][node_mask].mean()
            loss = loss_stop + loss_nodes + loss_stop_head
            stats['loss_stop'] = loss_stop.item()
            stats['loss_nodes'] = loss_nodes.item()
        elif stop_mask.sum() > 0:
            # All stop examples
            loss = -log_prob_dict['stop'].mean()
            stats['loss_stop'] = loss.item()
        else:
            # All non-stop examples (standard case)
            loss = -log_probs.mean()
            stats['loss_nodes'] = loss.item()

        stats['expert_loss'] = loss.item()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return stats

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
        }, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
