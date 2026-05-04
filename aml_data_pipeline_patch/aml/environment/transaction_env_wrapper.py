"""
Wrapper to integrate TransactionGraphEnv with existing GCPN training code.
"""

from .transaction_env import TransactionGraphEnv


class TransactionEnvWrapper:
    """
    Wrapper around TransactionGraphEnv for compatibility with GCPN training.
    """

    def __init__(self, config):
        """Initialize transaction environment wrapper."""
        self.config = config
        self.env = TransactionGraphEnv(config)

        # Store attributes for compatibility
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # atom_type_num = 1: NEW_NODE token reserved at max_nodes-1
        # Policy uses this to exclude NEW_NODE from first_node selection
        self.atom_type_num = 1

    def reset(self):
        """Reset environment."""
        return self.env.reset()

    def step(self, action):
        """
        Take step in environment.

        Args:
            action: Action array, shape (4,) or (1, 4) or (batch, 4)
        """
        # Handle batch dimension if present
        import numpy as np
        action = np.asarray(action)
        if action.ndim == 2:
            action = action.squeeze(0)  # Remove batch dimension
        return self.env.step(action)

    def seed(self, seed):
        """Set random seed."""
        self.env.seed(seed)

    def get_expert(self, batch_size, is_final=False, curriculum=0, level_total=6, level=0):
        """Get expert demonstrations."""
        return self.env.get_expert(
            batch_size=batch_size,
            is_final=is_final,
            curriculum=curriculum,
            level_total=level_total,
            level=level
        )

    def close(self):
        """Close environment."""
        pass
