# gcpn.py - Graph Convolutional Network layers in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    GCN layer for batched graph data with multiple edge types.

    Given adjacency matrices for different bond types and node features,
    performs message passing and aggregation.

    Args:
        in_channels: Input node feature dimension
        out_channels: Output node embedding dimension
        edge_dim: Number of edge types (bond types)
        aggregate: Aggregation method ('mean', 'sum', or 'concat')
        activation: Whether to apply ReLU activation
        normalize: Whether to apply L2 normalization
    """

    def __init__(self, in_channels, out_channels, edge_dim, aggregate='mean',
                 activation=True, normalize=False):
        super(GCNLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.aggregate = aggregate
        self.activation = activation
        self.normalize = normalize

        # Weights for each edge type: shape (1, edge_dim, in_channels, out_channels)
        self.W = nn.Parameter(torch.empty(1, edge_dim, in_channels, out_channels))
        nn.init.xavier_uniform_(self.W)

    def forward(self, adj, node_features):
        """
        Forward pass of GCN layer.

        Args:
            adj: Adjacency matrix (batch_size, edge_dim, num_nodes, num_nodes)
            node_features: Node features (batch_size, 1, num_nodes, in_channels)

        Returns:
            node_embeddings: Updated node features. Shape depends on aggregation:
                - 'mean'/'sum': (batch_size, 1, num_nodes, out_channels)
                - 'concat':    (batch_size, 1, num_nodes, edge_dim * out_channels)
        """
        batch_size = adj.shape[0]

        # Expand node features for each edge type
        # (batch_size, edge_dim, num_nodes, in_channels)
        node_features_expanded = node_features.repeat(1, self.edge_dim, 1, 1)

        # Expand weights for batch
        # (batch_size, edge_dim, in_channels, out_channels)
        W_expanded = self.W.repeat(batch_size, 1, 1, 1)

        # Message passing: adj @ node_features
        # (batch_size, edge_dim, num_nodes, in_channels)
        messages = torch.matmul(adj, node_features_expanded)

        # Apply weights: messages @ W
        # (batch_size, edge_dim, num_nodes, out_channels)
        node_embeddings = torch.matmul(messages, W_expanded)

        # Apply activation
        if self.activation:
            node_embeddings = F.relu(node_embeddings)

        # Aggregate over edge types
        if self.aggregate == 'mean':
            node_embeddings = torch.mean(node_embeddings, dim=1, keepdim=True)
        elif self.aggregate == 'sum':
            node_embeddings = torch.sum(node_embeddings, dim=1, keepdim=True)
        elif self.aggregate == 'concat':
            # Flatten edge dimension into feature dimension
            batch_size, edge_dim, num_nodes, out_channels = node_embeddings.shape
            node_embeddings = node_embeddings.permute(0, 2, 1, 3).contiguous()
            node_embeddings = node_embeddings.view(batch_size, 1, num_nodes, -1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregate}")

        # Apply L2 normalization
        if self.normalize:
            node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)

        return node_embeddings


class GCNStack(nn.Module):
    """
    Stack of GCN layers with optional residual connections and concatenation.

    Args:
        num_layers: Number of GCN layers
        in_channels: Input node feature dimension
        hidden_channels: Hidden dimension for GCN layers
        edge_dim: Number of edge types
        aggregate: Aggregation method ('mean', 'sum', or 'concat')
        has_residual: Whether to use residual connections (skip connections)
        has_concat: Whether to concatenate with input at each layer
        bn: Whether to use batch normalization
    """

    def __init__(self, num_layers, in_channels, hidden_channels, edge_dim,
                 aggregate='mean', has_residual=False, has_concat=False, bn=False):
        super(GCNStack, self).__init__()

        self.num_layers = num_layers
        self.has_residual = has_residual
        self.has_concat = has_concat
        self.bn = bn

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if bn else None

        # First layer: embedding layer
        current_in_channels = in_channels
        self.layers.append(GCNLayer(current_in_channels, hidden_channels, edge_dim,
                                    aggregate=aggregate, activation=True, normalize=False))
        if bn:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Determine dimensions for subsequent layers
        if has_concat:
            current_in_channels = hidden_channels + in_channels
        else:
            current_in_channels = hidden_channels

        # Middle layers
        for i in range(num_layers - 2):
            self.layers.append(GCNLayer(current_in_channels, hidden_channels, edge_dim,
                                        aggregate=aggregate, activation=True, normalize=False))
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

            if has_concat:
                current_in_channels = hidden_channels + current_in_channels
            else:
                current_in_channels = hidden_channels

        # Final layer (no activation, normalization depends on bn)
        self.layers.append(GCNLayer(current_in_channels, hidden_channels, edge_dim,
                                    aggregate=aggregate, activation=False, normalize=(not bn)))
        if bn:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, adj, node_features):
        """
        Forward pass through GCN stack.

        Args:
            adj: Adjacency matrix (batch_size, edge_dim, num_nodes, num_nodes)
            node_features: Node features (batch_size, 1, num_nodes, in_channels)

        Returns:
            node_embeddings: Final node embeddings (batch_size, num_nodes, hidden_channels)
        """
        x = node_features
        x_input = node_features  # Save for concatenation

        for i, layer in enumerate(self.layers):
            x_prev = x
            x = layer(adj, x)

            # Apply batch normalization if enabled
            if self.bn and self.batch_norms is not None:
                # Reshape for batch norm: (batch_size, 1, num_nodes, channels) -> (batch_size * num_nodes, channels)
                batch_size_, _, num_nodes, channels = x.shape
                x = x.squeeze(1).reshape(-1, channels)          # (batch_size * num_nodes, channels)
                x = self.batch_norms[i](x)
                x = x.reshape(batch_size_, num_nodes, channels).unsqueeze(1)  # Back to (batch_size, 1, num_nodes, channels)

            # Apply residual connection or concatenation (skip for first and last layer)
            if i < len(self.layers) - 1:
                if self.has_residual and i > 0:
                    x = x + x_prev
                elif self.has_concat:
                    if i == 0:
                        x = torch.cat([x, x_input], dim=-1)
                    else:
                        x = torch.cat([x, x_prev], dim=-1)

        # Squeeze the middle dimension: (batch_size, 1, num_nodes, channels) -> (batch_size, num_nodes, channels)
        x = x.squeeze(1)
        return x


# Example usage (optional)
if __name__ == "__main__":
    # Dummy data: batch_size=2, edge_dim=3, num_nodes=5, in_channels=4, hidden_channels=8
    batch_size, edge_dim, num_nodes, in_channels = 2, 3, 5, 4
    hidden_channels = 8

    adj = torch.randn(batch_size, edge_dim, num_nodes, num_nodes)
    node_features = torch.randn(batch_size, 1, num_nodes, in_channels)

    model = GCNStack(num_layers=3, in_channels=in_channels, hidden_channels=hidden_channels,
                     edge_dim=edge_dim, aggregate='mean', has_residual=True, bn=True)
    out = model(adj, node_features)
    print(f"Output shape: {out.shape}")  # Expected: (2, 5, 8)