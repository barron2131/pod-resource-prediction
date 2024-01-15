import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GATGRU(nn.Module):
    def __init__(self,
                 in_features: int, hidden_channels: int, num_nodes: int, window_len: int, dropout_p: float = 0.2):
        super().__init__()
        self.gcn1 = GATConv(in_features, hidden_channels)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(dropout_p)

        self.gcn2 = GATConv(in_features, hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(dropout_p)

        # LSTM block
        self.gru1 = nn.GRU(hidden_channels, hidden_channels)
        self.gru2 = nn.GRU(hidden_channels, hidden_channels)

        # Forward variables
        self.num_nodes = num_nodes
        self.window_len = window_len
        self.in_features = in_features
        self.hidden_channels = hidden_channels

        # Linear block
        self.linear1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.dropout3 = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(hidden_channels, 2)

    def forward(self, X: torch.FloatTensor, Ax: torch.FloatTensor) -> torch.FloatTensor:
        batch_feat = X.view(-1, self.in_features)
        batch_adj = Ax.view(-1, self.num_nodes, self.num_nodes)
        batch_adj = torch.block_diag(*batch_adj).to_sparse_coo()

        edge_index = batch_adj.indices()
        edge_weights = batch_adj.values()

        x = self.gcn1(batch_feat, edge_index, edge_weights)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.gcn2(batch_feat, edge_index, edge_weights)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = x.view(self.window_len, -1, self.hidden_channels)
        x, h0 = self.gru1(x)
        x, h1 = self.gru2(x)

        x = torch.cat([h0[0, :, :], h1[0, :, :]], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = x.view(-1, self.num_nodes, 2)

        return F.relu(x)
