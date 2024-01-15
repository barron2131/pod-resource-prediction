import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PodDataset(Dataset):
    def __init__(self, datasource: str, split: str = "train"):
        node_source = os.path.join(datasource, f"{split}_node_features.npz")
        edge_source = os.path.join(datasource, f"{split}_edge_features.npz")
        self.node_features = np.load(node_source)
        self.edge_features = np.load(edge_source)

    def __len__(self):
        return len(self.node_features["X"])

    def __getitem__(self, item):
        nodes_train = self.node_features["X"][item]
        edges_train = self.edge_features["X"][item]
        nodes_test = self.node_features["y"][item]
        edges_test = self.edge_features["y"][item]

        return (
            (torch.Tensor(nodes_train), torch.Tensor(edges_train)),
            (torch.Tensor(nodes_test), torch.Tensor(edges_test))
        )
