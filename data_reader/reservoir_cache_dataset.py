import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data, DataLoader


def load_cache_metadata(cache_dir):
    metadata_path = os.path.join(os.path.expanduser(cache_dir), "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


def _build_splits(ids, folds, seed):
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]

    if len(test_ids) != folds:
        raise ValueError(
            "Invalid number of folds for dataset size: expected {}, got {}".format(
                folds, len(test_ids)
            )
        )

    rng = np.random.RandomState(seed)
    valid_ids = []
    train_ids = []

    for fold in range(folds):
        test_set = set(test_ids[fold].tolist())
        remaining = np.array([idx for idx in ids if idx not in test_set], dtype=np.int64)

        valid_size = min(stride, len(remaining))
        valid_fold = rng.choice(remaining, size=valid_size, replace=False)
        valid_set = set(valid_fold.tolist())
        train_fold = np.array([idx for idx in remaining if idx not in valid_set], dtype=np.int64)

        valid_ids.append(valid_fold)
        train_ids.append(train_fold)

    return train_ids, test_ids, valid_ids


class ReservoirCacheDataset(Dataset):
    def __init__(self, cache_dir):
        super(ReservoirCacheDataset, self).__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.graph_dir = os.path.join(self.cache_dir, "graphs")
        if not os.path.exists(self.graph_dir):
            raise ValueError("Cache graph directory not found: {}".format(self.graph_dir))

        self.graph_files = sorted(
            [
                os.path.join(self.graph_dir, file_name)
                for file_name in os.listdir(self.graph_dir)
                if file_name.endswith(".npz") and file_name.startswith("graph_")
            ]
        )

        if not self.graph_files:
            raise ValueError("No cached graph files found in {}".format(self.graph_dir))

        with np.load(self.graph_files[0], allow_pickle=False) as first_graph:
            self.num_features = int(first_graph["reservoir"].shape[1])

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        file_path = self.graph_files[idx]
        with np.load(file_path, allow_pickle=False) as graph_npz:
            reservoir = torch.from_numpy(graph_npz["reservoir"]).float()
            y = torch.from_numpy(graph_npz["y"]).long().view(-1)

            data = Data(reservoir=reservoir, y=y)

            if "num_nodes" in graph_npz:
                data.num_nodes = int(graph_npz["num_nodes"][0])
            else:
                data.num_nodes = reservoir.shape[0]

            if "graph_id" in graph_npz:
                data.graph_id = torch.tensor(
                    [int(graph_npz["graph_id"][0])], dtype=torch.long
                )

            return data


def getcross_validation_split_from_cache(
    cache_dir,
    n_folds=10,
    batch_size=32,
    seed=1,
    shuffle=True,
    num_workers=0,
):
    dataset = ReservoirCacheDataset(cache_dir)
    ids = np.random.RandomState(seed).permutation(len(dataset))

    train_ids, test_ids, valid_ids = _build_splits(ids, folds=n_folds, seed=seed)
    splits = []

    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:
            split_dataset = Subset(dataset, split[fold_id].tolist())
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            loaders.append(loader)
        splits.append(loaders)

    return splits
