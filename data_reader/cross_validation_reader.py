
import torch
from random import seed
import numpy as np
import random
from torch_geometric.datasets import TUDataset
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from utils.utils_method import get_graph_diameter


rnd_state = np.random.RandomState(seed(1))


def split_ids(ids, folds=10):
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]

    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    valid_ids = []
    train_ids = []

    for fold in range(folds):
        valid_fold = []
        while len(valid_fold) < stride:
            id = random.choice(ids)
            if id not in test_ids[fold] and id not in valid_fold:
               valid_fold.append(id)

        valid_ids.append(np.asarray(valid_fold))
        train_ids.append(np.array([e for e in ids if e not in test_ids[fold] and e not in valid_ids[fold]]))
        assert len(train_ids[fold]) + len(test_ids[fold]) + len(valid_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]) + list(valid_ids[fold]))) == n, 'invalid splits'


    return train_ids, test_ids, valid_ids


def getcross_validation_split(dataset_path='~/Dataset/MUTAG', dataset_name='MUTAG', n_folds=2, batch_size=1, use_node_attr=False):

    dataset = TUDataset(root=dataset_path, name=dataset_name, pre_transform=get_graph_diameter, use_node_attr=use_node_attr)
    train_ids, test_ids, valid_ids = split_ids(rnd_state.permutation(len(dataset)), folds=n_folds)
    splits=[]

    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:

            gdata = dataset[torch.from_numpy(split[fold_id])]

            loader = DataLoader(gdata,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)

            loaders.append(loader)
        splits.append(loaders)


    return splits #0-train, 1-test, 2-valid

def getcross_validation_split_4_reservoir(dataset_path, dataset_name,  n_folds=2, batch_size=1, use_node_attr=False):
    dataset = TUDataset(root=dataset_path, name=dataset_name,  use_node_attr=use_node_attr)
    train_ids, test_ids, valid_ids = split_ids(rnd_state.permutation(len(dataset)), folds=n_folds)
    splits = []
    for fold_id in range(n_folds):
        loaders = []
        for split in [train_ids, test_ids, valid_ids]:

            gdata = dataset[torch.from_numpy(split[fold_id])]

            loader = DataLoader(gdata,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
            loaders.append(loader)
        splits.append(loaders)

    return splits #0-train, 1-test, 2-valid
