import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from torch_geometric.datasets import TUDataset
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader
from utils.utils_method import get_graph_diameter
from model.MRGNN import MRGNN

def Create_Reservoir_Dataset(native_dataset_path, native_dataset_name, n_units, n_classes,max_k, run, adjacency_matrix,  use_node_attr=False):

    reservoir_augmented_dataset_root = '~/Dataset/Reservoir_TANH_Dataset/'
    if not os.path.exists(reservoir_augmented_dataset_root):
        os.makedirs(reservoir_augmented_dataset_root)

    reservoir_augmented_dataset_name = 'RES_'+str(adjacency_matrix)+"_"+str(max_k)+"_n_units_"+str(n_units)+'_'+native_dataset_name

    device = 'cpu'

    dataset = TUDataset(root=native_dataset_path, name=native_dataset_name, pre_transform=get_graph_diameter,
                        use_node_attr=use_node_attr)

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=4)
    model = MRGNN(loader.dataset.num_features, n_units, n_classes, 0, max_k=max_k, device=device).to(device)


    for r in run:

        current_reservoir_augmented_dataset_name = "run_" + str(r) + "_TANH_" + reservoir_augmented_dataset_name

        if native_dataset_name == 'PROTEINS':
            if adjacency_matrix == 'A':
                TUDataset(
                    root=os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name),
                    name=native_dataset_name,
                    pre_transform=model.get_TANH_resevoir_A_PROTEINS,
                    use_node_attr=use_node_attr)
            elif adjacency_matrix == 'L':
                TUDataset(
                    root=os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name),
                    name=native_dataset_name,
                    pre_transform=model.get_TANH_resevoir_L_PROTEINS,
                    use_node_attr=use_node_attr)
            elif adjacency_matrix == 'D':
                TUDataset(
                    root=os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name),
                    name=native_dataset_name,
                    pre_transform=model.get_TANH_resevoir_D_PROTEINS,
                    use_node_attr=use_node_attr)

        else:
            if adjacency_matrix == 'A':
                TUDataset(
                    root=os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name),
                    name=native_dataset_name,
                    pre_transform=model.get_TANH_resevoir_A,
                    use_node_attr=use_node_attr)

            elif adjacency_matrix == 'L':
                TUDataset(
                    root=os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name),
                    name=native_dataset_name,
                    pre_transform=model.get_TANH_resevoir_L,
                    use_node_attr=use_node_attr)
            elif adjacency_matrix == 'D':
                TUDataset(
                    root=os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name),
                    name=native_dataset_name,
                    pre_transform=model.get_TANH_resevoir_D,
                    use_node_attr=use_node_attr)
