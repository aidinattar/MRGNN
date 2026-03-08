import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import itertools
import numpy as np
from mpi4py import MPI

# Already in the same dir, hence import just the module
from ReservoirExtraction import Create_Reservoir_Dataset

def main():
    # 1. MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 2. Grid definition
    run = [0, 1, 2, 3, 4]
    adj_mat_list = ['D','A','L']
    max_k_list = [3, 4, 5, 6]

    datasets_config = [
        {
            'name': 'ENZYMES',
            'path': '~/Dataset/ENZYMES',
            'n_classes': 6,
            'use_node_attr': True,
            'n_units_list': [50, 100]
        },
        {
            'name': 'PROTEINS',
            'path': '~/Dataset/PROTEINS',
            'n_classes': 2,
            'use_node_attr': False,
            'n_units_list': [25, 50]
        }
    ]

    # 3. Aggregating all tasks
    all_tasks = []
    for ds in datasets_config:
        for M, k, n_units in itertools.product(adj_mat_list, max_k_list, ds['n_units_list']):
            all_tasks.append({
                'dataset_name': ds['name'],
                'dataset_path': ds['path'],
                'n_classes': ds['n_classes'],
                'use_node_attr': ds['use_node_attr'],
                'M': M,
                'k': k,
                'n_units': n_units
            })

    # 4. Task division among ranks
    my_tasks = np.array_split(all_tasks, size)[rank]
    
    print(f"[Rank {rank}] Assegnati {len(my_tasks)} task su {len(all_tasks)} totali.")

    # --- FIX RACE CONDITION ON THE RAW DATASET ---
    if rank == 0:
        print("[Rank 0] Verifying/Creating raw datasets...")
        from torch_geometric.datasets import TUDataset
        from utils.utils_method import get_graph_diameter 
        
        for ds in datasets_config:
            TUDataset(root=ds['path'], 
                      name=ds['name'], 
                      pre_transform=get_graph_diameter,
                      use_node_attr=ds['use_node_attr'])
        print("[Rank 0] Raw datasets loaded and ready for the concurrent reading.")

    # All the processes stop at this point until Rank 0 finish.
    comm.Barrier()
    # -------------------------------------------
    
    # 5. Each rank runs the function for its hyperparameters configurations
    for task in my_tasks:
        print(f"[Rank {rank}] Generating for {task['dataset_name']}: M={task['M']}, max_k={task['k']}, n_units={task['n_units']}")
        
        Create_Reservoir_Dataset(
            native_dataset_path=task['dataset_path'],
            native_dataset_name=task['dataset_name'],
            n_units=int(task['n_units']),
            n_classes=task['n_classes'],
            max_k=int(task['k']),
            run=run,
            adjacency_matrix=task['M'],
            use_node_attr=task['use_node_attr']
        )

    # 6. Waiting all
    comm.Barrier()
    if rank == 0:
        print("All the pre-processed versions of the datasets have been created successfully in parallel!")

if __name__ == '__main__':
    main()