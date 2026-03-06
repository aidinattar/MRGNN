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
    adj_mat_list = ['L']
    max_k_list = [3, 4, 5, 6]
    n_units_list = [50, 100]

    native_dataset_path = '~/Dataset/ENZYMES'
    native_dataset_name = 'ENZYMES'
    n_classes = 6
    use_node_attr = True

    # 3. List that represents the cartesian product of all hyperparameters (3840 combinations)
    # In this case 1 * 4 * 2 = 8 task in total
    all_tasks = list(itertools.product(adj_mat_list, max_k_list, n_units_list))

    # 4. Task division among ranks
    my_tasks = np.array_split(all_tasks, size)[rank]
    
    print(f"[Rank {rank}] Assegnati {len(my_tasks)} task su {len(all_tasks)} totali.")

    # --- FIX RACE CONDITION ON THE RAW DATASET ---
    if rank == 0:
        print("[Rank 0] Verifying/Creating raw dataset...")
        from torch_geometric.datasets import TUDataset
        # Ensure that get_graph_diameter point to the right function, since it is imported in ReservoirExtraction.py and not here
        from utils.utils_method import get_graph_diameter 
        
        # Rank 0 loads files if they are missing. If they are already there, it just verifies that they are correct. In both cases, it prepares the dataset for concurrent reading.
        TUDataset(root=native_dataset_path, 
                  name=native_dataset_name, 
                  pre_transform=get_graph_diameter,
                  use_node_attr=use_node_attr)
        print("[Rank 0] Raw dataset loaded and ready for the concurrent reading.")

    # All the processes stop at this point until Rank 0 finish.
    comm.Barrier()
    # -------------------------------------------
    
    # 5. Each rank runs the function for its hyperparameters configurations
    for task in my_tasks:
        M, k, n_units = task
        # Cast to int since array_split could cast to np.int64
        k = int(k)
        n_units = int(n_units)
        
        print(f"[Rank {rank}] Generating: M={M}, max_k={k}, n_units={n_units}")
        
        Create_Reservoir_Dataset(
            native_dataset_path=native_dataset_path,
            native_dataset_name=native_dataset_name,
            n_units=n_units,
            n_classes=n_classes,
            max_k=k,
            run=run,
            adjacency_matrix=M,
            use_node_attr=use_node_attr
        )

    # 6. Waiting all
    comm.Barrier()
    if rank == 0:
        print("All the pre-processed versions of ENZYMES dataset had been created successfully in parallel!")

if __name__ == '__main__':
    main()