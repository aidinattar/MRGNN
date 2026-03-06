import os 
import numpy as np
import torch
from mpi4py import MPI
# from torch_geometric.utils import to_scipy_sparse_matrix # Import for GNN to be put

def main():
    # 1. MPI environment initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Id of this specific process (ex. 0,1,2,...) 
    size = comm.Get_size() # Total number of processes running in parallel

    # 2. Repository for the cache (safe way)
    cache_dir = ".pre_processed_dataset"

    # Only rank 0 creates the repo to avoid conflicts
    if rank == 0:
        os.makedirs(cache_dir, exist_ok = True)
        print(f"Process [Rank0] created the cache directory '{cache_dir}'.")


    # Synchronize all processes to ensure the directory is created before any process accesses it
    comm.Barrier()

    # 3. How many graphs there are? (Let's say 100)
    # Here the length of my dataset has to be read
    num_total_graphs = 100 #len(my_dataset)
    all_indices = np.arange(num_total_graphs)

    # 4. Division of labor: split the indices of the graphs among the ranks. Each rank gets a subset of the indices to work on.
    # np.array_split is a convenient way to split an array into nearly equal parts. Each rank will get a different part of the indices.
    my_indices = np.array_split(all_indices, size)[rank]
    
    print(f"[Rank {rank}] I manage {len(my_indices)} graphs: from {my_indices[0]} to {my_indices[-1]}")

    # 5. Each rank processes its assigned graphs
    for idx in my_indices:
        
        # a) Raw graph loading
        # raw_graph = my_dataset[idx]
        grafo_grezzo = {"id": idx, "dati": "matrice_densa_fittizia"} 

        # b) LA MATEMATICA (Qui in futuro scriveremo il codice vero)
        # - Trasforma in CSR
        # - Normalizza
        # - Estrai feature multiscala
        # grafo_processato = esegui_trasformazioni(grafo_grezzo)
        grafo_processato = grafo_grezzo # Per ora lo lasciamo inalterato
        
        # c) Save in the cache with a unique name (e.g., "grafo_pronto_0.pt", "grafo_pronto_1.pt", etc.)
        save_path = os.path.join(cache_dir, f"grafo_pronto_{idx}.pt")
        torch.save(grafo_processato, save_path)

    # 6. Synchronization point: wait for all ranks to finish processing their graphs before proceeding (optional but good practice)
    comm.Barrier() # This ensures that all ranks have completed their processing before any rank moves on to the next step (e.g., training, evaluation, etc.)
    if rank == 0:
        print("Preprocessing distribuito completato da tutti i rank con successo!")

if __name__ == "__main__":
    main()

