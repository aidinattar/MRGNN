import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

from model.MRGNN import MRGNN
from impl.binGraphClassifier import modelImplementation_GraphBinClassifier
from utils.utils_method import printParOnFile
from data_reader.cross_validation_reader import getcross_validation_split_4_reservoir

if __name__ == '__main__':
    from mpi4py import MPI
    import itertools

    # --- MPI INITIALIZATION---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Definition of the hyperparameters space
    run = [0, 1, 2, 3, 4]
    n_epochs = 200
    n_classes = 6
    n_folds = 10
    test_epoch = 1
    dataset_name = 'ENZYMES'
    
    n_units_list = [50, 100]
    lr_list = [1e-3, 1e-4]
    drop_prob_list = [0.3, 0.5]
    weight_decay_list = [5e-3, 5e-4]
    batch_size_list = [16, 32]
    max_k_list = [3, 4, 5, 6]
    output_list = ["funnel", "restricted_funnel", "one_layer"]
    aggregator = 'concat'
    adjacency_matrix_list = ['D', 'A', 'L']

    reservoir_augmented_dataset_root = '~/Dataset/Reservoir_TANH_Dataset/'

    # 2. Cartesian product
    # Create a 1D list of the 3840 possible combinations of hyperparameters
    all_combinations = list(itertools.product(
        run, n_units_list, lr_list, drop_prob_list, weight_decay_list, 
        batch_size_list, max_k_list, output_list, adjacency_matrix_list
    ))

    # 3. MPI PARTITIONING
    local_combinations = all_combinations[rank::size]
    print(f"[Rank {rank}] assigned {len(local_combinations)} run out of {len(all_combinations)} total.")

    # 4. ISOLATED EXECUTION
    for config in local_combinations:
        # touple of the current configuration of hyperparameters
        r, n_units, lr, drop_prob, weight_decay, batch_size, max_k, output, adjacency_matrix = config

        # --- from here the logic is the one of the original model ---
        current_reservoir_augmented_dataset_name = "run_" + str(r) + '_TANH_RES_' + \
                                                   str(adjacency_matrix) + "_" + \
                                                   str(max_k) + "_n_units_" + \
                                                   str(n_units) + '_' + dataset_name
        
        dataset_path = os.path.join(reservoir_augmented_dataset_root, current_reservoir_augmented_dataset_name)
        
        test_name = "run_" + str(r) + "_MRGNN_data-" + dataset_name + \
                    "_adjacency_matrix-" + adjacency_matrix + \
                    "_nFold-" + str(n_folds) + "_lr-" + str(lr) + \
                    "_drop_prob-" + str(drop_prob) + "_weight-decay-" + str(weight_decay) + \
                    "_batchSize-" + str(batch_size) + "_nHidden-" + str(n_units) + \
                    "_output-" + str(output) + "_maxK-" + str(max_k)

        training_log_dir = os.path.join("./test_log/", test_name)
        
        # FIX RACE CONDITION: exist_ok=True avoids crash if two ranks
        # try to open the directory "./test_log/" at the same time.
        os.makedirs(training_log_dir, exist_ok=True)

        printParOnFile(test_name=test_name, log_dir=training_log_dir,
                       par_list={"dataset_name": dataset_name, "n_fold": n_folds,
                                 "learning_rate": lr, "drop_prob": drop_prob,
                                 "weight_decay": weight_decay, "batch_size": batch_size,
                                 "n_hidden": n_units, "test_epoch": test_epoch,
                                 "aggregator": aggregator, "output": output, "max_k": max_k})

        # device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.NLLLoss()

        dataset_cv_splits = getcross_validation_split_4_reservoir(dataset_path, dataset_name, n_folds, batch_size)
        
        print(f"[Rank {rank}] Loading {test_name}")
        
        for split_id, split in enumerate(dataset_cv_splits):
            loader_train = split[0]
            loader_test = split[1]
            loader_valid = split[2]

            model = MRGNN(loader_train.dataset.num_features, n_units, n_classes,
                          drop_prob, max_k=max_k, output=output).to(device)

            model_impl = modelImplementation_GraphBinClassifier(model, lr, criterion, device).to(device)
            model_impl.set_optimizer(weight_decay=weight_decay)
            
            # TRAINING AND TESTING
            model_impl.train_test_model_readout(split_id, loader_train, loader_test,
                                                loader_valid, n_epochs, test_epoch,
                                                test_name, training_log_dir)




'''
if __name__ == '__main__':

    run = [0, 1, 2, 3, 4]
    n_epochs = 200
    n_classes = 6
    n_folds = 10
    test_epoch = 1
    dataset_name = 'ENZYMES'

    n_units_list = [50, 100]
    lr_list = [1e-3, 1e-4]
    drop_prob_list = [0.3, 0.5]
    weight_decay_list = [5e-3, 5e-4]
    batch_size_list = [16, 32]
    max_k_list = [3, 4, 5, 6]
    output_list = ["funnel", "restricted_funnel", "one_layer"]
    aggregator = 'concat'
    adjacency_matrix_list = ['A', 'L']

    reservoir_augmented_dataset_root = '~/Dataset/Reservoir_TANH_Dataset/'

    for r in run:
        for n_units in n_units_list:
            for lr in lr_list:
                for drop_prob in drop_prob_list:
                    for weight_decay in weight_decay_list:
                        for batch_size in batch_size_list:
                            for max_k in max_k_list:
                                for output in output_list:
                                    for adjacency_matrix in adjacency_matrix_list:

                                        current_reservoir_augmented_dataset_name = "run_" + str(r) + '_TANH_RES_' + \
                                                                                   str(adjacency_matrix) + "_" + \
                                                                                   str(max_k) + "_n_units_" + \
                                                                                   str(n_units) + '_' + dataset_name
                                        dataset_path = os.path.join(reservoir_augmented_dataset_root,
                                                                    current_reservoir_augmented_dataset_name)
                                        test_name = "run_" + str(r) + "_MRGNN"

                                        test_name = test_name + "_data-" + dataset_name + \
                                                    "_adjacency_matrix-" + adjacency_matrix + \
                                                    "_nFold-" + str(n_folds) + \
                                                    "_lr-" + str(lr) + \
                                                    "_drop_prob-" + str(drop_prob) + \
                                                    "_weight-decay-" + str(weight_decay) + \
                                                    "_batchSize-" + str(batch_size) + \
                                                    "_nHidden-" + str(n_units) + \
                                                    "_output-" + str(output) + \
                                                    "_maxK-" + str(max_k)

                                        training_log_dir = os.path.join("./test_log/", test_name)
                                        if not os.path.exists(training_log_dir):
                                            os.makedirs(training_log_dir)

                                        printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                       par_list={"dataset_name": dataset_name,
                                                                 "n_fold": n_folds,
                                                                 "learning_rate": lr,
                                                                 "drop_prob": drop_prob,
                                                                 "weight_decay": weight_decay,
                                                                 "batch_size": batch_size,
                                                                 "n_hidden": n_units,
                                                                 "test_epoch": test_epoch,
                                                                 "aggregator": aggregator,
                                                                 "output": output,
                                                                 "max_k": max_k})

                                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                        criterion = torch.nn.NLLLoss()

                                        dataset_cv_splits = getcross_validation_split_4_reservoir(dataset_path,
                                                                                                  dataset_name, n_folds,
                                                                                                  batch_size)
                                        for split_id, split in enumerate(dataset_cv_splits):
                                            loader_train = split[0]
                                            loader_test = split[1]
                                            loader_valid = split[2]

                                            model = MRGNN(loader_train.dataset.num_features, n_units, n_classes,
                                                          drop_prob, max_k=max_k, output=output).to(device)

                                            model_impl = modelImplementation_GraphBinClassifier(model, lr, criterion,
                                                                                                device, ).to(device)

                                            model_impl.set_optimizer(weight_decay=weight_decay)

                                            model_impl.train_test_model_readout(split_id, loader_train, loader_test,
                                                                                loader_valid, n_epochs, test_epoch,
                                                                                test_name, training_log_dir)
'''