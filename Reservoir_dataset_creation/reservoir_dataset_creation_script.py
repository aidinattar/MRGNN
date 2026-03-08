import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Reservoir_dataset_creation.ReservoirExtraction import Create_Reservoir_Dataset


run=[0,1,2,3,4]
adj_mat_list = ['D','A','L']


if __name__ == '__main__':

    max_k_list = [3, 4, 5, 6]

    #ENZYMES
    print("ENZYMES...")
    n_classes = 6
    n_units_list = [50,100]
    use_node_attr = True


    native_dataset_path = '~/Dataset/ENZYMES'
    native_dataset_name = 'ENZYMES'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:

                Create_Reservoir_Dataset(native_dataset_path = native_dataset_path ,
                                         native_dataset_name = native_dataset_name ,
                                         n_units = n_units,
                                         n_classes = n_classes,
                                         max_k = k,
                                         run = run,
                                         adjacency_matrix= M,
                                         use_node_attr=use_node_attr)
