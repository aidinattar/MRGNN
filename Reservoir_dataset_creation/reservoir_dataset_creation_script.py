import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Reservoir_dataset_creation.ReservoirExtraction import Create_Reservoir_Dataset


run=[0,1,2,3,4]
adj_mat_list=['A','L','D']


if __name__ == '__main__':

    max_k_list = [3, 4, 5, 6]


    ## PTC
    print("PTC...")
    n_classes = 2
    n_units_list = [15,30]
    use_node_attr=False


    native_dataset_path = '~/Dataset/PTC_MR'
    native_dataset_name = 'PTC_MR'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:
                Create_Reservoir_Dataset(native_dataset_path = native_dataset_path ,
                                         native_dataset_name = native_dataset_name ,
                                         n_units = n_units,
                                         n_classes = n_classes,
                                         max_k = k  ,
                                         run = run,
                                         adjacency_matrix= M,
                                         use_node_attr=use_node_attr)

    #PROTEINS
    print("PROTEINS...")
    n_classes = 2
    n_units_list = [25,50]
    use_node_attr = False

    native_dataset_path = '~/Dataset/PROTEINS'
    native_dataset_name = 'PROTEINS'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:

                Create_Reservoir_Dataset(native_dataset_path = native_dataset_path ,
                                         native_dataset_name = native_dataset_name ,
                                         n_units = n_units,
                                         n_classes = n_classes,
                                         max_k = k  ,
                                         run = run,
                                         adjacency_matrix= M,
                                         use_node_attr=use_node_attr)

    #NCI
    print("NCI...")
    n_classes = 2
    n_units_list = [50,100]
    use_node_attr = False


    native_dataset_path = '~/Dataset/NCI1'
    native_dataset_name = 'NCI1'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:

                Create_Reservoir_Dataset(native_dataset_path = native_dataset_path ,
                                         native_dataset_name = native_dataset_name ,
                                         n_units = n_units,
                                         n_classes = n_classes,
                                         max_k = k  ,
                                         run = run,
                                         adjacency_matrix= M,
                                         use_node_attr=use_node_attr)


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

    max_k_list = [3,7,8,9]

    #COLLAB
    print("COLLAB...")
    n_classes = 3
    n_units_list = [15,30,60]
    use_node_attr = False

    native_dataset_path = '~/Dataset/COLLAB'
    native_dataset_name = 'COLLAB'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:
                Create_Reservoir_Dataset(native_dataset_path=native_dataset_path,
                                         native_dataset_name=native_dataset_name,
                                         n_units=n_units,
                                         n_classes=n_classes,
                                         max_k=k,
                                         run=run,
                                         adjacency_matrix=M,
                                         use_node_attr=use_node_attr)

    #IMDB-B
    print("IMDB-BINARY...")
    n_classes = 3
    n_units_list = [15,30,60]
    use_node_attr = False

    native_dataset_path = '~/Dataset/IMDB-BINARY'
    native_dataset_name = 'IMDB-BINARY'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:
                Create_Reservoir_Dataset(native_dataset_path=native_dataset_path,
                                         native_dataset_name=native_dataset_name,
                                         n_units=n_units,
                                         n_classes=n_classes,
                                         max_k=k,
                                         run=run,
                                         adjacency_matrix=M,
                                         use_node_attr=use_node_attr)

    #IMDB-M
    print("IMDB-MULTI...")
    n_classes = 3
    n_units_list = [15,30,60]
    use_node_attr = False

    native_dataset_path = '~/Dataset/IMDB-MULTI'
    native_dataset_name = 'IMDB-MULTI'

    for M in adj_mat_list:
        for k in max_k_list:
            for n_units in n_units_list:
                Create_Reservoir_Dataset(native_dataset_path=native_dataset_path,
                                         native_dataset_name=native_dataset_name,
                                         n_units=n_units,
                                         n_classes=n_classes,
                                         max_k=k,
                                         run=run,
                                         adjacency_matrix=M,
                                         use_node_attr=use_node_attr)

