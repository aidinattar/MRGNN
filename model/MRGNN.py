import torch
from torch.functional import F
torch.set_printoptions(profile="full")
from torch_geometric.nn.conv.graph_conv import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd, global_max_pool as gmin
from torch_geometric.utils import get_laplacian
from math import floor
#cose aggiunte
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#######
from utils.Linear_masked_weight import Linear_masked_weight
from torch.nn.utils import spectral_norm

class MRGNN(torch.nn.Module):
    def __init__(self,in_channels, out_channels , n_class=2, drop_prob=0.5, max_k=3, output=None,
                 reservoir_act_fun = lambda x: x,  device=None):
        super(MRGNN, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_class = n_class
        self.output = output
        self.reservoir_act_fun = reservoir_act_fun
        self.max_k=max_k

        self.lin = spectral_norm(Linear_masked_weight(self.in_channels*(max_k), self.out_channels*(max_k))) #SPECTRAL NORM
        self.dropout = torch.nn.Dropout(p=drop_prob)

        #xhi_layer_mask
        xhi_layer_mask=[]
        for i in range(max_k):
            mask_ones = torch.ones(out_channels, in_channels * (i + 1)).to(self.device)
            mask_zeros=torch.zeros(out_channels,in_channels*(max_k-(i+1))).to(self.device)
            xhi_layer_mask.append(torch.cat([mask_ones,mask_zeros],dim=1))

        self.xhi_layer_mask=torch.cat(xhi_layer_mask,dim=0).to(self.device)
        self.bn_hidden_rec = torch.nn.BatchNorm1d(self.out_channels * max_k)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * max_k *3)
        self.out_fun = torch.nn.LogSoftmax(dim=1)

        self.lin1 = torch.nn.Linear(self.out_channels * max_k *3, self.out_channels* max_k * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * max_k *2, self.out_channels*max_k)
        self.lin3 = torch.nn.Linear(self.out_channels*max_k, self.n_class)

        if output == "one_layer":
            self.lin1 = torch.nn.Linear(self.out_channels * max_k * 3, self.n_class)

        elif output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * max_k * 3, floor(self.out_channels/2) * max_k)
            self.lin2 = torch.nn.Linear(floor(self.out_channels/2) * max_k, self.n_class)

        self.reset_parameters()

    def reset_parameters(self):

        print("reset parameters")
        self.bn_hidden_rec.reset_parameters()
        self.bn_out.reset_parameters()
        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data, hidden_layer_aggregator=None):
        X = data.x
        k = self.max_k

        #compute Laplacian
        L_edge_index, L_values =get_laplacian(data.edge_index, normalization="sym")
        L=torch.sparse.FloatTensor(L_edge_index,L_values,torch.Size([X.shape[0],X.shape[0]])).to_dense()

        H = [X]
        for i in range(k-1):
            xhi_layer_i=torch.mm(torch.matrix_power(L,i+1),X)
            H.append(xhi_layer_i)

        H=self.lin(torch.cat(H, dim=1), self.xhi_layer_mask)
        H = self.reservoir_act_fun(H)
        H = self.bn_hidden_rec(H)

        H_avg=gap(H, data.batch)
        H_add=gadd(H, data.batch)
        H_max=gmp(H, data.batch)

        H=torch.cat([H_avg, H_add, H_max],dim=1)


        if self.output=="funnel" or self.output is None:
            return self.funnel_output(H)
        elif self.output=="one_layer":
            return self.one_layer_out(H)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(H)
        else:
            assert False, "error in output stage"

    def add_unitary_x(self,data):

        data.x = torch.ones(data.num_nodes,1)
        return data

    def get_TANH_resevoir_A(self, data):
        tanh = torch.nn.Tanh()
        if data.x is None:
            data = self.add_unitary_x(data)

        X = data.x

        k = self.max_k

        # compute adjacency matrix A
        adjacency_indexes = data.edge_index
        A_rows = adjacency_indexes[0]
        A_data = [1] * A_rows.shape[0]
        v_index = torch.FloatTensor(A_data).to(self.device)
        A_shape = [X.shape[0], X.shape[0]]
        A = torch.sparse.FloatTensor(adjacency_indexes, v_index, torch.Size(A_shape)).to_dense()

        H = [X]

        xhi_layer_i = X
        for i in range(k - 1):
            xhi_layer_i = tanh(torch.mm(A, xhi_layer_i))
            H.append(xhi_layer_i)

        H = self.lin(torch.cat(H, dim=1), self.xhi_layer_mask)

        data.reservoir = H

        return data

    def get_TANH_resevoir_D(self, data):
        tanh = torch.nn.Tanh()
        relu = torch.nn.ReLU()

        if data.x is None:
            data = self.add_unitary_x(data)

        X = data.x
        k = self.max_k

        # Calcola la matrice di adiacenza A
        adjacency_indexes = data.edge_index
        A_rows = adjacency_indexes[0]
        A_data = [1] * A_rows.shape[0]
        v_index = torch.FloatTensor(A_data).to(self.device)
        A_shape = [X.shape[0], X.shape[0]]
        A = torch.sparse.FloatTensor(adjacency_indexes, v_index, torch.Size(A_shape)).to_dense()

        I = torch.eye(A.shape[0]).to(self.device)

        # Calcola la matrice Laplaciana
        L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym")
        L = torch.sparse.FloatTensor(L_edge_index, L_values, torch.Size([X.shape[0], X.shape[0]])).to_dense()

        lam = I - 0.5*L
        mu= I + 0.66667*L

        H = [X]

        xhi_layer_i = X

        for i in range(k - 1):

            if i % 2 == 0:
              xhi_layer_i = torch.mm(lam, xhi_layer_i)
            else:
              xhi_layer_i = torch.mm(mu, xhi_layer_i)
            H.append(xhi_layer_i)

        H = self.lin(torch.cat(H, dim=1), self.xhi_layer_mask)
        data.reservoir = H

        return data

    def get_TANH_resevoir_L(self, data):

        tanh=torch.nn.Tanh()
        if data.x is None:
            data = self.add_unitary_x(data)

        X = data.x
        k = self.max_k

        #compute Laplacian
        L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym")
        L = torch.sparse.FloatTensor(L_edge_index, L_values, torch.Size([X.shape[0], X.shape[0]])).to_dense()

        H = [X]
        xhi_layer_i=X
        for i in range(k - 1):
            xhi_layer_i = tanh(torch.mm(L, xhi_layer_i))
            H.append(xhi_layer_i)

        H = self.lin(torch.cat(H, dim=1), self.xhi_layer_mask)
        data.reservoir = H

        return data


    def get_TANH_resevoir_A_PROTEINS(self,data):

        data.x = data.x[:, 1:]
        return self.get_TANH_resevoir_A(data)

    def get_TANH_resevoir_L_PROTEINS(self,data):

        data.x = data.x[:, 1:]
        return self.get_TANH_resevoir_L(data)


    def readout_fw(self, data):
        H = data.reservoir
        H = self.reservoir_act_fun(H)
        H = self.bn_hidden_rec(H)
        H_avg = gap(H, data.batch)
        H_add = gadd(H, data.batch)
        H_max = gmp(H, data.batch)
        H = torch.cat([H_avg, H_add, H_max], dim=1)  # torch.cat([H_avg, H_add, H_max, H_min],dim=1)

        if self.output == "funnel" or self.output is None:
            return self.funnel_output(H)
        elif self.output == "one_layer":
            return self.one_layer_out(H)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(H)
        elif self.output == "svm":
            return self.svm_output(H)
        else:
            assert False, "error in output stage"



            

    def one_layer_out(self,H):

        x = self.bn_out(H)
        x = self.out_fun(self.lin1(x))

        return x

    def funnel_output(self,H):

        x = self.bn_out(H)
        x = (F.relu(self.lin1(x)))
        x = self.dropout(x)
        x = (F.relu(self.lin2(x)))
        x = self.dropout(x)
        x = self.out_fun(self.lin3(x))

        return x

    def restricted_funnel_output(self, H):

        x = self.bn_out(H)
        x = self.dropout(x)
        x = (F.relu(self.lin1(x)))
        x = self.dropout(x)
        x = self.out_fun(self.lin2(x))

        return x
