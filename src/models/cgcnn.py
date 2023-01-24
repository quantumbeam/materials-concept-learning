import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import global_max_pool, global_mean_pool
from models.utils import normalize_embedding

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea,
             nbr_fea
             ],dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class CGCNN(torch.nn.Module):
    """
    xrd_encoder: str
        XRDのデータポイント数に合わせてXRDのエンコーダ（1D CNN）の設定を切り替える。
        xrd_features_BN_9kは9000点、〜_5kは5000点
    """
    def __init__(self, params, output_intermediate_feat=False,
                 orig_atom_fea_len=92, nbr_fea_len=41,
                 atom_fea_len=64, n_conv=3):
        super(CGCNN, self).__init__()
        embedding_dim = params.embedding_dim
        self.params = params
        self.output_intermediate_feat = output_intermediate_feat

        # CGCNN部分
        self.cg_in_lin1 = nn.Linear(orig_atom_fea_len, atom_fea_len) # atom_fea_lenはデフォルト64だが、1024ぐらいまで大きくしたほうがいいか
        self.cg_convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)]) #n_convの数だけConvLayerを繰り返し定義
        self.cg_in_lin2 = nn.Linear(atom_fea_len, embedding_dim)
        self.cg_in_bn1 = nn.BatchNorm1d(num_features=embedding_dim)
        
        self.crystal_lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.cry_bn1 = nn.BatchNorm1d(num_features=embedding_dim)
        self.crystal_lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.cry_bn2 = nn.BatchNorm1d(num_features=embedding_dim)

        if not hasattr(params, 'targets') or params.targets is None:
            # embedding mode when targets = None
            self.crystal_lin3 = nn.Linear(embedding_dim, embedding_dim)
        else:
            # regression mode when targets = "hoge" or ["hoge", "foo"]
            final_dim = 1 if isinstance(params.targets, str) \
                else len(params.targets)
            self.crystal_regression = nn.Linear(embedding_dim, final_dim)


    def forward(self, data):
        #atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = cg_features

        # Convert torch_geometric.Data to CGCNN format
        N, E = data.x.shape[0], data.edge_attr.shape[0]
        atom_fea = data.x
        nbr_fea = data.edge_attr.reshape(N, E//N, -1)
        nbr_fea_idx = data.edge_index[1].reshape(N, E//N) # Use only col indices

        atom_fea = self.cg_in_lin1(atom_fea)
        for conv_func in self.cg_convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Use torch_geometric's pooling function instead.
        # x_crystal = self.pooling(atom_fea, crystal_atom_idx)
        x_crystal = global_mean_pool(atom_fea, data.batch)
        del atom_fea

        x_crystal = self.cg_in_lin2(x_crystal)
        x_crystal = F.relu(self.cg_in_bn1(x_crystal))

        # for encoder
        if self.output_intermediate_feat:
          out1 = F.relu(self.cry_bn1(self.crystal_lin1(x_crystal)))
          out2 = F.relu(self.cry_bn2(self.crystal_lin2(out1)))
          return out1, out2

        output_crystal = self.crystal_lin1(x_crystal)
        output_crystal = F.relu(self.cry_bn1(output_crystal))
        output_crystal = self.crystal_lin2(output_crystal)
        output_crystal = F.relu(self.cry_bn2(output_crystal))

        if hasattr(self, 'crystal_regression'):
            return self.crystal_regression(output_crystal)
        
        output_crystal = self.crystal_lin3(output_crystal)
        output_crystal = normalize_embedding(output_crystal, self.params.embedding_normalize)
        
        return output_crystal
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)    


