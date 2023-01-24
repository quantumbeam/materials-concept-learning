import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, global_mean_pool
from models.utils import MLP, normalize_scale, normalize_embedding

# def get_cart_coords(data):
#     coords = torch.index_select(data.trans_vec, 0, data.batch)
#     coords = (coords * data.pos[:, :, None]).sum(axis=1)
#     return coords

class SAModule(torch.nn.Module):
    """
    Set Abstraction module
    """
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio) if self.ratio<1 else Ellipsis
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch



class CrystalEncoder(torch.nn.Module):
    """
    ネットワーク本体
    """
    def __init__(self, params, output_intermediate_feat=False):
        super(CrystalEncoder, self).__init__()
        SA1_sample_ratio = params.SA1_sample_ratio
        SA1_sample_r = params.SA1_sample_r
        SA2_sample_ratio = params.SA2_sample_ratio
        SA2_sample_r = params.SA2_sample_r
        embedding_dim = params.embedding_dim
        self.params = params
        self.output_intermediate_feat = output_intermediate_feat
        
        # PointNet2の部分
        self.ATOM_FEAT_DIM = 98
        if self.params.use_cgcnn_feat:
            from models.cgcnn_atom_features import atom_features
            self.atom_feat = atom_features[:self.ATOM_FEAT_DIM]
            self.atom_feat = torch.tensor(self.atom_feat, dtype=torch.float, requires_grad=False)
            self.ATOM_FEAT_DIM = self.atom_feat.shape[1]

        self.sa1_module = SAModule(SA1_sample_ratio, SA1_sample_r, MLP([3+self.ATOM_FEAT_DIM, 64, 64, 128]))
        self.sa2_module = SAModule(SA2_sample_ratio, SA2_sample_r, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.crystal_lin1 = nn.Linear(1024, embedding_dim)
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
            
            # define with a different name than 'crystal_lin3'
            # so that pretrained weights are not used for this layer.
            self.crystal_regression = nn.Linear(embedding_dim, final_dim)

    def forward(self, data):
        if self.params.scale_crystal > 0:
            # Do centering here or not changes the behavior 
            # because GlobalSAModule uses absolute positions
            # while SAModule uses relative positions.
            # TODO: better way to get the batch-size?
            batch = data.batch[-1].item()+1
            scale = torch.full((batch, ), 1.0/self.params.scale_crystal, dtype=torch.float, device=data.x.device)
            data.pos, _ = normalize_scale(data.pos, data.batch, scale)
        else:
            data.pos, _ = normalize_scale(data.pos, data.batch)

        if self.params.use_cgcnn_feat:
            if data.x.device != self.atom_feat.device:
                self.atom_feat = self.atom_feat.to(data.x.device)
            # Matrix multiplication: (N, 98) x (98, C) = (N, C)
            data.x = data.x @ self.atom_feat

        # PN2の出力
        sa0_out = (data.x.squeeze(), data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x_crystal, pos, batch = sa3_out

        # for encoder
        if self.output_intermediate_feat:
          out1 = F.relu(self.cry_bn1(self.crystal_lin1(x_crystal)))
          out2 = F.relu(self.cry_bn2(self.crystal_lin2(out1)))
          return out1, out2

        output = self.crystal_lin1(x_crystal)
        output = F.relu(self.cry_bn1(output))
        output = self.crystal_lin2(output)
        output = F.relu(self.cry_bn2(output))

        if hasattr(self, 'crystal_regression'):
            return self.crystal_regression(output)
        
        output = self.crystal_lin3(output)
        output = normalize_embedding(output, self.params.embedding_normalize)
        
        return output
