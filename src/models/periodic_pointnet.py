import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
from pymatgen.core.structure import Structure
from models.pointnet import SAModule, GlobalSAModule
from models.utils import MLP, normalize_scale, normalize_embedding

class PeriodicSAModule(torch.nn.Module):
    """
    Periodic Structure Abstraction module.
    Adopts the neighbor detection in the phase space,
    which can detect the presense of mirror atoms in the radius
    but is unaware of individual instances of mirror atoms
    and their positions.
    """
    def __init__(self, ratio, r, nn):
        super(PeriodicSAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, fps_pos, batch, frac_pos, trans_vec, scale):
        idx = fps(fps_pos, batch, ratio=self.ratio) if self.ratio < 1 else Ellipsis

        query = frac_pos[idx]
        rows, cols = [], []
        total_row = total_col = 0
        flag = False
        for i in range(trans_vec.shape[0]):
            b = batch==i
            d = frac_pos[b][None, :] - query[b[idx]][:, None]    # (Q,N,3)
            if flag:
                d = d - torch.floor(d)
                d = d.unsqueeze(1).repeat(1,8,1,1)          # (Q,8,N,3)
                d[:, 1, :, 0] -= 1
                d[:, 2, :, 1] -= 1
                d[:, 3, :, 2] -= 1
                d[:, 4, :, 0:2] -= 1
                d[:, 5, :, 1:3] -= 1
                d[:, 6, :, 0] -= 1
                d[:, 6, :, 2] -= 1
                d[:, 7, :, :] -= 1
            D = d.reshape(-1, 3) @ trans_vec[i]         # (Q4N,3)@(3,3)
            D = D.reshape(d.shape)                      # (Q,8,N,3)
            D = (D*D).sum(dim=-1)                           # (Q,8,N)
            if flag:
                D = D.min(dim=1)[0]                             # (Q,N)
            M = D < (self.r/scale[i])**2
            r, c = D.shape[0:2]
            s = torch.argsort(D, dim=1)
            j = torch.arange(r,dtype=torch.long, device=x.device)[:, None]
            M = M[j, s]
            if M.shape[1] > 64:
                M[:, 64:] = False
            row = torch.arange(r, dtype=torch.long, device=D.device)[:,None].repeat(1,c)[j, s][M]
            col = torch.arange(c, dtype=torch.long, device=D.device)[None,:].repeat(r,1)[j, s][M]
            row += total_row
            col += total_col
            total_row += r
            total_col += c
            rows.append(row)
            cols.append(col)

        row = torch.cat(rows)
        col = torch.cat(cols)
        # row, col = radius(frac_pos, frac_pos[idx], self.r, batch, batch[idx],
        #                   max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        frac_pos = frac_pos[idx]
        fps_pos = fps_pos[idx]
        return x, pos, fps_pos, batch, frac_pos, trans_vec, scale

class PeriodicSAModule2(torch.nn.Module):
    """
    Periodic Structure Abstraction module.
    Adopts the neighbor detection in the real space,
    which enumerates all the instances of mirror atoms in the radius.
    Because it uses pymatgen's neighbor function performed on a CPU,
    the neighbor detection is accurate but slow.
    """
    def __init__(self, ratio, r, nn):
        super(PeriodicSAModule2, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, fps_pos, batch, frac_pos, trans_vec, scale):
        idx = fps(fps_pos, batch, ratio=self.ratio) if self.ratio<1 else Ellipsis
        
        rows, cols = [], []
        total_row = total_col = 0
        lattice_mat = trans_vec.cpu().numpy()
        poss = []
        expand = []
        total_p = 0
        for i in range(trans_vec.shape[0]):
            b = batch==i
            p = frac_pos[b].detach().cpu().numpy()     # (N,3)
            q = frac_pos[idx[b]].detach().cpu().numpy()   # (Q,3)
            struct = Structure(lattice_mat[i], [""]*p.shape[0], p)
            q_inds, p_inds, offset, dist = struct.get_neighbor_list(self.r/scale[i], None, exclude_self=True)
            
            q_inds = torch.tensor(q_inds, dtype=torch.long, device=x.device)
            p_inds = torch.tensor(p_inds, dtype=torch.long, device=x.device)
            offset = torch.tensor(offset, dtype=torch.float32, device=x.device)

            r = q.shape[0]
            c = p_inds.shape[0]
            p_pos = pos[b][p_inds]
            p_pos += (offset[:,:,None]*trans_vec[i:i+1]*scale[i]).sum(dim=2)   # (N,3,1)*(1,3,3)
            row = q_inds
            col = torch.arange(c, dtype=torch.long, device=x.device)
            row += total_row
            col += total_col
            p_inds += total_p
            total_row += r
            total_col += c
            total_p += p.shape[0]
            rows.append(row)
            cols.append(col)
            poss.append(p_pos)
            expand.append(p_inds)

            # TODO: 
            # x = x[col]
            # pos = torch.cat(poss)
            # inx = (offset==0).sum(dim=1) == 3

        row = torch.cat(rows)
        col = torch.cat(cols)
        
        pos2 = torch.cat(poss, dim=0)
        expand = torch.cat(expand, dim=0)
        x2 = x[expand]

        # row, col = radius(frac_pos, frac_pos[idx], self.r, batch, batch[idx],
        #                   max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x2, (pos2, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        frac_pos = frac_pos[idx]
        fps_pos = fps_pos[idx]
        return x, pos, fps_pos, batch, frac_pos, trans_vec, scale

class PeriodicSAModule3(torch.nn.Module):
    """
    Periodic Structure Abstraction module.
    Adopts the neighbor detection in the real space,
    by repeating the unit cell 3x3 times and computing the radius neighbors
    with the torch_geometric's radius function.
    It is relatively fast but can fail to capture neighbors
    when the unit cell size is smaller than [-1, 1]^3,
    which is very likely if the global normalization is used.
    """
    def __init__(self, ratio, r, nn):
        super(PeriodicSAModule3, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, fps_pos, batch, frac_pos, trans_vec, scale):
        idx = fps(fps_pos, batch, ratio=self.ratio) if self.ratio<1 else Ellipsis
        
        # For a set of points in a unit cell, we compute 27 neighbor images in 3D.
        # Note: it seems that the batch indices for 'radius' must be ordered,
        # ie, B=[0,0,0,1,1,2,3,3,3,...]. For this reason, we repeat points to get p
        # in the shape of (N,27,3) rather than (27,N,3). Repeating tensors in this way
        # may cause new GPU memory allocations (especially after reshaping it to (N*27,3)).
        N, C = x.shape[0:2]
        offset = torch.arange(-1, 2, dtype=torch.float32, device=x.device)
        offset = torch.meshgrid((offset, offset, offset))       #(3,3,3)x3
        offset = torch.stack(offset, dim=-1).reshape(1,27,3)    #(3,3,3,+3)->(1,27,3)
        p = frac_pos[:,None].expand(N,27,3) + offset            #(N,27,3)
        p = torch.matmul(p[:,:,None], (trans_vec*scale[:,None,None])[batch][:,None])     # (N,27,1,3)@(N,1,3,3)
        p = p.reshape(-1,3)
        q = p.reshape(-1,27,3)[:,13][idx]
        del offset
        # Note: 
        # p and q are not centered. However, 'radius' and 'conv' operations
        # always evaluate p and q in their relative positions, (p-q).
        # Thus, their absolute positions do not affect the algorithm.

        # Find nearest neighbors of q in p with the radius of self.r.
        bp = batch[:,None].repeat(1,27).flatten()
        bq = batch[idx]
        q_inds, p_inds = radius(p, q, self.r, bp, bq, max_num_neighbors=64)
        del bp

        if False:
            edge_index = torch.stack([p_inds, q_inds], dim=0)
            del p_inds, q_inds
            x = self.conv(x.repeat(1,27).reshape(-1,C), (p, q), edge_index)
            del p, q, edge_index
        else:
            # In this version, we remove unnecessary points in p that are
            # never referred in p_inds to reduce memory consumption.
            used_flags = torch.zeros(p.shape[0], dtype=torch.bool, device=x.device)
            used_flags[p_inds] = True
            used_num = used_flags.sum().item()
            ind_table = torch.full((p.shape[0],), -1, dtype=torch.long, device=x.device)
            ind_table[used_flags] = torch.arange(used_num, dtype=torch.long, device=x.device)
            p_inds_new = ind_table[p_inds]
            edge_index = torch.stack([p_inds_new, q_inds], dim=0)
            del p_inds, q_inds, p_inds_new, ind_table

            # Remove unused items in p and x.
            p = p[used_flags]
            # We repeat 'x' 27 times and subsample by used_flags:
            #   x = x.repeat(1,27).reshape(-1,C)[used_flags]
            # Below is equivalent to above but should be more memory-efficient.
            x = x[used_flags.nonzero().flatten()//27]
            del used_flags

            x = self.conv(x, (p, q), edge_index)
            del p, q, edge_index

        batch = bq
        pos = pos[idx]
        frac_pos = frac_pos[idx]
        fps_pos = fps_pos[idx]
        return x, pos, fps_pos, batch, frac_pos, trans_vec, scale


class PeriodicGlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(PeriodicGlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, fps_pos, batch, frac_pos, trans_vec, scale):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        frac_pos = frac_pos.new_zeros((x.size(0), 3))
        fps_pos = fps_pos.new_zeros((x.size(0), 3))
        return x, pos, fps_pos, batch, frac_pos, trans_vec, scale


def get_cart_coords(data):
    coords = torch.index_select(data.trans_vec, 0, data.batch)
    coords = (coords * data.pos[:, :, None]).sum(axis=1)
    return coords

class PeriodicPointNet(torch.nn.Module):
    """
    xrd_encoder: str
        XRDのデータポイント数に合わせてXRDのエンコーダ（1D CNN）の設定を切り替える。
        xrd_features_BN_9kは9000点、〜_5kは5000点
    """
    def __init__(self, params):
        super().__init__()
        SA1_sample_ratio = params.SA1_sample_ratio
        SA1_sample_r = params.SA1_sample_r
        SA2_sample_ratio = params.SA2_sample_ratio
        SA2_sample_r = params.SA2_sample_r
        embedding_dim = params.embedding_dim
        self.params = params

        self.ATOM_FEAT_DIM = 98
        if self.params.use_cgcnn_feat:
            from models.cgcnn_atom_features import atom_features
            self.atom_feat = atom_features[:self.ATOM_FEAT_DIM]
            self.atom_feat = torch.tensor(self.atom_feat, dtype=torch.float, requires_grad=False)
            self.ATOM_FEAT_DIM = self.atom_feat.shape[1]
            
        self.POS_ENC_LEVEL = 0
        self.POS_CHANNELS = (1+self.POS_ENC_LEVEL)*3
        if params.encoder_name=='periodic3':
            self.sa1_module = PeriodicSAModule3(SA1_sample_ratio, SA1_sample_r, MLP([self.POS_CHANNELS+self.ATOM_FEAT_DIM, 64, 64, 128]))
            self.sa2_module = PeriodicSAModule3(SA2_sample_ratio, SA2_sample_r, MLP([128 + self.POS_CHANNELS, 128, 128, 256]))
            self.sa3_module = PeriodicGlobalSAModule(MLP([256 + self.POS_CHANNELS, 256, 512, 1024]))
        elif params.encoder_name=='periodic2':
            self.sa1_module = PeriodicSAModule2(SA1_sample_ratio, SA1_sample_r, MLP([self.POS_CHANNELS+self.ATOM_FEAT_DIM, 64, 64, 128]))
            self.sa2_module = PeriodicSAModule2(SA2_sample_ratio, SA2_sample_r, MLP([128 + self.POS_CHANNELS, 128, 128, 256]))
            self.sa3_module = PeriodicGlobalSAModule(MLP([256 + self.POS_CHANNELS, 256, 512, 1024]))
        elif params.encoder_name=='periodic1':
            self.sa1_module = PeriodicSAModule(SA1_sample_ratio, SA1_sample_r, MLP([self.POS_CHANNELS+self.ATOM_FEAT_DIM, 64, 64, 128]))
            self.sa2_module = PeriodicSAModule(SA2_sample_ratio, SA2_sample_r, MLP([128 + self.POS_CHANNELS, 128, 128, 256]))
            self.sa3_module = PeriodicGlobalSAModule(MLP([256 + self.POS_CHANNELS, 256, 512, 1024]))
        elif params.encoder_name=='default':
            self.sa1_module = SAModule(SA1_sample_ratio, SA1_sample_r, MLP([self.POS_CHANNELS+self.ATOM_FEAT_DIM, 64, 64, 128]))
            self.sa2_module = SAModule(SA2_sample_ratio, SA2_sample_r, MLP([128 + self.POS_CHANNELS, 128, 128, 256]))
            self.sa3_module = GlobalSAModule(MLP([256 + self.POS_CHANNELS, 256, 512, 1024]))
        else:
            raise Exception(f"Not defined: params.encoder_name ({params.encoder_name})")

        self.crystal_lin1 = nn.Linear(1024, embedding_dim) # ここまではpre-trainedなweightが使えるかも
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
        cart_coords = get_cart_coords(data)
        if self.params.scale_crystal > 0:
            # TODO: Somehow the centering normalization changes the behavior.
            # To my thinking, considering that points are always evalauted in
            # relative positions, whether or not we centerize points should not
            # change the results. Needs to be investigated.
            scale = 1.0 / self.params.scale_crystal
            scale = torch.full((data.trans_vec.shape[0], ), scale, dtype=torch.float, device=cart_coords.device)
            cart_coords, scale = normalize_scale(cart_coords, data.batch, scale)
        else:
            cart_coords, scale = normalize_scale(cart_coords, data.batch)
            
        if self.params.use_cgcnn_feat:
            if data.x.device != self.atom_feat.device:
                self.atom_feat = self.atom_feat.to(data.x.device)
            # Matrix multiplication: (N, 98) x (98, C) = (N, C)
            data.x = data.x @ self.atom_feat

        pos = cart_coords
        if self.POS_ENC_LEVEL > 0:
            pos = []
            for k in range(1, self.POS_ENC_LEVEL+1):
                pos.extend([
                    torch.sin(data.pos*2.0*k*np.pi),
                    torch.cos(data.pos*2.0*k*np.pi),
                ])
            pos = torch.cat(pos, dim=1)

        if isinstance(self.sa1_module, (PeriodicSAModule, PeriodicSAModule2, PeriodicSAModule3)):
            sa_out = (data.x.squeeze(), pos, cart_coords, data.batch, data.pos, data.trans_vec, scale)
            sa_out = self.sa1_module(*sa_out)
            sa_out = self.sa2_module(*sa_out)
            sa_out = self.sa3_module(*sa_out)
            x_crystal = sa_out[0]
            del sa_out
        else:
            sa_out = (data.x.squeeze(), cart_coords, data.batch)
            sa_out = self.sa1_module(*sa_out)
            sa_out = self.sa2_module(*sa_out)
            sa_out = self.sa3_module(*sa_out)
            x_crystal = sa_out[0]
            del sa_out
        del cart_coords, pos, scale

        output_cry = self.crystal_lin1(x_crystal)
        output_cry = F.relu(self.cry_bn1(output_cry))
        output_cry = self.crystal_lin2(output_cry)
        output_cry = F.relu(self.cry_bn2(output_cry))
        
        if hasattr(self, 'crystal_regression'):
            return self.crystal_regression(output_cry)
        
        output_cry = self.crystal_lin3(output_cry)
        output_cry = normalize_embedding(output_cry, self.params.embedding_normalize)
        
        return output_cry

