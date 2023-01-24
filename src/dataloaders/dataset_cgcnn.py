import os
import warnings
import functools
from itertools import compress
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm_notebook
from models.cgcnn_atom_features import atom_features
from dataloaders.common import MultimodalDatasetMP, MultimodalDatasetCOD, RegressionDatasetMP

from torch_geometric.data import Data
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from dataloaders.common import generate_site_species_vector

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class ExportCrystalGraph:
    def __init__(self, atom_feat_mode='original', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.atom_feat_mode = atom_feat_mode
        self.ATOM_NUM_UPPER = 98
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.atom_features = np.array(atom_features, dtype=float)

        """
        結晶構造（pymatgenのstructureオブジェクト）からCrystal Graphを生成する。
        
        Parameters
        ----------
        atom_init_path: str
            atom_init.json までのパス
        atom_feat_mode: str
            atom_fea（各原子の特徴ベクトルの初期値）として何を用いるか選ぶ。
            "original"の場合は、atom_init.jsonを用いる（CGCNNのデフォルト）
            "xyz_pos_and_elem_num"の場合は、各原子のデカルト座標（長辺の長さを1に規格化）と原子番号のonehot vectorを使う（PointNetに少し近づける）

        """

    def extract_atom_feature(self, pymg_structure_obj):
        # spe = np.array([site.specie.number for site in pymg_structure_obj])
        # spe = np.eye(self.ATOM_NUM_UPPER)[spe]
        spe = generate_site_species_vector(pymg_structure_obj, self.ATOM_NUM_UPPER)

        xyz = np.vstack([ [site.x, site.y, site.z] for site in pymg_structure_obj])
        pmin, pmax = xyz.min(axis=0, keepdims=True), xyz.max(axis=0, keepdims=True)
        xyz = xyz - (pmin + pmax) / 2
        scale = (pmax-pmin).max() / 2
        xyz = xyz / scale
        return np.concatenate([xyz, spe], axis=1)

    def get_crystal_graph_feature(self, pymg_structure_obj, material_id):

        # crystal graphの特徴量生成
        if self.atom_feat_mode == 'original':
            if hasattr(pymg_structure_obj, 'species'):
                atom_num = np.array([x.specie.number-1 for x in pymg_structure_obj])
                atom_fea = self.atom_features[atom_num]
            else:
                # This code can be generally used for the above case too,
                # but involves redundant mat-mat multiplication when occup is binary.
                occup = generate_site_species_vector(pymg_structure_obj, self.ATOM_NUM_UPPER)
                atom_fea = occup @ self.atom_features[:occup.shape[1]]
                atom_fea = atom_fea.float()

        elif self.atom_feat_mode == 'xyz_pos_and_elem_num':
            atom_fea = self.extract_atom_feature(pymg_structure_obj)
        else:
            raise NameError
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = pymg_structure_obj.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(material_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return atom_fea, nbr_fea, nbr_fea_idx


def make_data(material, exporter, primitive):
    if "final_structure" in material:
        structure = material['final_structure']
    elif "structure" in material:
        structure = material['structure']
    else:
        raise AttributeError("Material has no structure!")

    if not primitive:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

    if "material_id" in material:
        id = material['material_id']
    else:
        id = material['file_id']
        
    atom_fea, nbr_fea, nbr_fea_idx = exporter.get_crystal_graph_feature(structure, id)

    # atom_fea: (N, atom_fea_len)
    # nbr_fea: (N, M, nbr_fea_len)
    # nbr_fea_idx: (N, M)
    N, M, dim = nbr_fea.shape
    edge_attr = nbr_fea.reshape(N*M, dim)

    cols = nbr_fea_idx.reshape(N*M)
    rows = torch.arange(N, dtype=torch.long)
    rows = rows[:, None].expand((N, M)).reshape(N*M)
    edge_index = torch.stack((rows, cols), dim=0)

    data = Data(x=atom_fea, edge_attr=edge_attr, edge_index=edge_index)
    data.material_id = id
    
    return data

class MultimodalDatasetMP_CGCNN(MultimodalDatasetMP):
    def __init__(self, target_data, params, \
        atom_feat_mode='original', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True
        self.atom_fea_original = atom_feat_mode == 'original'
        self.cg_exporter = ExportCrystalGraph(atom_feat_mode, max_num_nbr, radius, dmin, step)

        super().__init__(target_data)

    @property
    def processed_file_names(self):
        suf = "" if self.atom_fea_original else "_pn"
        if self.use_primitive:
            return f'processed_data_cgcnn{suf}.pt'
        else:
            return f'processed_data_convcell_cgcnn{suf}.pt'

    def process_input(self, material):
        return make_data(material, self.cg_exporter, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()


class RegressionDatasetMP_CGCNN(RegressionDatasetMP):
    def __init__(self, target_data, params, \
        atom_feat_mode='original', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True
        self.atom_fea_original = atom_feat_mode == 'original'
        self.cg_exporter = ExportCrystalGraph(atom_feat_mode, max_num_nbr, radius, dmin, step)

        super().__init__(target_data)

    @property
    def processed_file_names(self):
        suf = "" if self.atom_fea_original else "_pn"
        if self.use_primitive:
            return f'processed_data_cgcnn{suf}.pt'
        else:
            return f'processed_data_convcell_cgcnn{suf}.pt'

    def process_input(self, material):
        return make_data(material, self.cg_exporter, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()


class MultimodalDatasetCOD_CGCNN(MultimodalDatasetCOD):
    def __init__(self, target_data, params, \
        atom_feat_mode='original', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True
        self.atom_fea_original = atom_feat_mode == 'original'
        self.cg_exporter = ExportCrystalGraph(atom_feat_mode, max_num_nbr, radius, dmin, step)

        super().__init__(target_data)

    @property
    def processed_file_names(self):
        suf = "" if self.atom_fea_original else "_pn"
        if self.use_primitive:
            return f'processed_data_cgcnn{suf}.pt'
        else:
            return f'processed_data_convcell_cgcnn{suf}.pt'

    def process_input(self, material):
        return make_data(material, self.cg_exporter, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()