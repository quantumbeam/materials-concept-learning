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
        Initialize GaussianDistance class instance

        Parameters:
            dmin: float
                Minimum interatomic distance
            dmax: float
                Maximum interatomic distance
            step: float
                Step size for the Gaussian filter
            var: optional float
                A variance value for the Gaussian filter (default=step)
        
        Returns: None

        """

        # Ensure that min is less than max and max-min is greater than step.
        assert dmin < dmax
        assert dmax - dmin >= step 

        # Create a filter kernel.
        self.filter = np.arange(dmin, dmax+step, step) 

        # If no variance value was provided, set it to the step size.
        if var is None:
          var = step
        
        # Store the variance and kernel as instance variables.
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters:
            distances: np.array
                A distance matrix of any shape

        Returns:
            expanded_distance: np.array 
                Expanded distance matrix with the last dimension of length len(self.filter)

        """

        # Expand the dimensions of the distances array and interpolate it with the stored kernel
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2) 


class ExportCrystalGraph:
    def __init__(self, atom_feat_mode='original', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        """
        Initializes the object that generates a Crystal Graph from a crystal structure (pymatgen structure object).
        
        Parameters
        ----------
        atom_feat_mode: str
            Determines what is used as atom_fea (initial vector of each atom's features).
             - "original" uses atom_init.json (default for CGCNN)
             - "xyz_pos_and_elem_num" uses a one-hot vector of the atomic number along with the x, y, z Cartesian coordinates (normalized) (similar to PointNet)
        
        max_num_nbr: int
            Max number of neighbors allowed
        radius: float
            Maximum distance for finding neighbouring atoms
        dmin: float
            Minimum distance (used for GaussianDisatance)
        step: float
            Step size (used for GaussianDistance)

        """
        self.atom_feat_mode = atom_feat_mode
        self.ATOM_NUM_UPPER = 98
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.atom_features = np.array(atom_features, dtype=float)
       

    def extract_atom_feature(self, pymg_structure_obj):
        """
        Extracts features (cartesian coordinates and one-hot vector of atomic numbers) for each atom in the given PyMg crystal structure.

        Parameters
        ----------
        pymg_structure_obj: pymatgen.Structure
            The PyMatGen Structure object representing the crystal.
            

        Returns
        -------
        numpy.ndarray:
            Numpy array with feature vectors for each atom.
        """
       
        spe = generate_site_species_vector(pymg_structure_obj, self.ATOM_NUM_UPPER)
        xyz = np.vstack([[site.x, site.y, site.z] for site in pymg_structure_obj])
        pmin, pmax = xyz.min(axis=0, keepdims=True), xyz.max(axis=0, keepdims=True)
        xyz = xyz - (pmin + pmax) / 2
        scale = (pmax - pmin).max() / 2
        xyz = xyz / scale
        return np.concatenate([xyz, spe], axis=1)

    def get_crystal_graph_feature(self, pymg_structure_obj, material_id):
        """
        Generates the Crystal Graph feature for a crystal structure plotly.py

        Parameters
        ----------
        pymg_structure_obj: pymatgen.Structure
            The PyMatGen Structure object representing the crystal.
        material_id: str
            Integer padding of six zeros for Material Project IDs or internal MPIDs
            
        Returns
        -------
        tuple:
            
            1. torch.tensor (size = [num_atoms, num_features]) representing the features (φa) of each atom
            2. torch.tensor representing the one-hot encoded neighboring atoms (φbr)
            3. torch.Longtensor representing the indices of the neighboring atoms relative to the atom with φa
        """
        # generate crystal graph features
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
    # Get the structure of the material, check first for final_structure, then for structure
    if "final_structure" in material:
        structure = material['final_structure']
    elif "structure" in material:
        structure = material['structure']
    else:
        raise AttributeError("Material has no structure!")
    
    # If not primitive, convert the structure to conventional standard
    if not primitive:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    
    # Get crystal graph features of the structure
    if "material_id" in material:
        id = material['material_id']
    else:
        id = material['file_id']        
    atom_fea, nbr_fea, nbr_fea_idx = exporter.get_crystal_graph_feature(structure, id)

    # Reshape edge information to create a one-dimensional tensor of edge attributes
    N, M, dim = nbr_fea.shape
    edge_attr = nbr_fea.reshape(N*M, dim)
    
    # Reshape neighbor indexing information to be two-dimensional
    cols = nbr_fea_idx.reshape(N*M)
    rows = torch.arange(N, dtype=torch.long)
    rows = rows[:, None].expand((N, M)).reshape(N*M)
    edge_index = torch.stack((rows, cols), dim=0)

    # Return data object containing atom and edge feature tensors
    data = Data(x=atom_fea, edge_attr=edge_attr, edge_index=edge_index)
    data.material_id = id
    
    return data


class MultimodalDatasetMP_CGCNN(MultimodalDatasetMP):
    def __init__(self, target_data, params, \
        atom_feat_mode='original', max_num_nbr=12, radius=8, dmin=0, step=0.2):
        # Set parameters for crystal graph feature extraction
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True
        self.atom_fea_original = atom_feat_mode == 'original'
        self.cg_exporter = ExportCrystalGraph(atom_feat_mode, max_num_nbr, radius, dmin, step)
        # Initialize the class by calling the constructor of the parent class
        super().__init__(target_data)

    @property
    def processed_file_names(self):
        """
        Set the names of the processed files
        """
        suf = "" if self.atom_fea_original else "_pn"
        if self.use_primitive:
            return f'processed_data_cgcnn{suf}.pt'
        else:
            return f'processed_data_convcell_cgcnn{suf}.pt'

    def process_input(self, material):
        """
        Process each input to generate the data samples
        """
        return make_data(material, self.cg_exporter, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture the inherited functions.
    # So, here explicitly claim the process and download functions
    def process(self):
        """
        Process the whole dataset, i.e., apply `process_input()` to each instance of the dataset
        """
        super().process()
    
    def download(self):
        """
        Download the dataset
        """
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