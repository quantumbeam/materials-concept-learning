import os
import pandas as pd
import torch
import numpy
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure


def structure2tensor(structure):
    """
    Takes a pymatgen structure object and y (e.g. label) and return a PyG data object.

    Return
    atom_pos: Tensor
        3D coordinates of atoms in a crystal
    x_el: Tensor
        One-hot vector of atomic number of each point (=atom) in the crystal
    """
    atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    x_el = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(-1)
    return atom_pos, x_el

def exclude_one_atom_crystal(data):
    """
    Crystal structures with only one atom in the unit cell are excluded to prevent point cloud scaling failure.
    """
    n_atom_in_crystal = data.pos.shape[0]
    if n_atom_in_crystal > 1:
        return True
    else:
        return False

class PyMgStructureMP(InMemoryDataset):
    """
    PyTorch Geometric (PyG) Dataset class
    - Generates a point cloud dataset for PyG from a pymatgen structure object exported from MaterialsProject.
    - Generate a point cloud dataset that can be handled by PyG.
    """

    def __init__(self, target_data, params, transform=None, pre_transform=None, pre_filter=exclude_one_atom_crystal):
        self.ATOM_NUM_UPPER = 98
        root = "data/less_than_quinary20200608_with_xrd_10_110/" + target_data
        urls = {'train':'https://multimodal.blob.core.windows.net/train/less_than_quinary20200608_with_xrd_10_110.zip',
                'val':'https://multimodal.blob.core.windows.net/val/less_than_quinary20200608_with_xrd_10_110.zip',
                'test':'https://multimodal.blob.core.windows.net/test/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_on_all':'https://multimodal.blob.core.windows.net/all/less_than_quinary20200608_with_xrd_10_110.zip'}
        self.url = urls[target_data]
        super(PyMgStructureMP, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'less_than_quinary20200608_with_xrd_10_110.pkl'

    @property
    def processed_file_names(self):
        return 'processed_data_full.pt'

    def download(self):
        """
        Downloading dataset and extract on raw/
        """
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        
    def process(self):
        crystals = pd.read_pickle(self.raw_paths[0])
        print('loaded data: ', self.raw_paths[0])

        data_list = []
        one_hots = torch.eye(self.ATOM_NUM_UPPER)
        for material in crystals:
            structure = material['final_structure']
            
            atom_pos = torch.tensor(structure.frac_coords, dtype=torch.float)
            atom_num = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(-1)
            
            x_atom_onehot = one_hots[atom_num - 1] # -1 since atom_num is starting with 1.
            data = Data(x=x_atom_onehot, y=None, pos=atom_pos)
            data.material_id = material['material_id']
            data.xrd = torch.tensor(material['xrd_hist'], dtype=torch.float)[None, None]
            data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float)[None]

            data.energy = material['energy']
            data.density = material['density']
            data.bandgap = material['band_gap']
            data.spacegroup = material['spacegroup.number']
            data.energy_per_atom = material['energy_per_atom']
            data.total_magnetization = material['total_magnetization']
            data.formation_energy_per_atom = material['formation_energy_per_atom']

            # Exclude invalid entries.
            # mp-1245128 (MgO) in test data has a None value.
            if None in (
                data.energy,
                data.density,
                data.bandgap,
                data.spacegroup,
                data.energy_per_atom,
                data.total_magnetization,
                data.formation_energy_per_atom):
                print(f"Skipped: {material['material_id']} ({material['pretty_formula']})")
                continue

            data.energy = torch.tensor(data.energy, dtype=torch.float)[None]
            data.density = torch.tensor(data.density, dtype=torch.float)[None]
            data.bandgap = torch.tensor(data.bandgap, dtype=torch.float)[None]
            data.spacegroup = torch.tensor(data.spacegroup, dtype=torch.int32)[None]
            data.energy_per_atom = torch.tensor(data.energy_per_atom, dtype=torch.float)[None]
            data.total_magnetization = torch.tensor(data.total_magnetization, dtype=torch.float)[None]
            data.formation_energy_per_atom = torch.tensor(data.formation_energy_per_atom, dtype=torch.float)[None]

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])