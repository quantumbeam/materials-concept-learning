import torch

from torch_geometric.data import Data
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dataloaders.common import MultimodalDatasetMP, RegressionDatasetMP
from dataloaders.common import generate_site_species_vector

def make_data(material, ATOM_NUM_UPPER, primitive):
    structure = material['final_structure']
    if not primitive:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

    if "material_id" in material:
        id = material['material_id']
    else:
        id = material['file_id']

    atom_pos = torch.tensor(structure.frac_coords, dtype=torch.float)
    atom_fea = generate_site_species_vector(structure, ATOM_NUM_UPPER)
    data = Data(x=atom_fea, y=None, pos=atom_pos)
    data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float)[None]
    data.material_id = id
    return data

class MultimodalDatasetMP_Periodic(MultimodalDatasetMP):
    def __init__(self, target_data, params):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True

        super(MultimodalDatasetMP_Periodic, self).__init__(target_data)
    
    @property
    def processed_file_names(self):
        if self.use_primitive:
            return 'processed_data_periodic.pt'
        else:
            return 'processed_data_convcell_periodic.pt'

    def process_input(self, material):
        return make_data(material, self.ATOM_NUM_UPPER, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()


class RegressionDatasetMP_Periodic(RegressionDatasetMP):
    def __init__(self, target_data, params):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True

        super(RegressionDatasetMP_Periodic, self).__init__(target_data)
    
    @property
    def processed_file_names(self):
        if self.use_primitive:
            return 'processed_data_periodic.pt'
        else:
            return 'processed_data_convcell_periodic.pt'

    def process_input(self, material):
        return make_data(material, self.ATOM_NUM_UPPER, self.use_primitive)
        
    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()
