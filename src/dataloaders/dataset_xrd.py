import os
import pandas as pd
import torch
import numpy
from torch_geometric.data import Data
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from dataloaders.common import RegressionDatasetMP
from dataloaders.common import try_to_get_xrd


class RegressionDatasetMP_XRD(RegressionDatasetMP):
    def __init__(self, target_data, params):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True

        super().__init__(target_data)
    
    @property
    def processed_file_names(self):
        if self.use_primitive:
            return 'processed_data_xrd.pt'
        else:
            return 'processed_data_convcell_xrd.pt'

    def process_input(self, material):
        structure = material['final_structure']
        if "material_id" in material:
            id = material['material_id']
        else:
            id = material['file_id']
            
        if not self.use_primitive:
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        if len(structure.cart_coords) == 1:
            return None

        data = Data()
        data.xrd = try_to_get_xrd(material)
        data.xrd = torch.tensor(data.xrd, dtype=torch.float)[None, None]
        data.material_id = id
        return data

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()
