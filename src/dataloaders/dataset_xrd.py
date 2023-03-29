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
    """
        A custom dataset class for regression with X-Ray Diffraction data.

        Parameters:
            target_data (Dataframe): The input target data containing structures and material properties.
            params (object): Object containig parameters needed to preprocess the data.
        
    """
    def __init__(self, target_data, params):
        # check if use_primitive parameter is present in params object, otherwise default True
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True

        super().__init__(target_data)

    @property
    def processed_file_names(self):
        # Return the processed file name based on whether primitive cells were used or not
        if self.use_primitive:
            return 'processed_data_xrd.pt'
        else:
            return 'processed_data_convcell_xrd.pt'

    def process_input(self, material):
        # Prepare input data for this particular material
        structure = material['final_structure']
        if "material_id" in material:
            id = material['material_id']
        else:
            id = material['file_id']

        # Convert to conventional standard structure if necessary
        if not self.use_primitive:
            structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        # Check there are at least two atoms in the structure, otherwise skip this material
        if len(structure.cart_coords) == 1:
            return None

        # Construct the input data as a PyTorch geometric data object
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
