import os
import pandas as pd
import torch
import numpy
import pymatgen
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def generate_site_species_vector(structure: pymatgen.core.structure.Structure, ATOM_NUM_UPPER):

    if hasattr(structure, 'species'):
        atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        atom_num = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(-1)
        x_species_vector = torch.eye(ATOM_NUM_UPPER)[atom_num - 1].squeeze()

    else:
        x_species_vector = []
        for site in structure.species_and_occu:
            site_species_and_occupancy = []
            # For each element at the site, get one-hot encoding and multiply the site occupancy to calculate the element occupancy vector.
            for elem in site.elements:
                if type(elem) == pymatgen.core.Element:
                    occupancy = site.element_composition[elem]
                elif type(elem) == pymatgen.core.periodic_table.Specie or type(elem) == pymatgen.core.periodic_table.Species:
                    occupancy = site.element_composition[elem.element]
                elif type(elem) == pymatgen.core.composition.Composition:
                    occupancy = site.element_composition[elem.element]
                    # print(elem, occupancy)
                elif type(elem) == pymatgen.core.periodic_table.DummySpecie or type(elem) == pymatgen.core.periodic_table.DummySpecies:
                    raise ValueError(f'Unsupported specie: {site}! Skipped')
                else:
                    print(site, type(site))
                    raise AttributeError
                atom_num = torch.tensor(elem.Z, dtype=torch.long)
                elem_onehot = torch.eye(ATOM_NUM_UPPER)[atom_num - 1]
                site_species_and_occupancy.append(elem_onehot*occupancy)
            # Sum of one-hot vector for each element at the site and convert to site occupancy
            site_species_and_occupancy_sum = torch.stack(site_species_and_occupancy).sum(0)
            x_species_vector.append(site_species_and_occupancy_sum)
        x_species_vector = torch.stack(x_species_vector, 0)
        
    return x_species_vector

def exclude_one_atom_crystal(data):
    # Set the default n > 1. This is to ensure that
    # when data has neither pos nor x (eg, xrd data)
    # the code returns True (ie, not exclude).
    n = 2
    if hasattr(data, 'pos') and data.pos is not None:
        n = data.pos.shape[0]
    elif hasattr(data, 'x') and data.x is not None:
        n = data.x.shape[0]

    if n > 1:
        return True

    return False

def try_to_get_xrd(material):
    if 'xrd_hist' in material:
        return material['xrd_hist']

    c = XRDCalculator()
    structure = material['final_structure']
    two_theta_range = (10, 110)
    diff_peaks = c.get_pattern(structure, two_theta_range=two_theta_range, scaled=False)
    xrd_hist = numpy.histogram(diff_peaks.x, bins=5000, range=two_theta_range,
                            weights=diff_peaks.y)

    return xrd_hist[0]


class MultimodalDatasetMP(InMemoryDataset):
    def __init__(self, target_data):
        self.ATOM_NUM_UPPER = 98
        root = "data/less_than_quinary20200608_with_xrd_10_110/" + target_data
# 
        urls = {'train':'https://ndownloader.figshare.com/files/38534981',
                'val':'https://ndownloader.figshare.com/files/38534975',
                'test':'https://ndownloader.figshare.com/files/38534984',
                'train_on_all':'https://ndownloader.figshare.com/files/38534972'}
        self.url = urls[target_data]
        super(MultimodalDatasetMP, self).__init__(root, pre_filter=exclude_one_atom_crystal)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'less_than_quinary20200608_with_xrd_10_110.pkl'

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def download(self):
        """
        Downloading dataset and extract on raw/
        """
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        
    def process_input(self, material):
        raise NotImplementedError()

    def process(self):
        crystals = pd.read_pickle(self.raw_paths[0])
        print('loaded data: ', self.raw_paths[0])

        data_list = []
        for material in crystals:
            data = self.process_input(material)
            data.xrd = try_to_get_xrd(material)
            data.xrd = torch.tensor(data.xrd, dtype=torch.float)[None, None]
            if data is None:
                continue
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class RegressionDatasetMP(InMemoryDataset):
    """
    PyTorch Geometric (PyG) Dataset class
    - Generates a point cloud dataset for PyG from a pymatgen structure object exported from MaterialsProject.
    - Generate a point cloud dataset that can be handled by PyG.
    """

    def __init__(self, target_data):
        root = "data/less_than_quinary20210610/" + target_data

        self.ATOM_NUM_UPPER = 98
        urls = {'train':'https://ndownloader.figshare.com/files/38883888',
                'val':'https://ndownloader.figshare.com/files/38883885',
                'test':'https://ndownloader.figshare.com/files/38883882'}
        self.url = urls[target_data]
        
        self.load_filename = 'less_than_quinary_asof2021_06_10_with_xrd.pkl'

        super().__init__(root, pre_filter=exclude_one_atom_crystal)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'less_than_quinary_asof2021_06_10_with_xrd.pkl'

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def download(self):
        """
        Downloading dataset and extract on raw/
        """
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        
    def process_input(self, material):
        raise NotImplementedError()

    def process(self):
        print(f'loaded data: {self.raw_paths[0]}')
        crystals = pd.read_pickle(self.raw_paths[0])
        
        data_list = []
        for material in tqdm(crystals):
            try:
                assert material['pretty_formula'] is not None
                assert material['band_gap'] is not None
                assert material['formation_energy_per_atom'] is not None
                assert material['e_above_hull'] is not None, f'Warning! Value "e_above_hull" is None in {material["material_id"]}. skipped'
                assert material['total_magnetization'] is not None
                assert material['energy_per_atom'] is not None
                assert material['energy'] is not None
                assert material['density'] is not None
                assert material['volume'] is not None
                assert material['spacegroup.number'] is not None
                
                data = self.process_input(material)
                if data is None:
                    continue

                data.material_id = material['material_id']
                data.pretty_formula = material['pretty_formula']
                data.bandgap = material['band_gap']
                data.formation_energy_per_atom = material['formation_energy_per_atom']
                data.e_above_hull = material['e_above_hull']
                data.total_magnetization = material['total_magnetization']
                data.energy_per_atom = material['energy_per_atom']
                data.energy = material['energy']
                data.density = material['density']
                data.volume = material['volume']
                data.sgr_class = material['spacegroup.number']

                data_list.append(data)
            except AssertionError as e:
                print(e)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
class MultimodalDatasetCOD(InMemoryDataset):
    def __init__(self, target_data):
        self.ATOM_NUM_UPPER = 98
        # target_data: "cif_sc", "cif_NOTsc", "cif_thermoelectric", "cif_NOTthermoelectric"
        root = "data/cod/" + target_data

        urls = {'sc':'https://ndownloader.figshare.com/files/38883474',
                'NOTsc':'https://ndownloader.figshare.com/files/38883483',
                'thermoelectric':'https://ndownloader.figshare.com/files/38883480',
                'NOTthermoelectric':'https://ndownloader.figshare.com/files/38883477'
                }
        self.url = urls[target_data]
        self.target_data = target_data
        super(MultimodalDatasetCOD, self).__init__(root, pre_filter=exclude_one_atom_crystal)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'{self.url.split("/")[-1]}'

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def download(self):
        path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.unlink(path)
        
    def process_input(self, material):
        raise NotImplementedError()

    def process(self):
        crystals = pd.read_pickle(self.raw_paths[0])
        print('loaded data: ', self.raw_paths[0])

        data_list = []
        for material in tqdm(crystals):
            try:
                assert material['file_id'] is not None
                assert material['formula'] is not None
                assert material['title'] is not None
                assert material['journal'] is not None
                assert material['year'] is not None
                assert material['svnrevision'] is not None
                
                data = self.process_input(material)
                if data is None:
                    continue
                data.material_id = material['file_id']
                data.pretty_formula = material['formula']
                data.title = material['title']
                data.journal = material['journal']
                data.title = material['title']
                data.year = material['year']
                data.svnrevision = material['svnrevision']
                # set a dummy XRD pattern for compatibility
                data.xrd = torch.zeros(5000)
                data_list.append(data)
            except AssertionError as e:
                print(e)
                print(f"material id: {material['file_id']}")
            except AttributeError as e:
                print(e)
                print(f"material id: {material['file_id']}")
            except IndexError as e:
                print(e)
                print(f"material id: {material['file_id']}")
            except ValueError as e:
                print(e)                
                print(f"material id: {material['file_id']}")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])