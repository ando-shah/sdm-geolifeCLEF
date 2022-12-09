from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING


from environmental_raster import EnvPatchExtractor
from si_raster import SIPatchExtractor

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.

    Parameters
    ----------
    raster_root : string or pathlib.Path
        Root directory of raster datasets -> pass in an S3 dataset or local
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    observations : GeoDataFrame
        for train mode: list of grids with observations with a KDE applied typically -> {geometry, grid_id, prob_*}
        fot val/train mode: list of observations -> {geometry, observation_id}
    centroids : GeoDataFrame 
        list of centroids for each grid cell in the train dataset: has a lon, lat and url fields
        for train_mode : centroid of grids
        for test/val : location of observations
        index of observations and centroids need to match 1:1
    side_len_m : int
        length in meters of each side of extracted raster; both SI and env
    side_px : int
        size in pixels of extracted patch
    si_patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'nir'
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    si_patch_extractor : SIPatchExtractor object 
        Patch extractor for satellite images (sentinel).
    env_patch_extractor : EnvPatchExtractor object 
        Patch extractor for environmental covariate rasters.
    si_transform, env_transform : callable (optional)
        A function/transform that takes a tensor and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        raster_root: Optional[str],
        subset: str,
        centroids: gpd.GeoDataFrame,
        observations: gpd.GeoDataFrame,
        side_len_m: int=10000,
        side_px: int=64,
        region: str = "us",
        si_patch_data: str = "all",
        use_rasters: bool = True,
        si_patch_extractor: SIPatchExtractor = None,
        env_patch_extractor: EnvPatchExtractor = None,
        si_transform: Optional[Callable] = None,
        env_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.raster_root = raster_root
        self.subset = subset
        self.si_patch_data = si_patch_data
        self.si_transform = si_transform
        self.env_transform = env_transform
        self.target_transform = target_transform
        
        self.centroids = centroids
        self.observations = observations
        
        self.si_patch_extractor = si_patch_extractor
        self.env_patch_extractor = env_patch_extractor
        
        self.side_len_m = side_len_m
        self.side_px = side_px

        possible_subsets = ["train", "val",  "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        if self.subset == "train":
            # subset_file_suffix = "test_val"
            self.training_data = True
        else:
            # subset_file_suffix = "train"
            self.training_data = False

        #Setup Patch Extractors
        if (self.si_patch_extractor == None):
            print("Setting up SI Patch Extractor..")
            self.si_patch_extractor = SIPatchExtractor(self.centroids, side_length_m=self.side_len_m, side_px=self.side_px)
        
        if (self.env_patch_extractor == None):
            print("Setting up env raster extractor..")
            self.env_patch_extractor = EnvPatchExtractor(self.raster_root, side_len_m=self.side_len_m, norm="std")
            # self.env_patch_extractor.append("bio_1")
            # self.env_patch_extractor.add_all_pedologic_rasters()
            self.env_patch_extractor.add_all_rasters()
            # self.env_patch_extractor.add_all_bioclimatic_rasters()
            
        if self.training_data:
            # self.grid_ids = self.centroids.index
            # self.coordinates = self.centroids[["lon", "lat"]].values
            #convert to numpy (get rid of geometry and grid_id columns; grid_id is same as index), requantize to 0 to 1.
            self.targets = self.observations.drop(['geometry'], axis=1).set_index('grid_id', drop=True).to_numpy()/255.

        else: #for val
            self.targets = self.observations.drop(['geometry'], axis=1).to_numpy()

        # FIXME: add back landcover one hot encoding?
        # self.one_hot_size = 34
        # self.one_hot = np.eye(self.one_hot_size)


    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(
        self,
        index: int,
    ) :
        
        #Extract Si patch first
        print("Extracting SI patches...")
        t_si, aoi_si, coods = self.si_patch_extractor[index]
        print("Extracting env patches...")
        t_env = self.env_patch_extractor[(coods, aoi_si)]
       
        # FIXME: add back landcover one hot encoding?
        # lc = patches[3]
        # lc_one_hot = np.zeros((self.one_hot_size,lc.shape[0], lc.shape[1]))
        # row_index = np.arange(lc.shape[0]).reshape(lc.shape[0], 1)
        # col_index = np.tile(np.arange(lc.shape[1]), (lc.shape[0], 1))
        # lc_one_hot[lc, row_index, col_index] = 1


        print("Pre-SI transform[{}]: {:.4}/{:.4}".format(t_si.shape, t_si.max(), t_si.min()))
        if self.si_transform:
            # t_si_rgb = self.si_transform(t_si[0:3])
            # t_si_nir = self.si_transform(t_si[3:])
            # t_si = torch.cat((t_si_rgb, t_si_nir),0)
            t_si = self.si_transform(t_si)
            
        print("Post-SI transform[{}]: {:.4}/{:.4}".format(t_si.shape, t_si.max(), t_si.min()))
            
        if self.env_transform:
            t_env = self.env_transform(t_env)
            
        #combine the patch tensors
        patch =  torch.cat((t_si, t_env), 0)
        
        print(patch.shape)

        if self.training_data:
            target = self.targets[index]
            #convert to tensor
            target = torch.from_numpy(target)

            if self.target_transform:
                target = self.target_transform(target)

            return patch, target
        else:
            return patch, target
