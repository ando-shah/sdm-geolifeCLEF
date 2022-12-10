from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr


class Zarr_Dataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.

    Parameters
    ----------
    filepath : string 
         pass in an S3 dataset or local path to .zarr file with raster data
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    si_transform, env_transform : callable (optional)
        A function/transform that takes a tensor and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    verbose:
        whether to print verbose logs or not
    """

    def __init__(
        self,
        filepath: str,
        subset: str="train",
        patch_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        verbose: bool=False
    ):
        self.filepath = filepath
        self.subset = subset
        self.patch_transform = patch_transform
        self.target_transform = target_transform
        self.verbose = verbose

        possible_subsets = ["train", "val",  "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )
            
        if self.subset == "train":
            self.training_data = True
        else:
            self.training_data = False
        
        self.data = zarr.open(self.filepath, mode='r')
        
    def __getitem__(
        self,
        index: int) :
        
        #Extract Si patch first
        if self.verbose:
            print("[Extracting Patches]:", index)
            
        patch = self.data[index][0]
        target = self.data[index][1]
        
        patch = torch.tensor(patch, dtype=torch.float32)
        target = torch.tensor(target/255., dtype=torch.float32)
        
        if self.patch_transform:
            patch = self.patch_transform(patch)
            
        if self.target_transform:
            target = self.target_transform(target)
            
        # if self.verbose:
        print("[Completed Extraction]:", index)
            
        return patch, target
    
    def __len__(self) -> int:
        # return 32
        return self.data.shape[0]
        
        
        