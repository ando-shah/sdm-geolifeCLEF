from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr
from numcodecs import blosc

blosc.use_threads = False

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
    persist:
        whether to store it in memory or keep on disk
        True: load to memory
    """

    def __init__(
        self,
        filepath_f: str=None,
        filepath_l: str=None,
        subset: str="train",
        patch_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        persist: bool=False,
        verbose: bool=False
    ):
        self.filepath_f = filepath_f
        self.filepath_l = filepath_l
        self.subset = subset
        self.patch_transform = patch_transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.persist = persist
        
        blosc.use_threads = False

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
        
        if not self.persist:
            if self.filepath_f:
                self.features = zarr.open(self.filepath_f, mode='r',synchronizer=zarr.ThreadSynchronizer())
            if self.filepath_l:
                self.labels = zarr.open(self.filepath_l, mode='r',synchronizer=zarr.ThreadSynchronizer())
            
        else:
            if self.filepath_f:
                self.features = zarr.load(self.filepath_f)
            if self.filepath_l:
                self.labels   = zarr.load(self.filepath_l)
            
        
    def __getitem__(
        self,
        index: int) :
        
        #Extract Si patch first
        if self.verbose:
            print("[Extracting Patches]:", index)
            
        patch = self.features[index]
        target = self.labels[index]
        
        patch = torch.tensor(patch, dtype=torch.float32)
        target = torch.tensor(target/255., dtype=torch.float32)
        
        if self.patch_transform:
            patch = self.patch_transform(patch)
            
        if self.target_transform:
            target = self.target_transform(target)
            
        # if self.verbose:
        # print("[Completed Extraction]:", index)
            
        return patch, target
    
    def __len__(self) -> int:
        # return 32
        return self.features.shape[0]
    
    def info(self):
        print ("Features:\n ", self.features.info)
        print ("Labels:\n ", self.labels.info)
        
        # return (self.features.info, self.labels.info)
    
    
    
class Zarr_GroupDataset(Dataset):
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
    persist:
        whether to store it in memory or keep on disk
        True: load to memory
    """

    def __init__(
        self,
        filepath: str,
        subset: str="train",
        patch_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        persist: bool=False,
        verbose: bool=False
    ):
        self.filepath = filepath
        self.subset = subset
        self.patch_transform = patch_transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.persist = persist
        

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
            
        
        self.store = zarr.DirectoryStore(filepath)
        
        if not self.persist:
            self.data = zarr.group(store=self.store, overwrite=False)
            # self.data = zarr.open(self.filepath, mode='r',synchronizer=zarr.ThreadSynchronizer())
        else:
            self.data = zarr.load(self.filepath)
            
        self.size = self.data.info_items()[5][1]
        print ("Initialized with size = ", self.size)
        
    def __getitem__(
        self,
        index: int) :
        
        #Extract Si patch first
        if self.verbose:
            print("[Extracting Patches]:", index)
            
        patch = np.array(self.data[index])
        # target = self.data[index][1]
        
        patch = torch.tensor(patch, dtype=torch.float32)
        # target = torch.tensor(target/255., dtype=torch.float32)
        
        if self.patch_transform:
            patch = self.patch_transform(patch)
            
#         if self.target_transform:
#             target = self.target_transform(target)
            
        # if self.verbose:
        # print("[Completed Extraction]:", index)
            
        return patch#, target
    
    def __len__(self) -> int:
        # returns the number of arrays in dataset
        return self.size
        
        
        