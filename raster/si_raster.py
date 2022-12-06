import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
# from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import rasterio
import rasterio.warp
import rasterio.mask
import shapely.geometry
import geopandas as gpd
import dask_geopandas
# from dask.distributed import Client
# from dask_gateway import GatewayCluster

import pystac_client
import planetary_computer as pc
from dask.distributed import Client
import rioxarray

# Train : centroids are always the centroid of the grids
# Test : centroids will be the points at which we will be sampling to test
class SIPatchExtractor(object):
    """
    Handles the loading and extraction of an environmental tensor from multiple rasters given GPS coordinates.
    """
    
    def __init__(self, data: gpd.GeoDataFrame, side_length_m:int=1000, side_px:int=64):
        """Constructor

        Parameters
        ----------
        centroids : a geopandas frame with 4 columns:
            grid_id, lons, lats, geometry (of the centroid)
            In Train mode, these coordinates are centroids,
            In Test mode, these would be the coordinates of observations
            
        side_length_m : integer
            Size in meters (side_length_m x side_length_m) of the patches to extract around each location.
            
        side_px : integer
            Size in pixels of the image patch output
        """
        
        self.side_length = side_length_m
        self.side_px = side_px
        self.data = data
    

    def clean(self) -> None:
        """Remove all rasters from the extractor."""
        # self.rasters_fr = []
        self.data = None



    def __repr__(self) -> str:
        return str(self)


    def __getitem__(self, idx:int=-1): #, coordinates: Coordinates) -> npt.NDArray[np.float32]:
        """Extracts the patches around the given GPS coordinates for all the previously loaded rasters.

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)
        idx : int containing the grid id of the patch in question
            **Used only in training mode**
            **-1 for test mode**
        
        Returns
        -------
        patch : 3d array of floats, [n_rasters, size, size]
            Extracted patches around the given coordinates.
        """


        if (idx > -1):
            lon, lat = self.data.lon[idx], self.data.lat[idx]
            fn_RGB = self.data.urls[idx][0]
            fn_NIR = self.data.urls[idx][1]
        ##TODO
        else: #Test mode where a gps coordinates are entered (prob not required)
            idx = 0 # do something dumb for now
            
            
            
        mask_geom = None
        t_rgb = None
        t_nir = None

        if fn_RGB is None or fn_NIR is None:
            print("URLS are messed up. Aborting")
            return None
        else:
            point_geom = shapely.geometry.mapping(shapely.geometry.Point(lon, lat))

            #RGB
            
            with rasterio.Env():
                with rasterio.open(fn_RGB, "r") as f:
                    # Grid points are in 4326 -> move to 32610
                    point_geom = rasterio.warp.transform_geom("epsg:4326", f.crs.to_string(), point_geom)
                    
                    #Convert the point to a shape
                    point_shape = shapely.geometry.shape(point_geom)
                    #Create a square out of it with side_length = buffer*2
                    mask_shape = point_shape.buffer(self.side_length/2).envelope
                    mask_geom = shapely.geometry.mapping(mask_shape)

                    try:

                        image_rgb = rioxarray.open_rasterio(f).rio.clip([mask_geom], from_disk=True)
                        #TODO Convert to EA projection here:

                        t = torch.from_numpy(image_rgb.values.astype(np.uint8))

                        postprocess = T.Compose ([T.CenterCrop((t.shape[1])),
                                                 T.Resize(self.side_px)])

                        t_rgb = postprocess(t)


                    except ValueError as e:
                        if "Input shapes do not overlap raster." in str(e):
                            print("Couldnt open RGB URL or requested grid doesnt overlap")
                            return None
            
            
            #NIR
            with rasterio.Env():
                with rasterio.open(fn_NIR, "r") as f:

                    try:

                        image_nir = rioxarray.open_rasterio(f).rio.clip([mask_geom], from_disk=True)
                        t = torch.from_numpy(image_nir.values.astype(np.uint8)) #divides by 255 effectively

                        postprocess = T.Compose ([T.CenterCrop((t.shape[1])),
                                                 T.Resize(self.side_px)])

                        t_nir = postprocess(t)
                        #otherwise images dont display properly
                        #but uint8 conversion has already taken care of this downresolution
                        # t /= 255 
                        # print(t_nir.shape, type(t_nir))

                    except ValueError as e:
                        if "Input shapes do not overlap raster." in str(e):
                            print("Couldnt open NIR URL or requested grid doesnt overlap")
                            return None
            
            
                        
            #Combine the (3,W,H) and (1,W,H) tensors into (4,W,H)
            t_out = torch.cat((t_rgb, t_nir), 0)
            assert(t_out.shape == (4,self.side_px,self.side_px))
            # out_image = np.vstack((image_rgb, image_nir))
            # out_image = out_image / 255.0

            # print(t_out.shape)
            # assert(out_image.shape )
            
            return t_out

    def __len__(self) -> int:
        """Number of variables/rasters loaded.

        Returns
        -------
        n_rasters : integer
            Number of loaded rasters
        """
        return len(self.data)


    def plot(
        self,
        idx: int):
        """Plot the SI-image tensor (RGB Only)

        Parameters
        ----------
        index

        Returns
        -------
        PIL Image
        """

        tensor = self[idx]
        
        transform = T.ToPILImage()
        image_RGB = transform(tensor[0:3,:,:])
        image_NIR = transform(tensor[3,:,:])
        image_RGB.show()
        image_NIR.show()
        # return (image_RGB, image_NIR)