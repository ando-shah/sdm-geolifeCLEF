import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
# from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import rasterio
import rasterio.warp
import rasterio.mask
from rasterio.enums import Resampling

import shapely.geometry
import geopandas as gpd
import dask_geopandas


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
        self.data = data.copy()
    

    def clean(self) -> None:
        """Remove all rasters from the extractor."""
        # self.rasters_fr = []
        self.data = None



    def __repr__(self) -> str:
        return str(self)
    
    def getlonlat(self, idx):
        """returns a tuple consisting of the longitude and latitude for that given grid_id or idx
        """
        
        return (self.data.lon[idx], self.data.lat[idx])


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
                aspect_ratio, new_height, new_width = -1., -1., -1.
                
                with rasterio.open(fn_RGB, "r") as f:
                    # Grid points are in 4326 -> move to 32610
                    
                    
                    point_geom = rasterio.warp.transform_geom("epsg:4326", f.crs.to_string(), point_geom)
                    
                    #Convert the point to a shape
                    point_shape = shapely.geometry.shape(point_geom)
                    #Create a square out of it with side_length = buffer*2
                    mask_shape = point_shape.buffer(self.side_length/2).envelope
                    mask_geom = shapely.geometry.mapping(mask_shape)

                    try:
                        
                        
                        aoi_si = rioxarray.open_rasterio(f).rio.clip([mask_geom], from_disk=True)
                        
                        aspect_ratio = aoi_si.shape[2]/aoi_si.shape[1]
                        new_height = self.side_px
                        new_width = aspect_ratio * new_height
                        # print(new_height, new_width)

                        #downsample raster                                    # (height, width)
                        aoi_si = aoi_si.rio.reproject(aoi_si.rio.crs, \
                                                      shape=(int(new_height), int(new_width)), resampling=Resampling.bilinear)
                        
                        crop_px = aoi_si.shape[1]
                        # print(aoi_si.shape, aoi_si.values.astype(float).shape)
                        # print("RGB max = ", aoi_si.values.max())
                        
                        #dont convert to uint8, otherwise future transforms dont work -> needs to be float
                        # t_rgb = torch.from_numpy(aoi_si.values.astype(np.uint8)) 
                        #reduce bit resolution to 8 bit by dividing by 255 and convert to float (not uint8)
                        t_rgb = torch.tensor(aoi_si.values/255., dtype=float)
                        t_rgb = T.CenterCrop((crop_px))(t_rgb)
                        # postprocess = T.Compose ([T.ToTensor(),
                        #                           T.CenterCrop((crop_px))])
                                                 # T.Resize(self.side_px)]) TODO Put in normalize
                        # t_rgb = postprocess(aoi_si.values.astype(float))
                        # print("t_rgb:", t_rgb.shape)
                        # t_rgb = t_rgb/255 #convert to 8 bit


                    except ValueError as e:
                        if "Input shapes do not overlap raster." in str(e):
                            print("Couldnt open RGB URL or requested grid doesnt overlap")
                        return (None, None, self.getlonlat(idx))
            
            #NIR
            with rasterio.Env():
                with rasterio.open(fn_NIR, "r") as f:

                    try:

                        aoi_si_nir = rioxarray.open_rasterio(f).rio.clip([mask_geom], from_disk=True)
                        aoi_si_nir = aoi_si_nir.rio.reproject(aoi_si_nir.rio.crs, \
                                                      shape=(int(new_height), int(new_width)), resampling=Resampling.bilinear)
                        # print("NIR max = ", aoi_si_nir.values.max())
                        crop_px = aoi_si_nir.shape[1]
                    
                        #It seems that NIR is in the range 1-10000, not 1 to 256 like RGB!!
                        t_nir = torch.tensor(aoi_si_nir.values/10000., dtype=float)
                        t_nir = T.CenterCrop((crop_px))(t_nir)


#                         t_nir = torch.from_numpy(aoi_si_nir.values.astype(np.uint8))
#                         postprocess = T.Compose ([T.CenterCrop((t_nir.shape[1])),])
                        
#                         t_nir = postprocess(t_nir)
                        
                        #otherwise images dont display properly
                        #but uint8 conversion has already taken care of this downresolution
                        # t /= 255 
                        # print(t_nir.shape, type(t_nir))

                    except ValueError as e:
                        if "Input shapes do not overlap raster." in str(e):
                            print("Couldnt open NIR URL or requested grid doesnt overlap")
            
                        return (None, None, self.getlonlat(idx))
                        
            #Combine the (3,W,H) and (1,W,H) tensors into (4,W,H)
            # print(t_rgb.shape, t_nir.shape)
            t_out = torch.cat((t_rgb, t_nir), 0)
            assert(t_out.shape == (4,self.side_px,self.side_px))
            # out_image = np.vstack((image_rgb, image_nir))
            # out_image = out_image / 255.0

            # print(t_out.shape)
            # assert(out_image.shape )
            
            return (t_out, aoi_si, self.getlonlat(idx))
        

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

        tensor, _ = self[idx]
        
        transform = T.ToPILImage()
        image_RGB = transform(tensor[0:3,:,:])
        image_NIR = transform(tensor[3,:,:])
        image_RGB.show()
        image_NIR.show()
        # return (image_RGB, image_NIR)