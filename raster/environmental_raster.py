from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import rasterio

import torch
# import torchvision.transforms.functional as F
import torchvision.transforms as T

import rioxarray
import shapely.geometry

if TYPE_CHECKING:
    import numpy.typing as npt

    Coordinates = tuple[float, float]
    Patch = npt.NDArray[np.float32]


MIN_ALLOWED_VALUE = -10000

# fmt: off
bioclimatic_raster_names = [
    "bio_1", "bio_2", "bio_3", "bio_4", "bio_5", "bio_6", "bio_7", "bio_8", "bio_9",
    "bio_10", "bio_11", "bio_12", "bio_13", "bio_14", "bio_15", "bio_16", "bio_17",
    "bio_18", "bio_19"
]

pedologic_raster_names = [
    "bdticm", "bldfie", "cecsol", "clyppt", "orcdrc", "phihox", "sltppt", "sndppt"
]

raster_names = bioclimatic_raster_names + pedologic_raster_names
# fmt: on


# class Raster(object):
#     """
#     Handles the loading and the patch extraction for a single raster
#     """

#     def __init__(
#         self,
#         path: str,
#         name: str,
#         country: str = "USA",
#         side_len_m: int = 1000,
#         dst_crs: str = "EPSG:32610", #california centric UTM
#     ):
#         """Loads a GeoTIFF file containing an environmental raster

#         Parameters
#         ----------
#         path : string / pathlib.Path
#             Path to the folder containing all the rasters.
#         country : string, either "FR" or "USA"
#             Which country to load raster from.
#         size : integer
#             Size in pixels (size x size) of the patch to extract around each location.
#         nan : float or None
#             Value to use to replace missing data in original rasters, if None, leaves default values.
#         out_of_bounds : string, either "error", "warn" or "ignore"
#             If "error", raises an exception if the location requested is out of bounds of the rasters. Set to "warn" to only produces a warning and to "ignore" to silently ignore it and return a patch filled with missing data.
#         """
#         # path = Path(path)
#         # if not path.exists():
#         #     raise ValueError(
#         #         "path should be the path to a raster, given non-existant path: {}".format(
#         #             path
#         #         )
#         #     )

#         self.path = path 
#         self.name = name
#         self.raster = None
        
#         filename = "{:}/{:}_{:}.tif".format(self.path,self.name,country)
    
        
#         with rasterio.Env():
#             with rasterio.open(filename, "r") as dataset:
#                 with rasterio.vrt.WarpedVRT(dataset, crs=dst_crs) as src:
#                     # self.dataset = dataset
#                     raster = src.read(1, masked=True, out_dtype=src.dtypes[0])
#                     print(type(src))
                    
#                     self.x_min = src.bounds.left
#                     self.y_min = src.bounds.bottom
#                     self.x_resolution = src.res[0]
#                     self.y_resolution = src.res[1]
#                     self.n_rows = src.height
#                     self.n_cols = src.width
#                     self.crs = src.crs.data['init']

#                     # loading the raster
#                     # raster = np.squeeze(src.read())
#                     src.close()

#                     # raster = raster.astype(float)

#                     # value bellow min_value are considered incorrect and therefore no_data
#                     raster[raster < MIN_ALLOWED_VALUE] = None
#                     raster[np.isnan(raster)] = None
#                     #convert to 0-255, and then to uint8
#                     raster = (255 * (raster - np.nanmin(raster)) / (np.nanmax(raster) - np.nanmin(raster))).astype(np.uint8)
#                     self.raster = raster.filled(0)

#         # print(self.raster, self.raster.shape)
#         # setting the shape of the raster
#         self.shape = self.raster.shape
#         print("Completed Setup of Raster {:} Patch Size {:}x{:} in CRS = {:}".format(self.name, self.n_rows, self.n_cols, self.crs))



#     def _extract_patch(self, coordinates: Coordinates) -> Patch:
#         """Extracts the patch around the given GPS coordinates.
#         Avoid using this method directly.

#         Parameter
#         ----------
#         coordinates : tuple containing two floats
#             GPS coordinates (latitude, longitude)

#         Returns
#         -------
#         patch : 2d array of floats, [size, size], or 0d array with a single float if size == 1
#             Extracted patch around the given coordinates.
#         """
#         row, col = self.dataset.index(coordinates[1], coordinates[0])

#         if self.size == 1:
#             # Environmental vector
#             patch = self.raster[row, col]
#         else:
#             half_size = self.size // 2
#             height, width = self.shape

#             # FIXME: only way to trigger an exception? (slices don't)
#             self.raster[row, col]

#             raster_row_slice = slice(max(0, row - half_size), row + half_size)
#             raster_col_slice = slice(max(0, col - half_size), col + half_size)

#             patch_row_slice = slice(
#                 max(0, half_size - row), self.size - max(0, half_size - (height - row))
#             )
#             patch_col_slice = slice(
#                 max(0, half_size - col), self.size - max(0, half_size - (width - col))
#             )

#             patch = np.full(
#                 (self.size, self.size), fill_value=self.nan, dtype=np.float32
#             )
#             patch[patch_row_slice, patch_col_slice] = self.raster[
#                 raster_row_slice, raster_col_slice
#             ]

#         patch = patch[np.newaxis]
#         return patch
 
    
class Raster(object):
    """
    Handles the loading and the patch extraction for a single raster
    """

    def __init__(
        self,
        path: str,
        name: str,
        country: str = "USA",
        side_len_m: int = 1000,
        side_px : int = 64,
        dst_crs: str = "EPSG:32610", #california centric UTM
        display: bool = False
    ):
        """Loads a GeoTIFF file containing an environmental raster

        Parameters
        ----------
        path : string / pathlib.Path
            Path to the folder containing all the rasters.
        country : string, either "FR" or "USA"
            Which country to load raster from.
        display : bool
            Will the raster be displayed
        """
        
        self.path = path 
        self.name = name
        self.raster = None
        self.dst_crs = dst_crs
        self.side_length = side_len_m
        self.side_px = side_px
        
        filename = "{:}/{:}_{:}.tif".format(self.path,self.name,country)
        print("Opening Raster file for ", self.name)
        # print(filename, name)
    
        with rasterio.Env():
            with rasterio.open(filename, "r") as f:

                self.raster = rioxarray.open_rasterio(f, masked=True)#.rio.clip([mask_geom], from_disk=True)
                self.raster = self.raster.rio.reproject(self.dst_crs)
                
        # self.raster.data[self.raster.data < MIN_ALLOWED_VALUE] = None
        # self.raster.data[np.isnan(self.raster.data)] = None
        #convert to 0-255, and then to uint8
        self.raster.data = (255 * (self.raster.data - np.nanmin(self.raster.data)) / (np.nanmax(self.raster.data) - np.nanmin(self.raster.data)))#.astype(np.uint8) -> dont convert to int here otherwise rio.clip will not work
        # raster.data = raster.data.filled(0) -> might be useful for Mask values

        
        self.crs = self.raster.rio.crs

        # setting the shape of the raster
        self.shape = self.raster.shape
        print("Completed Setup of Raster {:} in CRS = {:} and dtype {:}".format(self.name, self.crs, self.raster.data.dtype))
        
        if(display):
            self.raster.sel(band=1).plot.imshow()
        
        
    #Rasterio implementation with the correct behavior
    # 1. Read raster in rasterio (in 4326)
    # 2. Convert raster to 32610
    # 3. Get coordinate, convert to 32610
    # 4. Get mask in 32610
    # 5. Clip raster with mask
    # 6. Transform with center crop and resize
    
    def _extract_patch(self, coordinates: Coordinates):
        """Extracts the patch around the given GPS coordinates.
        Avoid using this method directly.

        Parameter
        ----------
        coordinates : tuple containing two floats and a string
            GPS coordinates (longitude,latitude)
            ((lat,lon),crs)

        Returns
        -------
        patch : 2d array of floats, [size, size], or 0d array with a single float if size == 1
            Extracted patch around the given coordinates.
        """
        
        lon, lat, dst_crs = coordinates[0][0], coordinates[0][1], coordinates[1]
        
        if dst_crs == 'None':
            dst_crs = self.dst_crs
        
        dst_crs = rasterio.crs.CRS.from_string(dst_crs)
        
        print(dst_crs, lon, lat)

        point_geom = shapely.geometry.mapping(shapely.geometry.Point(lon, lat))

        # Grid points are in 4326 -> move to 32610
        point_geom = rasterio.warp.transform_geom("epsg:4326", dst_crs, point_geom)
        # point_geom = rasterio.warp.transform_geom(self.crs, dst_crs, point_geom)

        #Convert the point to a shape
        point_shape = shapely.geometry.shape(point_geom)
        #Create a square out of it with side_length = buffer*2
        mask_shape = point_shape.buffer(self.side_length/2).envelope
        mask_geom = shapely.geometry.mapping(mask_shape)
        
        # print(mask_geom)

        try:

            aoi = self.raster.rio.clip([mask_geom], from_disk=True)
            
            #TODO Convert to EA projection here:
            #Convert to uint8, since the values were already scaled to between 0 and 255 before
            t = torch.from_numpy(aoi.values.astype(np.uint8))

            postprocess = T.Compose ([T.CenterCrop((t.shape[1])),
                                     T.Resize(self.side_px)])

            t = postprocess(t)
            #otherwise images dont display properly
            #but uint8 conversion has already taken care of this downresolution
            # t /= 255 
            print(t.shape, type(t))

        except ValueError as e:
            if "No data found in bounds" in str(e):
                print("ERROR: grid doesnt overlap for ", self.name)
                return None
            
        return t

    def __len__(self) -> int:
        """Number of bands in the raster (should always be equal to 1).

        Returns
        -------
        n_bands : integer
            Number of bands in the raster
        """
        return len(self.raster.band)

    def __getitem__(self, coordinates):
        """Extracts the patch around the given GPS coordinates.

        Parameters
        ----------
        coordinates : tuple containing two floats and a string for destination crs
            GPS coordinates (latitude, longitude)
            dst_crs

        Returns
        -------
        patch : 2d array of floats, [size, size], or 0d array with a single float if size == 1
            Extracted patch around the given coordinates.
        """
        
        return self._extract_patch(coordinates)
        
#         try:
#             return self._extract_patch(coordinates)
#         except IndexError as e:
#             if self.out_of_bounds == "error":
#                 raise e
#             else:
#                 if self.out_of_bounds == "warn":
#                     warnings.warn(
#                         "GPS coordinates ({}, {}) out of bounds".format(*coordinates)
#                     )

#                 if self.size == 1:
#                     patch = np.array([self.nan], dtype=np.float32)
#                 else:
#                     patch = np.full(
#                         (1, self.size, self.size), fill_value=self.nan, dtype=np.float32
#                     )

#                 return patch

        # print ("Should not be here : Raster.__getitem__()")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "name: " + self.name + "\n"


class PatchExtractor(object):
    """
    Handles the loading and extraction of an environmental tensor from multiple rasters given GPS coordinates.
    """

    def __init__(self, root_path: str, side_len_m: int = 1000, side_px:int = 64):
        """Constructor

        Parameters
        ----------
        root_path : string or pathlib.Path
            Path to the folder containing all the rasters.
        side_len_m : integer
            Size in meters in the real world of patch to be extracted
        side_px : integer
            Size in pixels (size x size) of final patches to be sent back.
        """
        self.root_path = root_path
        # self.root_path = Path(root_path)
        # if not self.root_path.exists():
        #     raise ValueError(
        #         "root_path should be the directory containing the rasters, given a non-existant path: {}".format(
        #             root_path
        #         )
        #     )
      

        self.side_len = side_len_m
        self.side_px = side_px
        self.rasters_us: list[Raster] = []

    def add_all_rasters(self, **kwargs: Any) -> None:
        """Add all variables (rasters) available

        Parameters
        ----------
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        for raster_name in raster_names:
            self.append(raster_name, **kwargs)

    def add_all_bioclimatic_rasters(self, **kwargs: Any) -> None:
        """Add all bioclimatic variables (rasters) available

        Parameters
        ----------
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        for raster_name in bioclimatic_raster_names:
            self.append(raster_name, **kwargs)

    def add_all_pedologic_rasters(self, **kwargs: Any) -> None:
        """Add all pedologic variables (rasters) available

        Parameters
        ----------
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        for raster_name in pedologic_raster_names:
            self.append(raster_name, **kwargs)

    def append(self, raster_name: str, **kwargs: Any) -> None:
        """Loads and appends a single raster to the rasters already loaded.

        Can be useful to load only a subset of rasters or to pass configurations specific to each raster.

        Parameters
        ----------
        raster_name : string
            Name of the raster to load, should be a subfolder of root_path.
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        r_us = Raster(self.root_path + raster_name, raster_name, "USA", side_len_m=self.side_len, side_px=self.side_px, **kwargs)
        # r_fr = Raster(self.root_path / raster_name, "FR", size=self.size, **kwargs)

        self.rasters_us.append(r_us)
        # self.rasters_fr.append(r_fr)

    def clean(self) -> None:
        """Remove all rasters from the extractor."""
        # self.rasters_fr = []
        self.rasters_us = []

    def _get_rasters_list(self, coordinates: Coordinates) -> list[Raster]:
        """Returns the list of rasters from the appropriate country

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)

        Returns
        -------
        rasters : list of Raster objects
            All previously loaded rasters.
        """
        # if coordinates[1] > -10.0:
        #     return self.rasters_fr
        # else:
        return self.rasters_us

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        result = ""

        for rasters in [self.rasters_fr, self.rasters_us]:
            for raster in rasters:
                result += "-" * 50 + "\n"
                result += str(raster)

        return result

    def __getitem__(self, coordinates) -> npt.NDArray[np.float32]:
        """Extracts the patches around the given GPS coordinates for all the previously loaded rasters.

        Parameters
        ----------
        coordinates : tuple containing two floats and a string for destination crs
            GPS coordinates (longitude,latitude)
            dst_crs : string
            ((lon, lat),crs)
        Returns
        -------
        patch : 3d array of floats, [n_rasters, size, size], or 1d array of floats, [n_rasters,], if size == 1
            Extracted patches around the given coordinates.
        """
        rasters = self.rasters_us
        
        ## Convert each raster to the required output size
        # for r in rasters:
        #     p = torch.from_numpy(r[coordinates]).float()
        #     p = F.resize(p, (self.side_px, self.side_px))
        
        return np.concatenate([r[coordinates] for r in rasters])

    def __len__(self) -> int:
        """Number of variables/rasters loaded.

        Returns
        -------
        n_rasters : integer
            Number of loaded rasters
        """
        return len(self.rasters_us)

    def plot(
        self,
        coordinates,
        return_fig: bool = False,
        n_cols: int = 5,
        fig: Optional[plt.Figure] = None,
        resolution: float = 1.0,
    ) -> Optional[plt.Figure]:
        """Plot an environmental tensor (only works if size > 1)

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)
        return_fig : boolean
            If True, returns the created plt.Figure object
        n_cols : integer
            Number of columns to use
        fig : plt.Figure or None
            If not None, use the given plt.Figure object instead of creating a new one
        resolution : float
            Resolution of the created figure

        Returns
        -------
        fig : plt.Figure
            If return_fig is True, the used plt.Figure object
        """
        

        rasters = self._get_rasters_list(coordinates)

        # Metadata are the name of the variables and the bounding boxes in latitude-longitude coordinates
        metadata = [
            (
                raster.name,
                [
                #     coordinates[0][0] - (raster.size // 2) * raster.dataset.res[0],
                #     coordinates[0][1] + (raster.size // 2) * raster.dataset.res[0],
                #     coordinates[0][0] - (raster.size // 2) * raster.dataset.res[1],
                #     coordinates[0][1] + (raster.size // 2) * raster.dataset.res[1],
                ],
            )
            for raster in rasters
        ]

        # Extracts the patch
        patch = self[coordinates]

        # Computing number of rows and columns
        n_rows = (patch.shape[0] + (n_cols - 1)) // n_cols

        if fig is None:
            fig = plt.figure(
                figsize=(n_cols * 6.4 * resolution, n_rows * 4.8 * resolution)
            )

        axes = fig.subplots(n_rows, n_cols)
        axes = axes.ravel()

        for i, (ax, k) in enumerate(zip(axes, metadata)):
            p = np.squeeze(patch[i])  #remove the first dimension of 1,64,64 -> 64,64
            # im = ax.imshow(p, extent=k[1], aspect="equal", interpolation="none")
            im = ax.imshow(p, aspect="equal", interpolation="none")

            ax.set_title(k[0], fontsize=20)
            fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)

        for ax in axes[len(metadata) :]:
            ax.axis("off")

        fig.tight_layout()

        if return_fig:
            return fig

        return None
