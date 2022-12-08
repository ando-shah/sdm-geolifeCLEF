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

from rasterio.enums import Resampling

if TYPE_CHECKING:
    import numpy.typing as npt

    Coordinates = tuple[float, float]
    Patch = npt.NDArray[np.float32]


MIN_ALLOWED_VALUE = -10000

# metadata used to setup some rasters
raster_metadata = {
    'bio_1': {'min_val':-7.0958333015441895, 'max_val':27.608333587646484, 'mean':10.678206443786621, 'std':6.177802562713623, 'nan':-3.3999999521443642e+38,'bits':16},
    'bio_2': {'min_val':1.0, 'max_val':22.233333587646484, 'mean':13.550623893737793, 'std':2.5262041091918945, 'nan':-3.3999999521443642e+38,'bits':16},
    'bio_3': {'min_val':20.337738037109375, 'max_val':100.0, 'mean':37.29763412475586, 'std':9.503908157348633, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_4': {'min_val':0.0, 'max_val':1414.1912841796875, 'mean':873.933837890625, 'std':233.8889923095703, 'nan':-3.3999999521443642e+38, 'bits': 16},
    'bio_5': {'min_val':0.5, 'max_val':46.20000076293945, 'mean':29.70604133605957, 'std':4.532764911651611, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_6': {'min_val':-27.399999618530273, 'max_val':20.700000762939453, 'mean':-7.717265605926514, 'std':8.616811752319336, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_7': {'min_val':1.0, 'max_val':51.69999694824219, 'mean':37.42325973510742, 'std':6.396346092224121, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_8': {'min_val':-12.883333206176758, 'max_val':33.883331298828125, 'mean':15.953109741210938, 'std':8.053345680236816, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_9': {'min_val':-18.133333206176758, 'max_val':33.66666793823242, 'mean':5.389002799987793, 'std':11.827828407287598, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_10': {'min_val':-1.5833332538604736, 'max_val':36.38333511352539, 'mean':21.236074447631836, 'std':4.630420684814453, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_11': {'min_val':-18.516666412353516, 'max_val':24.149999618530273, 'mean':-0.26365163922309875, 'std':8.591254234313965, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_12': {'min_val':44.0, 'max_val':3452.0, 'mean':747.0883178710938, 'std':406.6944580078125, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_13': {'min_val':9.0, 'max_val':618.0, 'mean':101.50006103515625, 'std':52.118675231933594, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_14': {'min_val':0.0, 'max_val':175.0, 'mean':31.740530014038086, 'std':26.625768661499023, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_15': {'min_val':5.005423069000244, 'max_val':142.28643798828125, 'mean':43.1767692565918, 'std':24.751798629760742, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_16': {'min_val':20.0, 'max_val':1571.0, 'mean':274.7115173339844, 'std':145.2008514404297, 'nan':-3.3999999521443642e+38, 'bits':16},
    'bio_17': {'min_val':0.0, 'max_val':550.0, 'mean':110.51180267333984, 'std':87.41494750976562, 'nan':-3.3999999521443642e+38,'bits':16},
    'bio_18': {'min_val':2.0, 'max_val':975.0, 'mean':221.00624084472656, 'std':115.31585693359375, 'nan':-3.3999999521443642e+38,'bits':16},
    'bio_19': {'min_val':11.0, 'max_val':1461.0, 'mean':155.13890075683594, 'std':143.87017822265625, 'nan':-3.3999999521443642e+38,'bits':16},
    'bdticm': {'min_val': 0., 'max_val': 112467., 'mean': 2632.716751534344, 'std':3209.621533299418,'nan': -2147483647.0, 'bits': 32},
    'bldfie': {'min_val':93.0, 'max_val':1828.0, 'mean':1378.64892578125, 'std':137.49325561523438, 'nan':-32768.0,'bits':16},
    'cecsol': {'min_val':0.0, 'max_val':385.0, 'mean':21.063919067382812, 'std':7.909461975097656, 'nan':-32768.0, 'bits':16},
    'clyppt': {'min_val':0.0, 'max_val':81.0, 'mean':22.082963943481445, 'std':9.078008651733398, 'nan':-32768.0,'bits':16},
    'orcdrc': {'min_val':0.0, 'max_val':524.0, 'mean':23.83712387084961, 'std':21.804183959960938, 'nan':-32768.0, 'bits':16},
    'phihox': {'min_val':32.0, 'max_val':98.0, 'mean':64.52721405029297, 'std':11.172294616699219, 'nan':-32768.0, 'bits':16},
    'sltppt': {'min_val':0.0, 'max_val':86.0, 'mean':35.99134826660156, 'std':11.597208023071289, 'nan':-32768.0, 'bits':16},
    'sndppt': {'min_val':0.0, 'max_val':99.0, 'mean':41.921875, 'std':13.934831619262695, 'nan':-32768.0, 'bits':16}
}



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
        norm:str = "None",
        out_dtype:str = "float"
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
        norm : str = {"none", "min-max", "std"}
            if none: simply divide the value by the 2^('bits' - 8) in it's metadata,e.g. for 16 bit-> divide by 256, to get it to within 256 bits
                        might have to add + 128 to remove all negative values and then convert to uint8
            if min-max: convert to min-max normalization and then to between 0 and 255 ->uint8
            if std: convert
        out_dtype : str = {"uint8", "uint16", "float"}
            float -> implies that the values are scaled b/w 0 and 1
            uint8 and int16 -> values are b/w 0 and 255 or -32767 and 32,767
        """
        
        self.path = path 
        self.name = name
        self.side_length_deg = side_len_m / (111110.)
        
        self.raster = None
        self.norm = norm
        
        self.crop_buf_perc = 0.15 #-> 15% extra crop for warping issues
        
        
        #save all the metadata
        self.min, self.range = raster_metadata[self.name]['min_val'], raster_metadata[self.name]['max_val'] - raster_metadata[self.name]['min_val']
        self.mean, self.std = raster_metadata[self.name]['mean'], raster_metadata[self.name]['std']
        self.div = 2**(int(raster_metadata[self.name]['bits']) - 8)
        
        self.out_dtype = out_dtype
        
        
        filename = "{:}/{:}_{:}.tif".format(self.path,self.name,country)
        print("Processing Raster file for ", self.name)
        # print(filename, name)
    
        with rasterio.Env():
            with rasterio.open(filename, "r") as f:
                self.raster = rioxarray.open_rasterio(f, masked=True)
                #Dont do this now, then it's lazy loaded
                # self.raster = self.raster.rio.reproject(self.dst_crs)

        # print("Completed Setup of Raster {:} in CRS = {:}".format(self.name, self.crs, self.raster.data.dtype))
        # print("Completed Setup of Raster {:} with mean[{:.2} / std[{:.2}]] ".format(self.name, self.mean, self.std))

    
    def _extract_patch(self, coordinates):
        """Extracts the patch around the given GPS coordinates.
        Avoid using this method directly.

        Parameter
        ----------
        aoi : tuple containing two floats and a reference to a SI raster object
            GPS coordinates (longitude,latitude)
            ((lon, lat),aoi)
        Returns
        -------
        patch : tensor of data of the same size as the si_aoi.
        """
        
        lon, lat, aoi_si = coordinates[0][0], coordinates[0][1], coordinates[1]
        point_geom = shapely.geometry.mapping(shapely.geometry.Point(lon, lat))

        #Convert the point to a shape -> KEEP IN 4326
        point_shape = shapely.geometry.shape(point_geom)
        #Create a square out of it with side_length = buffer*2 + some buffer
        #Buffer is needed because when it gets reprojected, we will lose some of edges
        mask_shape = point_shape.buffer(self.side_length_deg/2 * (1+self.crop_buf_perc)).envelope
        mask_geom = shapely.geometry.mapping(mask_shape)

        try:
            #pre-clip
            aoi = self.raster.rio.clip([mask_geom], from_disk=True).rio.reproject_match(aoi_si,resampling=Resampling.bilinear)
            # aoi = self.raster.rio.reproject_match(aoi_si)            
            # print(aoi.values.shape)
            
            
            
#             if (self.norm == "min-max"):
#                 #rescale to between min and max -> this should be precomputed and stored in raster_metadata
#                 aoi.values = (aoi.values - self.min) / self.range
#                 if(self.out_dtype == "uint8"):
#                     aoi.data *= 255
#                     t = torch.from_numpy(aoi.values.astype(np.uint8))
#                     # aoi.values.astype(np.uint8)

#                 elif (self.out_dtype == "int16"):
#                     aoi.data *= (65536/2 - 1) 
#                     t = torch.from_numpy(aoi.values.astype(np.int16))
#                     # aoi.values.astype(np.uint16)
                                     
            
            #TODO Convert to EA projection here:
            #Convert to uint8, since the values were already scaled to between 0 and 255 before
            # t_env = torch.from_numpy(cropped_env_raster.values.astype(np.uint8))
            t = torch.from_numpy(aoi.values)
            if self.norm == 'std':
                # print("standardizing")
                transform = T.Compose ([T.CenterCrop(size=t.shape[1]),
                                        T.Normalize((self.mean),(self.std))])
            else:
                transform = T.CenterCrop(size=(t.shape[1]))
            #crop to shorter side
            t = transform(t)
            #set nans to zero
            t[torch.isnan(t)] = 0.
            
            print(self.name, t.max(), t.min())
            
            # print(t.shape, type(t))

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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "name: " + self.name + "\n"


class EnvPatchExtractor(object):
    """
    Handles the loading and extraction of an environmental tensor from multiple rasters given GPS coordinates.
    """

    def __init__(self, root_path: str, side_len_m: int = 1000, out_dtype:str="float", norm:str="None"):
        """Constructor

        Parameters
        ----------
        root_path : string or pathlib.Path
            Path to the folder containing all the rasters.
        side_len_m : integer
            Size in meters in the real world of patch to be extracted
        out_dtype : str = {"uint8", "uint16", "float"}
            float -> implies that the values are scaled b/w 0 and 1
            uint8 and uint16 -> values are b/w 0 and 255 or 0 and 32,767
        """
        self.root_path = root_path

        self.side_len = side_len_m
        self.rasters_us: list[Raster] = []
        self.out_dtype = out_dtype
        self.norm = norm

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
        r_us = Raster(self.root_path + raster_name, raster_name, "USA", side_len_m=self.side_len, out_dtype=self.out_dtype, norm=self.norm, **kwargs)
        # r_fr = Raster(self.root_path / raster_name, "FR", size=self.size, **kwargs)

        self.rasters_us.append(r_us)
        # self.rasters_fr.append(r_fr)
        
    def _dump_metadata(self):
        
        for r in self.rasters_us:
            print("[{:}]  'min_val':{}, 'max_val':{}, 'mean':{}, 'std':{}, 'nan':{}, type:{}".format(r.name, np.nanmin(r.raster.values), np.nanmax(r.raster.values), np.nanmean(r.raster.values), np.nanstd(r.raster.values), r.raster.encoding['_FillValue'], r.raster.encoding['dtype']))

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

    def __getitem__(self, coordinates):
        """Extracts the patches around the given GPS coordinates for all the previously loaded rasters.

        Parameters
        ----------
        coordinates : tuple containing two floats and a raster object reference for destination SI image
            GPS coordinates (longitude,latitude)
            dst_crs : string
            ((lon, lat), aoi_si)
        Returns
        -------
        patch : 3d array of floats, [n_rasters, size, size], tensors
        """
        rasters = self.rasters_us
        
        return torch.stack([r[coordinates].squeeze() for r in rasters])


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
