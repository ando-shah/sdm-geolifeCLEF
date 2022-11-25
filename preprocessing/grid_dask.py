import numpy as np
from shapely.geometry import Polygon, box
import geopandas as gpd
from sklearn.neighbors import KDTree
from tqdm import tqdm
import dask

import dask_gateway
gateway = dask_gateway.Gateway()
cluster_options = gateway.cluster_options()

client, cluster = None, None

def check_for_existing_clusters():
    if len(gateway.list_clusters()) == 0:
        return False   
    return True

#max is 227
def setup_dask_cluster(max=50, mem=16, adapt=True):
    global cluster_options, cluster, client
    
    cluster_options["worker_memory"] = mem
    
    if check_for_existing_clusters():
        print ("Clusters already exist, latching onto the first one")
        
        clusters = gateway.list_clusters()
        cluster = gateway.connect(clusters[0].name)
        
        client = cluster.get_client()
        
    else:
        print("Setting up new cluster..")
        cluster = gateway.new_cluster(cluster_options, shutdown_on_close=False)
        print("Getting client..")
        client = cluster.get_client()
        if adapt:
            cluster.adapt(minimum=2, maximum=max)
        else:
            cluster.scale(max)

    
    
    print(client)
    print(cluster.dashboard_link)
    return cluster, client

    
def shutdown_all_clusters():
    
    clusters = gateway.list_clusters()
    if clusters is not None:
        for c in clusters:
            cluster = gateway.connect(c.name)
            cluster.shutdown()
            print (cluster)
    

def dashboard():
    return cluster.dashboard_link



class Observations:
    

    def __init__(self, obs_gdf, map_gdf:gpd.GeoDataFrame,shape:str='square', grid_res_deg:float=1.0, crs:str='epsg:4326'):
        
        self.gdf_obs = obs_gdf
        self.total_bounds = map_gdf.total_bounds
        
        self.min_x, self.min_y, self.max_x, self.max_y = self.total_bounds
        self.xrange = np.arange(self.min_x - grid_res_deg, self.max_x + grid_res_deg, grid_res_deg)
        self.yrange = np.arange(self.min_y - grid_res_deg, self.max_y + grid_res_deg, grid_res_deg)
        
        self.shape = shape
        self.grid_resolution = grid_res_deg
        self.crs = crs
        self.grid_gd = None #Geopandas frame -> one for every cell
        self.grid_np = None #Numpy array corresponding to Geopandas frame -> one for every cell
        self.grid_kde = None #geopandas frame that has the geometry, grid_id and a column for the probability that a species_id was observered there
        self.land_mask = None
        self.kde_radius = None
        
        self.class_list = self.gdf_obs.species_id.unique()
        print("Number of classes in entire dataset: ", len(self.class_list))
        self.num_classes = len(self.class_list)
        
        print("Gridding..")
        self.create_grid(shape = self.shape,
                         side_length = self.grid_resolution,
                         crs = self.crs)
        
        print("Creating Land Mask..")
        self.aoi_map = map_gdf.simplify(self.grid_resolution)
        self.create_mask(self.aoi_map.geometry)
        
        print("Clipping observation to AOI..")
        self.mask_obs()
        self.class_list = self.gdf_obs.species_id.unique()
        print("Number of classes in AOI dataset: ", len(self.class_list))
        
        self.prob_min = 1e-5
        self.prob_max = 1.0
        
        # global client, cluster
        # print(client, cluster.dashboard_link)
        
        # print(len(self.xrange), len(self.yrange), self.grid_np.shape)
        print(self.grid_np.shape)
        
    def __getitem__(self, idx):
        """
        returns the geoseries corresponding to the observation id
        """
        return self.gdf_obs.loc(idx)

    

            
    def save_kde(self, filename:str):
        if (filename is None):
            filename = "geolifeclef_usa_" + str(self.grid_resolution)
            if grid_kde is not None:
                filename = filename + "kde_" + str(self.kde_radius)            
            
        else:
            filename = filename + '.feather'
            
            
        self.grid_kde.to_feather(filename)
        
    def load_kde(self, filename):
        self.grid_kde = gpd.read_feather(filename)

    
    def get_kde(self):
        """Returns the outputs of the (cumulative) KDE fn
        """
        
        return self.grid_kde
    
    def clear_kde(self):
        self.grid_kde = None
        
    def create_mask(self, geoseries):
        print("Num entries before masking: ", len(self.grid_gd))
        #intersects will include cells that touch the boundary but are not inside
        self.land_mask = self.grid_gd.intersects(geoseries.iloc[0]).to_numpy()
        #within only considers grids that are *entirely* inside the boundary
        # self.land_mask = self.grid_gd.within(geoseries.iloc[0]).to_numpy()
        self.grid_np = self.grid_np[self.land_mask]
        self.grid_gd = self.grid_gd[self.land_mask].reset_index(drop = True)
        
        # Create a column that assigns each grid a number
        self.grid_gd["grid_id"] = np.arange(len(self.grid_gd))
        
        
        print("Num entries after masking: ", len(self.grid_gd))      
        assert(len(self.grid_np) == len(self.grid_gd))
        
    def mask_obs(self):
        """
        Mask the observation gdf file by the area of interest
        """
        self.gdf_obs['valid'] = self.gdf_obs.intersects(self.aoi_map.geometry.iloc[0])
        self.gdf_obs = self.gdf_obs[self.gdf_obs.valid == True]


    def create_grid(self, shape='square', side_length=1.0, crs='epsg:4326'):
        '''Create a grid consisting of either rectangles or hexagons with a 
        specified side length that covers the extent of input feature.

        Inputs
        total_bounds = output of a geopands.total_bounds for defining the extents of the grid (has to be rectangular)
        shape (str) : {'square', 'rectangle', 'box', 'hexagon'}
        side_length (float): resolution of the grid in degrees
        crs (string): coordinate reference system you want for the gridded output

        Outputs:
        geopandas frame with the grids
        x_range: list of ?

        returns a geopandas frame with a cell of given shape per row
        '''

        #extra outputs:
        # xrange, yrange = [], []

        # Get extent of buffered input feature
        

        # Create empty list to hold individual cells that will make up the grid
        cells_list = []

        # Create grid of squares if specified
        if shape in ["square", "rectangle", "box"]:

            # Adapted from https://james-brennan.github.io/posts/fast_gridding_geopandas/
            # Create and iterate through list of x values that will define column positions with specified side length
            for x in self.xrange:
                # xrange.append(x)
                # yrange = []

                # Create and iterate through list of y values that will define row positions with specified side length
                for y in self.yrange:

                    # Create a box with specified side length and append to list
                    cells_list.append(box(x, y, x + side_length, y + side_length))
                    # yrange.append(y)


        # Otherwise, create grid of hexagons
        elif shape == "hexagon":

            # Set horizontal displacement that will define column positions with specified side length (based on normal hexagon)
            x_step = 1.5 * self.grid_resolution

            # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
            # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
            y_step = math.sqrt(3) * self.grid_resolution

            # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
            apothem = (math.sqrt(3) * self.grid_resolution / 2)

            # Set column number
            column_number = 0

            # Create and iterate through list of x values that will define column positions with vertical displacement
            for x in np.arange(self.min_x, self.max_x + x_step, x_step):

                # Create and iterate through list of y values that will define column positions with horizontal displacement
                for y in np.arange(self.min_y, self.max_y + y_step, y_step):

                    # Create hexagon with specified side length
                    hexagon = [[x + math.cos(math.radians(angle)) * self.grid_resolution, y + math.sin(math.radians(angle)) * self.grid_resolution] for angle in range(0, 360, 60)]

                    # Append hexagon to list
                    cells_list.append(Polygon(hexagon))

                # Check if column number is even
                if column_number % 2 == 0:

                    # If even, expand minimum and maximum y values by apothem value to vertically displace next row
                    # Expand values so as to not miss any features near the feature extent
                    min_y -= apothem
                    max_y += apothem

                # Else, odd
                else:

                    # Revert minimum and maximum y values back to original
                    min_y += apothem
                    max_y -= apothem

                # Increase column number by 1
                column_number += 1

        # Else, raise error
        else:
            raise Exception("Specify a rectangle or hexagon as the grid shape.")

        # Create grid from list of cells
        self.grid_gd = gpd.GeoDataFrame(cells_list, columns = ['geometry'], crs = crs)

        
        grid_x = np.asarray([c.centroid.x for c in cells_list])
        grid_y = np.asarray([c.centroid.y for c in cells_list])
        
        self.grid_np = np.vstack([grid_x, grid_y]).T
        
   