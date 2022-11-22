import numpy as np
from shapely.geometry import Polygon, box
import geopandas as gpd
from sklearn.neighbors import KDTree
from tqdm import tqdm

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
        
        self.class_list = self.gdf_obs.species_id.unique()
        self.num_classes = len(self.class_list)
        
        print("Gridding..")
        self.create_grid(shape = self.shape,
                         side_length = self.grid_resolution,
                         crs = self.crs)
        
        print("Creating Land Mask..")
        self.create_mask(map_gdf.simplify(self.grid_resolution).geometry)
        
        self.prob_min = 1e-5
        self.prob_max = 1.0
        
        
        
        print(len(self.xrange), len(self.yrange), self.grid_np.shape)
        
    def __getitem__(self, idx):
        """
        returns the geoseries corresponding to the observation id
        """
        return self.gdf_obs.loc(idx)

    
    
    def kde(self, radius:float=0.5, kernel:str='gaussian', display_every:int=100):
        """Populates the grid_kde data structure
           with 1 column per species_id for every grid_cell
           **fingers crossed**
        """
        for id in tqdm(self.class_list):
            # if i % display_every == 0:
            #     print("Gridding and KDE for species ID ", id)
            self.kernel_per_species(kde=True, chosen_id=id, radius=radius, kernel=kernel, cumulative=True)
            
    def save_kde(self, filename:str='geolifeclef_usa_kde'):
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
        self.land_mask = self.grid_gd.within(geoseries.iloc[0]).to_numpy()
        self.grid_gd['mask'] = self.land_mask
        # self.grid_np = self.grid_np[self.land_mask]


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

        # Create a column that assigns each grid a number
        #Index is the GridID
        self.grid_gd["grid_id"] = np.arange(len(self.grid_gd))
        
        grid_x = np.asarray([c.centroid.x for c in cells_list])
        grid_y = np.asarray([c.centroid.y for c in cells_list])
        
        self.grid_np = np.vstack([grid_x, grid_y]).T
        
        # grid_x = cells_list.geometry.centroid.x.to_numpy()
        # grid_y = cells_list.geometry.centroid.y.to_numpy()

        # Return grid
        # return grid, np.asarray(xrange), np.asarray(yrange)



    def kernel_per_species(self, kde:bool=True, chosen_id:int=0, cumulative:bool=False, radius:float=0.5, kernel:str='gaussian'):

        """
        Inputs:
        kde: whether or not to apply KDE
            if False: only group observations within a cell
            if True: group and then apply KDE
        chosen_id: Species ID for a single chosen species
        grid: grid to which these observations must conform
        radius: radius in degrees for the kernel function
        kernel: 'linerar' / 'epanechnikov' / 'gaussian' ..
        cumulative: keep adding to a master species list -> grid_kde

        Output:

        """

        ##Group observations per cell:
        gdf_chosen = self.gdf_obs[self.gdf_obs.species_id==chosen_id]
        # gdf_chosen = gdf_chosen[gdf_chosen.mask == True]
        
        # Remove duplicate counts
        # With intersect, those that fall on a boundary will be allocated to all cells that share that boundary
        # chosen_species_grid = chosen_species_grid.drop_duplicates(subset = ['speciesID']).reset_index(drop = True)

        chosen_species_grid = gpd.sjoin(gdf_chosen, self.grid_gd, how='inner', predicate='within').drop(['index_right'], axis=1)
        # display(chosen_species_grid)

        # Add a field with constant value of 1
        chosen_species_grid['num_obs'] = 1.0

        # Group GeoDataFrame by cell while aggregating the Count values
        chosen_species_grid = chosen_species_grid.groupby('grid_id').agg({'num_obs':'sum'})

        chosen_species_grid = self.grid_gd.merge(chosen_species_grid, on = 'grid_id', how = "right")
 
        if (kde and radius*2 > self.grid_resolution):
            
            #List of grids that have observations for this species
            chosen_grids = list(chosen_species_grid['grid_id'])
            # print(chosen_grids, chosen_species_grid)
            #sanity check
            if len(chosen_grids) == 0:
                return 
            
            # Create training set for KDE
            x_train = self.grid_np[chosen_grids,:]
            y_train = chosen_species_grid['num_obs'].to_numpy()

            #Apply the KDE
            # y_hat = np.zeros_like(self.grid_np)
            y_hat = np.zeros_like(self.land_mask)

            tree = KDTree(x_train)
            y_hat = tree.kernel_density(self.grid_np, h=radius, kernel=kernel)*(1*(radius**2))
            
            # y_hat /= y_hat.max()
            # print("Max y_hat = ", y_hat.max())
            #Cap all cells at 1
            y_hat[y_hat > self.prob_max] = self.prob_max
            y_hat[y_hat < self.prob_min] = 0.0

            # print("Shapes: X_train={:}, y_train={:}, arr_xy={:}, y_hat={:}, sum(y_hat)={:}" .format(x_train.shape, y_train.shape, self.grid_np.shape, y_hat.shape, y_hat.sum()))
            # y_hat_grid = y_hat.reshape(len(self.xrange), len(self.yrange))
            # # y_hat_grid = y_hat.reshape(x_range.shape[0], y_range.shape[0])
            # display = np.rot90(y_hat_grid)
        
        
        #Create a new column name for this species
        col_name = 'prob_{}'.format(chosen_id)
        #Now convert back into a geopandas frame that has one entry for each grid cell
        if cumulative:
            if self.grid_kde is None: #not initialized yet
                self.grid_kde = self.grid_gd.copy()
            
            self.grid_kde[col_name] = y_hat.tolist()
            #no return
            
        else: #Only for one-time test usage
            self.grid_kde = self.grid_gd.copy()
            self.grid_kde[col_name] = y_hat.tolist()

    