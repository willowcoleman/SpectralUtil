import click
import numpy as np
from scipy.spatial import KDTree
from spec_io import load_data, write_cog
import time
from spec_io import ObservationMetadata, write_cog
from osgeo import osr
import logging


def get_ul_lr_from_files(filelist):
    """
    Get the upper left and lower right coordinates from the input file list.

    Args:
        filelist (list): list of input files.

    Returns:
        tuple: The upper left and lower right coordinates in the format (ul_x, ul_y, lr_x, lr_y).
    """
    ul_lr = [[np.nan,np.nan],[np.nan,np.nan]]
    for file in filelist:
        loc_ullr = get_extent_from_file(file.strip())
        ul_lr[0] = np.min([ul_lr[0], loc_ullr[0]])
        ul_lr[1] = np.max([ul_lr[1], loc_ullr[1]])
        ul_lr[2] = np.max([ul_lr[2], loc_ullr[2]])
        ul_lr[3] = np.min([ul_lr[3], loc_ullr[3]])
    return ul_lr


def get_subgrid_from_bounds(y_grid: np.array, x_grid: np.array, y_bounds: tuple, x_bounds: tuple):
    """
    Get the subgrid from the main grid based on the provided bounds.

    Args:
        y_grid (np.ndarray): The y coordinates of the main grid.
        x_grid (np.ndarray): The x coordinates of the main grid.
        y_bounds (tuple): The y bounds (min, max) for the subgrid.
        x_bounds (tuple): The x bounds (min, max) for the subgrid.

    Returns:
        np.ndarray: The subgrid of coordinates within the specified bounds (y)
        np.ndarray: The subgrid of coordinates within the specified bounds (x)
        int: The starting row index of the subgrid in the main grid
        int: The starting column index of the subgrid in the main grid
    """
    # Create a mask for the points within the bounds
    mask = (y_grid >= y_bounds[0]) & (y_grid <= y_bounds[1]) & \
           (x_grid >= x_bounds[0]) & (x_grid <= x_bounds[1])

    if np.sum(mask) == 0:
        return None, None, None, None
    else:
        y_locs = np.where(np.any(mask, axis=1))[0]
        x_locs = np.where(np.any(mask, axis=0))[0]

    # Extract the subgrid points
    subgrid_y = y_grid[y_locs,:][:,x_locs]
    subgrid_x = x_grid[y_locs,:][:,x_locs]

    return subgrid_y, subgrid_x, y_locs[0], x_locs[0]



def find_subgrid_locations(y_grid: np.array, x_grid: np.array, y_subgrid: np.array, x_subgrid: np.array):
    """
    Find the locations of the subgrid elements within the main grid.

    Args:
        y_grid (np.ndarray): The y coordinates of the main grid.
        x_grid (np.ndarray): The x coordinates of the main grid.
        y_subgrid (np.ndarray): The y coordinates of the subgrid.
        x_subgrid (np.ndarray): The x coordinates of the subgrid.

    Returns:
        np.ndarray: The (row, col) indices of the subgrid elements in the main grid.
    """

    # Start by subsetting the (presumably) much larger main grid
    # to only the ROI
    y_grid_minor, x_grid_minor, y_start, x_start = get_subgrid_from_bounds(y_grid, x_grid, (np.min(y_subgrid), np.max(y_subgrid)), (np.min(x_subgrid), np.max(x_subgrid)))
    if y_grid_minor is None:
        return None

    # Flatten the main grid coordinates
    main_grid_points = np.column_stack((y_grid_minor.ravel(), x_grid_minor.ravel()))

    # Create a KDTree for the main grid points
    tree = KDTree(main_grid_points)

    # Flatten the subgrid coordinates
    subgrid_points = np.column_stack((y_subgrid.ravel(), x_subgrid.ravel()))

    # Query the KDTree for the nearest neighbors
    distances, indices = tree.query(subgrid_points)

    # Convert the flat indices to row, col indices
    row_indices, col_indices = np.unravel_index(indices, y_grid_minor.shape)
    row_indices += y_start
    col_indices += x_start

    # Return the row, col indices as a 2D array
    row_indices = row_indices.reshape(y_subgrid.shape)
    col_indices = col_indices.reshape(x_subgrid.shape)

    return np.stack((row_indices, col_indices), axis=-1)


#def build_mosaic(input_file_list, ul_x, ul_y, lr_x, lr_y, x_resolution, y_resolution, output_file):
def build_mosaic_test():
    """
    Build a mosaic from the input file.

    Args:
        output_file (str): Path to the output file.
        criteria_file_list (str): Path to the input file.
        ul_x (float): Upper left x coordinate.
        ul_y (float): Upper left y coordinate.
        lr_x (float): Lower right x coordinate.
        lr_y (float): Lower right y coordinate.
        x_resolution (float): X resolution of the output mosaic.
        y_resolution (float): Y resolution of the output mosaic.
    """
    #click.echo(f"Building Mosaic from {input_file_list}")
    #meta, rfl = load_data(input_file, lazy=True, load_glt=ortho)

    #x_steps = np.arange(ul_x, lr_x, x_resolution)
    #y_steps = np.arange(ul_y, lr_y, y_resolution)

    #y_grid, x_grid = np.meshgrid(x_steps, y_steps, indexing='ij')
    x_steps = np.arange(10000)
    y_steps = np.arange(20000)
    y_grid, x_grid = np.meshgrid(x_steps, y_steps, indexing='ij')

    y_subgrid, x_subgrid = np.meshgrid(np.linspace(1000.5,2000.5,1000), np.linspace(2000.5,3000.5,1000), indexing='ij')

    start_time = time.time()
    find_subgrid_locations(y_grid, x_grid, y_subgrid, x_subgrid)
    click.echo(f"Time taken: {time.time() - start_time} seconds")


@click.command()
@click.argument('output_file', type=click.Path())
@click.argument('input_file_list', type=click.Path(exists=True))
@click.argument('x_resolution', type=float)
@click.option('--y_resolution', type=float, default=None)
@click.option('--target_extent_ul_lr', type=float, nargs=4, default=None)
@click.option('--output_epsg', type=str, default=None)
@click.option('--criteria_band', type=int, default=None)
@click.option('--criteria_mode', type=str, default="min", options=["min", "max"])
@click.option('--log_file', type=str, default=None)
def from_obs_netcdfs(output_file, input_file_list, x_resolution, y_resolution, target_extent_ul_lr, output_epsg, criteria_band, criteria_mode, log_file):
    """
    Build a mosaic from the input file.

    Args:
        output_file (str): Path to the output file.
        input_file_list (str): Path to the input file.
        x_resolution (float): X resolution of the output mosaic.
        y_resolution (float): Y resolution of the output mosaic.
        target_extent_ul_lr ((float, float, float, float)): Target extent for the mosaic in the format (ul_x, ul_y, lr_x, lr_y).
        output_epsg (str): EPSG code for the output projection.
        criteria_band (int): Band to use for the criteria.
        criteria_mode (str): Mode to use for the criteria.
        log_file (str): Path to the log file.
    """
    logging.debug(f"Building Mosaic from {input_file_list}")
    logging.debug(f"Output file: {output_file}")
    logging.debug(f"x_resolution: {x_resolution}")
    logging.debug(f"y_resolution: {y_resolution}")
    logging.debug(f"target_extent_ul_lr: {target_extent_ul_lr}")
    logging.debug(f"output_epsg: {output_epsg}")

    if y_resolution is not None and y_resolution > 0:
        logging.warning("y_resolution is set to a positive value, which is not common.  Unless this is being done very intentionally, stop, and make y negative.")

    # get projection from EPSG
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(int(output_epsg))

    input_files = open(input_file_list, 'r').readlines()

    if target_extent_ul_lr:
        ul_lr = target_extent_ul_lr
    else:
        ul_lr = get_ul_lr_from_files(input_files)

    trans = [ul_lr[0], x_resolution, 0, ul_lr[1], 0, y_resolution]
    meta = ObservationMetadata(['GLT X', 'GLT Y'], projection=proj.ExportToWkt(), geotransform=trans, pre_orthod=True, nodata_value=0)

    glt = np.zeros(( int(np.ceil((ul_lr[3] - ul_lr[1]) / y_resolution)), 
                     int(np.ceil((ul_lr[2] - ul_lr[0]) / x_resolution)),
                     4), dtype=np.uint16)
    

    for file in input_files:
        meta, obs = load_data(file.strip(), lazy=True, load_glt=False)
        loc = load_location(file.strip())
        if criteria_band is not None:
            ob = obs[:,:,criteria_band]
            
        
        if rfl is None:
            continue
        if rfl.shape[0] != glt.shape[0] or rfl.shape[1] != glt.shape[1]:
            logging.warning(f"Skipping {file.strip()} due to shape mismatch.")
            continue
        if rfl.shape[2] != 4:
            logging.warning(f"Skipping {file.strip()} due to band mismatch.")
            continue
        glt = np.maximum(glt, rfl)







@click.group()
def cli():
    pass


cli.add_command(from_obs_netcdfs)


if __name__ == '__main__':
    cli()
    #build_mosaic_test()