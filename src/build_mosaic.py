import click
import numpy as np
from scipy.spatial import KDTree
import time
import spec_io
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
    ul_lr = [np.nan,np.nan,np.nan,np.nan]
    for file in filelist:
        loc_ullr = spec_io.get_extent_from_obs(file.strip())
        ul_lr[0] = np.nanmin([ul_lr[0], loc_ullr[0]])
        ul_lr[1] = np.nanmax([ul_lr[1], loc_ullr[1]])
        ul_lr[2] = np.nanmax([ul_lr[2], loc_ullr[2]])
        ul_lr[3] = np.nanmin([ul_lr[3], loc_ullr[3]])
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
    #row_indices += y_start
    #col_indices += x_start

    # Return the row, col indices as a 2D array
    row_indices = row_indices.reshape(y_subgrid.shape)
    col_indices = col_indices.reshape(x_subgrid.shape)

    return np.stack((row_indices, col_indices), axis=-1), x_start, y_start


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
@click.option('--criteria_mode', type=click.Choice(["min","max"]), default="min")
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
    elif y_resolution is None:
        y_resolution = -1 * x_resolution


    if input_file_list.endswith(".nc"):
        input_files = [input_file_list]
    else:
        input_files = open(input_file_list, 'r').readlines()

    # get projection from EPSG
    if output_epsg is not None:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(int(output_epsg))
        proj = proj.ExportToWkt()
    else:
        meta, obs = spec_io.load_data(input_files[0].strip(), lazy=True, load_glt=False)
        proj = meta.projection

    if target_extent_ul_lr:
        ul_lr = target_extent_ul_lr
    else:
        ul_lr = get_ul_lr_from_files(input_files)
    
    print("Bounding box (ul_lr): " + str(ul_lr)) 

    trans = [ul_lr[0], x_resolution, 0, ul_lr[1], 0, y_resolution]
    meta = spec_io.ObservationMetadata(['GLT X', 'GLT Y', 'File Index', 'OBS val'], projection=proj, geotransform=trans, pre_orthod=True, nodata_value=0)

    glt = np.zeros(( int(np.ceil((ul_lr[3] - ul_lr[1]) / y_resolution)), 
                     int(np.ceil((ul_lr[2] - ul_lr[0]) / x_resolution)),
                     4), dtype=np.uint16)
    y_grid, x_grid = np.meshgrid(np.linspace(trans[3], trans[3] + trans[5]*glt.shape[0],glt.shape[0]), 
                                 np.linspace(trans[0], trans[0] + trans[1]*glt.shape[1],glt.shape[1]),
                                 indexing='ij')
    

    for _file, file in enumerate(input_files):
        local_meta, obs, loc = spec_io.load_data(file.strip(), lazy=True, load_glt=False, load_loc=True)
        if criteria_band is not None:
            ob = obs[:,:,criteria_band]
            
        sub_glt, x_offset, y_offset = find_subgrid_locations(y_grid, x_grid, loc[...,1], loc[...,0])  

        # handle offset
        existing_crit = glt[sub_glt[...,0]+y_offset, sub_glt[...,1]+x_offset, 3]
        if criteria_mode == "min":
            crit_mask = np.logical_and(ob < existing_crit, np.isnan(existing_crit) == False)
        elif criteria_mode == "max":
            crit_mask = np.logical_and(ob > existing_crit, np.isnan(existing_crit) == False)
        else:
            raise ValueError(f"Invalid criteria_mode: {criteria_mode}")

        glt[sub_glt[crit_mask,0]+y_offset, sub_glt[crit_mask,1]+x_offset, :2] = sub_glt[crit_mask,:]
        glt[sub_glt[crit_mask,0]+y_offset, sub_glt[crit_mask,1]+x_offset, 2] = _file
        glt[sub_glt[crit_mask,0]+y_offset, sub_glt[crit_mask,1]+x_offset, 3] = ob[crit_mask]


    spec_io.write_cog(output_file, glt, meta, nodata_value=0)




@click.group()
def cli():
    pass


cli.add_command(from_obs_netcdfs)


if __name__ == '__main__':
    cli()
    #build_mosaic_test()