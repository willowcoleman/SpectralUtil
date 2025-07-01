import click
import os
import numpy as np
from scipy.spatial import KDTree
from scipy.signal import convolve2d

import time
import spec_io
from osgeo import osr, gdal, ogr 
import pyproj
import logging
from tqdm import tqdm
from copy import deepcopy
osr.UseExceptions()

@click.command()
@click.argument('ortho_file', type=click.Path(exists=True))
@click.argument('json_file', type=click.Path(exists=True))
@click.argument('urban_data', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--output_res', type=float, default = 0.000542232520256) ## default to EMIT res
@click.option('--nodata_value', type=float, default=0)
def urban_mask(ortho_file, json_file, urban_data, output_file, output_res, nodata_value):

    # Get SRS info from orthoed file 
    ds = gdal.Open(ortho_file)
    wkt = ds.GetProjection()
    ds = None

    # Build warp options
    warp_options = gdal.WarpOptions(
        cutlineDSName=json_file,
        cropToCutline=True,
        dstNodata=nodata_value,
        xRes=output_res,
        yRes=-output_res,
        dstSRS=wkt
    )
    gdal.Warp(destNameOrDestDS="clipped.tif", srcDSOrSrcDSTab=urban_data, options=warp_options)

    # Generate geotiff mask of urban areas (50 in ESA worldcover)
    ds = gdal.Open("clipped.tif") # temporary file 
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    result = np.logical_and(array >= 0, array == 50).astype(np.uint8)

    # Create output file with the same georeference and projection
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # Write result and set NoData value
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result)
    out_band.SetNoDataValue(nodata_value)

    os.remove("clipped.tif") # del temporary file 


@click.command()
@click.argument('json_file', type=click.Path(exists=True))
@click.argument('coastal_data', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=True))
@click.option('--output_res', type=float, default = 0.000542232520256) ## default to EMIT res
def coastal_mask(json_file, coastal_data, output_file, output_res): 

    # Clip large coastal shapefile to EMIT extent (too slow) 
    gdal.VectorTranslate(
        "temp.shp",                              # Output file
        coastal_data,                            # Input file (coastal data)
        options=gdal.VectorTranslateOptions(
            format="ESRI Shapefile",
            clipSrc=json_file                    # Clip to tile extent 
        )
    )

    # Get extent from shapefile for new raster 
    shp_ds = ogr.Open("temp.shp")
    layer = shp_ds.GetLayer()   
    minx, maxx, miny, maxy = layer.GetExtent()
    x_res = int((maxx - minx) / output_res)
    y_res = int((maxy - miny) / output_res)

    # Create output raster
    out_ds = gdal.GetDriverByName("GTiff").Create(output_file, x_res, y_res, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform((minx, output_res, 0, maxy, 0, -output_res))
    out_band = out_ds.GetRasterBand(1)
    out_band.Fill(1)

    # Set projection and rasterize 
    srs = layer.GetSpatialRef()
    if srs:
        out_ds.SetProjection(srs.ExportToWkt())

    gdal.RasterizeLayer(
        out_ds,
        [1],  # band index
        layer,
        burn_values=[0],
        options=["INVERT=FALSE"]
    )

    os.remove("temp.shp") # del temporary file 
    


def remove_negatives(glt, clean_contiguous=False, clean_interpolated=False):
    """
    Remove the negative values from the GLT.

    Args:
        glt (np.ndarray): The GLT to clean.
        clean_contiguous (bool): Whether to clean contiguous negative values.
    """

    if clean_contiguous:
        nm = glt[...,0] < 0
        # put 2d convolution onto nm
        nm = convolve2d(nm, np.ones((3,3)), mode='same', boundary='fill', fillvalue=0)
        glt[nm >= 3,:] = 0

    # Remove interpolation indicator 
    if clean_interpolated:
        glt[...,:2] = np.abs(glt[...,:2])
    return glt


def get_ul_lr_from_files(filelist, get_resolution=False):
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


def find_subgrid_locations(y_grid: np.array, x_grid: np.array, y_subgrid: np.array, x_subgrid: np.array, max_distance: float = None, n_workers: int = 1) -> tuple[np.ndarray, int, int, int, int]:
    """
    Find the locations of the subgrid elements within the main grid.

    Args:
        y_grid (np.ndarray): The y coordinates of the main grid.
        x_grid (np.ndarray): The x coordinates of the main grid.
        y_subgrid (np.ndarray): The y coordinates of the subgrid.
        x_subgrid (np.ndarray): The x coordinates of the subgrid.
        max_distance (float): The maximum distance to search for the nearest neighbor.
        n_workers (float): Number of works to execute call in parallel with.

    Returns:
        np.ndarray: The (row, col) indices of the subgrid elements in the main grid.
        int: The starting x index of the subgrid in the main grid
        int: The starting y index of the subgrid in the main grid
        int: The number of columns in the subgrid
        int: The number of rows in the subgrid
    """
    st = time.time()
    # Start by subsetting the (possibly) much larger main grid
    # to only the ROI
    y_grid_minor, x_grid_minor, y_start, x_start = get_subgrid_from_bounds(y_grid, x_grid, (np.min(y_subgrid), np.max(y_subgrid)), (np.min(x_subgrid), np.max(x_subgrid)))
    if y_grid_minor is None:
        return None, None

    # Flatten the main grid coordinates
    main_grid_points = np.column_stack((y_grid_minor.ravel(), x_grid_minor.ravel()))

    # Flatten the subgrid coordinates
    subgrid_points = np.column_stack((y_subgrid.ravel(), x_subgrid.ravel()))

    # Create a KDTree for the main grid points
    tree = KDTree(subgrid_points)
    logging.debug(f"Time to build tree: {time.time() - st}")

    # Query the KDTree for the nearest neighbors
    st = time.time()
    distances, indices = tree.query(main_grid_points, workers=n_workers)
    logging.debug(f"Time to querry tree: {time.time() - st}")

    # Convert the flat indices to row, col indices
    row_indices, col_indices = np.unravel_index(indices, y_subgrid.shape)

    # Offset by 1; 0 is nodata
    row_indices += 1
    col_indices += 1

    # Return the row, col indices as a 2D array
    row_indices = row_indices.reshape(y_grid_minor.shape)
    col_indices = col_indices.reshape(x_grid_minor.shape)


    sub_glt_insert_idx = np.meshgrid(np.arange(y_start, y_start + y_grid_minor.shape[0]), 
                                     np.arange(x_start, x_start + y_grid_minor.shape[1]), indexing='ij')

    sub_glt_insert_idx = np.stack(sub_glt_insert_idx, axis=-1)

    # Interpolated values are negative
    if max_distance is not None:
        mask = distances.reshape(y_grid_minor.shape) > max_distance
        col_indices[mask] *= -1
        row_indices[mask] *= -1
    else:
        if all([s > 1 for s in x_grid_minor.shape]) and all([s > 1 for s in y_grid_minor.shape]):
            md = np.sqrt((y_grid_minor[1,0] - y_grid_minor[0,0])**2 + \
                        (x_grid_minor[0,1] - x_grid_minor[0,0])**2) * 1.5
        else:
            # corner case where only one line of grid is present
            md = np.sqrt((y_grid_minor.flatten()[1] - y_grid_minor.flatten()[0])**2 + \
                        (x_grid_minor.flatten()[1] - x_grid_minor.flatten()[0])**2) * 1.5
        mask = distances.reshape(y_grid_minor.shape) > md
        col_indices[mask] *= -1
        row_indices[mask] *= -1

    # Esnure nodata masking is consistent between rows and columns
    row_indices[col_indices == 0] = 0
    col_indices[row_indices == 0] = 0

    return np.stack((col_indices, row_indices), axis=-1), sub_glt_insert_idx


@click.command()
@click.argument('glt_files', type=click.Path(exists=True))
@click.argument('obs_file_lists', type=click.Path(exists=True))
@click.argument('output_glt_file', type=click.Path())
@click.argument('output_file_list', type=click.Path())
def stack_glts(glt_files, obs_file_lists, output_glt_file, output_file_list):
    """
    Stack the GLTs from the input files.

    Args:
        glt_files (list): List of GLT files to stack, in order.
        obs_file_lists (list): List of observation file lists, matching order to glt_files.
        output_glt_file (str): Path to the output glt file.
        output_file_list (str): Path to the output file list.
    """
    
    glt = None
    file_list = None
    
    file_lists_raw = []
    obs_file_lists = [x.strip() for x in open(obs_file_lists, 'r').readlines()]
    glt_files = [x.strip() for x in open(glt_files, 'r').readlines()]
    for fl in obs_file_lists:
        if not os.path.exists(fl):
            raise ValueError(f"File {fl} does not exist.")
        file_lists_raw.extend([x.strip() for x in open(fl, 'r').readlines()])
    merged_file_list, file_list_idx = np.unique(file_lists_raw, return_inverse=True)
    
    intra_list_idx = 0
    for glt_file, obs_file_list in zip(glt_files, obs_file_lists):
        glt_meta, glt_data = spec_io.load_data(glt_file, lazy=False)
        converted_glt_data = glt_data.copy()
        converted_glt_data[...,2] = -1
        
        original_file_idxs = np.unique(glt_data[...,2])
        #for idx in range(intra_list_idx, intra_list_idx + len(obs_file_list)):
            #converted_glt_data[...,2][glt_data[...,2] == idx - intra_list_idx] = file_list_idx[idx]
        for idx in original_file_idxs:
            converted_glt_data[...,2][glt_data[...,2] == idx] = file_list_idx[idx]
        
        if glt is None:
            glt = glt_data
        else:
            to_copy = np.logical_and(glt[...,2] == 0, converted_glt_data[...,2] > 0)
            glt[to_copy,:] = converted_glt_data[to_copy,:]

        intra_list_idx += len(obs_file_list)
    spec_io.write_cog(output_glt_file, glt, glt_meta, nodata_value=0)
    np.savetxt(output_file_list, merged_file_list, fmt="%s")


@click.command()
@click.argument('output_file', type=click.Path())
@click.argument('input_file_list', type=click.Path(exists=True))
@click.option('--ignore_file_list', type=click.Path(exists=True), default=None)
@click.option('--x_resolution', type=float, default=None)
@click.option('--y_resolution', type=float, default=None)
@click.option('--target_extent_ul_lr', type=float, nargs=4, default=None)
@click.option('--output_epsg', type=str, default=4326)
@click.option('--criteria_band', type=int, default=None)
@click.option('--criteria_mode', type=click.Choice(["min","max"]), default="min")
@click.option('--n_cores', type=int, default=1)
@click.option('--max_distance', type=float, default=None)
@click.option('--log_file', type=str, default=None)
@click.option('--log_level', type=click.Choice(["DEBUG","INFO","WARN","ERROR"]), default="INFO")
def build_obs_nc(output_file, input_file_list, ignore_file_list, x_resolution, y_resolution, target_extent_ul_lr, output_epsg, criteria_band, criteria_mode, n_cores, max_distance, log_file, log_level):
    """
    Build a mosaic from the input file.

    Args:
        output_file (str): Path to the output file.
        input_file_list (str): Path to the input file.
        ignore_file_list (str): Path to a list of files (subset of input_file_list) to be ignored during the mosaic.
        x_resolution (float): X resolution of the output mosaic.
        y_resolution (float): Y resolution of the output mosaic.
        target_extent_ul_lr ((float, float, float, float)): Target extent for the mosaic in the format (ul_x, ul_y, lr_x, lr_y).
        output_epsg (str): EPSG code for the output projection.
        criteria_band (int): Band to use for the criteria.
        criteria_mode (str): Mode to use for the criteria.
        n_cores (int): Number of cores to use for processing.
        max_distance (float): Maximum distance, in output CRS units, to solve for
        log_file (str): Path to the log file.
        log_level (str): Logging verbosity.
    """

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s || %(filename)s:%(funcName)s() | %(message)s",
        level=log_level,
        filename=log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    logging.debug(f"Building Mosaic from {input_file_list}")
    if ignore_file_list is not None:
        logging.debug(f"Excluding files from {ignore_file_list}")
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
        input_files = [x.strip() for x in open(input_file_list, 'r').readlines()]

    ignore_files = []
    if ignore_file_list is not None:
        ignore_files = [x.strip() for x in open(ignore_file_list, 'r').readlines()]

    gproj = osr.SpatialReference()
    gproj.ImportFromEPSG(int(output_epsg))
    wkt = gproj.ExportToWkt()
    proj = pyproj.Proj(f"epsg:{output_epsg}")

    if target_extent_ul_lr:
        ul_lr = target_extent_ul_lr # in output epsg projection
    else:
        # Always gets this in 4326
        ul_lr = get_ul_lr_from_files(input_files, get_resolution=False)
        # convert to output epsg
        ul = proj(ul_lr[0], ul_lr[1])
        lr = proj(ul_lr[2], ul_lr[3])
        ul_lr = [ul[0], ul[1], lr[0], lr[1]]
    
    if str(output_epsg)[0] == "4" and x_resolution > 1:
        raise ValueError(f"x_resolution is {x_resolution} (indicating meters), and EPSG is {output_epsg}.  Smells like lat/lon and UTM mismatch.  Terminating.")

    logging.info("Bounding box (ul_lr): " + str(ul_lr)) 

    trans = [ul_lr[0] - x_resolution/2., x_resolution, 0, 
             ul_lr[1] - y_resolution/2., 0, y_resolution]
    meta = spec_io.GenericGeoMetadata(['GLT X', 'GLT Y', 'File Index', 'OBS val'], 
                                      projection=wkt, 
                                      geotransform=trans, 
                                      pre_orthod=True, 
                                      nodata_value=0)

    glt = np.zeros(( int(np.ceil((ul_lr[3] - ul_lr[1]) / y_resolution)), 
                     int(np.ceil((ul_lr[2] - ul_lr[0]) / x_resolution)),
                     3), dtype=np.int32)
    criteria = np.zeros((glt.shape[0], glt.shape[1]), dtype=np.float32)
    criteria[...] = np.nan

    y_grid_steps = np.arange(ul_lr[1], ul_lr[3] - trans[5]*0.01,trans[5])
    x_grid_steps = np.arange(ul_lr[0], ul_lr[2] - trans[1]*0.01,trans[1])
    y_grid, x_grid = np.meshgrid(y_grid_steps, 
                                 x_grid_steps,
                                 indexing='ij')
    
    for _file, file in enumerate(tqdm(input_files, desc="Calculating GLT, File:", unit="files", ncols=80)):
        if file in ignore_files:
            logging.debug(f'{file} Ignored')
            continue

        local_meta, obs = spec_io.load_data(file.strip(), lazy=True, load_glt=False, load_loc=True)
        loc = np.stack(proj(local_meta.loc[...,0],local_meta.loc[...,1]),axis=-1)

        sub_glt, sub_glt_insert_idx = find_subgrid_locations(y_grid, x_grid, loc[...,1], loc[...,0], n_workers=n_cores, max_distance=max_distance)  

        if sub_glt is None:
            logging.debug(f'{file} OOB')
            continue

        remove_negatives(sub_glt, clean_contiguous=True)

        if criteria_band is not None:
            raw_ob = obs[:,:,criteria_band]
            ob = raw_ob[np.abs(sub_glt[...,1])-1, np.abs(sub_glt[...,0])-1]
            existing_crit = criteria[sub_glt_insert_idx[...,0], sub_glt_insert_idx[...,1]]
            valid =  np.logical_and(sub_glt[...,0] != meta.nodata_value, ob != local_meta.nodata_value)

            if criteria_mode == "min":
                crit_mask = np.logical_and(ob < existing_crit, valid)
            elif criteria_mode == "max":
                crit_mask = np.logical_and(ob > existing_crit, valid)
            else:
                raise ValueError(f"Invalid criteria_mode: {criteria_mode}")
            
            # In any mode, if there were no previous data, that counts too
            crit_mask = np.logical_or(crit_mask, np.logical_and(valid, np.isnan(existing_crit)))

            # only assign criteria band if used
            criteria[sub_glt_insert_idx[crit_mask,0], sub_glt_insert_idx[crit_mask,1]] = ob[crit_mask]
        else:
            crit_mask = sub_glt[...,0] != 0 

        glt[sub_glt_insert_idx[crit_mask,0], sub_glt_insert_idx[crit_mask,1], :2] = sub_glt[crit_mask,:]
        glt[sub_glt_insert_idx[crit_mask,0], sub_glt_insert_idx[crit_mask,1], 2] = _file + 1

    logging.info("Cleaning GLT")
    remove_negatives(glt, clean_contiguous=True)

    spec_io.write_cog(output_file, glt, meta, nodata_value=0)


@click.command()
@click.argument('glt_file', type=click.Path(exists=True))
@click.argument('raw_files', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--nodata_value', type=float, default=-9999)
@click.option('--bands', default=None, type=int, multiple=True)
@click.option('--output_format', default='tif', type=str, help="Output format")
@click.option('--glt_nodata_value', default=None, type=int, help="GLT Nodata Value")
def apply_glt(glt_file, raw_files, output_file, nodata_value, bands, output_format, glt_nodata_value):
    """
    Apply the GLT to the input files.

    Args:
        glt_file (str): Path to the GLT file.
        raw_files (str): Path to the raw files.
        output_file (str): Path to the output file.
        nodata_value (float): Nodata value for the output.
        bands (int): Bands to use for the output (None = all)
        glt_nodata_value (int): Override the nodata value in the GLT file; used to support legacy files only, generally should be ignored
    """
    glt_meta, glt = spec_io.load_data(glt_file, lazy=False)
    if glt_nodata_value is not None:
        glt_meta.nodata_value = glt_nodata_value
    glt = glt.astype(np.int32) # make sure we're not in legacy uint format
    glt_nonblank = glt[...,0] != glt_meta.nodata_value
    glt[...,:2] = np.abs(glt[...,:2])
    if glt_meta.nodata_value == 0:
        glt[...,:3] -= 1

    if raw_files.endswith(".txt"):
        input_files = open(raw_files, 'r').readlines()
    else:
        input_files = [raw_files]
        if glt.shape[-1] == 2:
            glt = np.append(glt, np.zeros((glt.shape[0],glt.shape[1],1),dtype=np.int32),axis=2)

    outdata = None
    for _file, file in enumerate(tqdm(input_files, ncols=80, desc="Apply GLT, File:", unit="files")):
        if np.any(glt[...,2] == _file):
            valid_glt = glt[...,2] == _file

            meta, dat = spec_io.load_data(file.strip(), lazy=True, load_glt=False)
            if bands is None or len(bands) == 0:
                bands = np.arange(dat.shape[2])
            dat = dat[...,bands]

            if outdata is None:
                outdata = np.zeros((glt.shape[0], glt.shape[1], dat.shape[2]), dtype=dat.dtype) + nodata_value

            outdata[valid_glt, :] = dat[glt[valid_glt, 1], glt[valid_glt, 0], :]

    if outdata is not None:
        logging.info(f'Writing: {output_file}')
        output_meta = deepcopy(meta)
        output_meta.projection = glt_meta.projection
        output_meta.geotransform = glt_meta.geotransform
        if output_format == 'tif':
            spec_io.write_cog(output_file, outdata, output_meta, nodata_value=nodata_value)
        elif output_format == 'envi':
            spec_io.create_envi_file(output_file, outdata.shape, output_meta, outdata.dtype)
            spec_io.write_bil_chunk(outdata.transpose((0,2,1)), output_file, 0, (glt.shape[0], outdata.shape[-1], glt.shape[1]) )
        else:
            logging.error('Unsupported output file time')
    else:
        logging.info('No data found; skipping output file write')


@click.group()
def cli():
    pass


cli.add_command(build_obs_nc)
cli.add_command(apply_glt)
cli.add_command(stack_glts)
cli.add_command(urban_mask)
cli.add_command(coastal_mask)


if __name__ == '__main__':
    cli()
    #build_mosaic_test()