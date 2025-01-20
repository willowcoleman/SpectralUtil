from spectral.io import envi
import netCDF4 as nc
import os
import numpy as np

class SpectralMetadata:
    def __init__(self, wavelengths, fwhm, geotransform=None, projection=None, glt=None, pre_orthod=False, nodata_value=None):
        """
        Initializes the SpectralMetadata object.

        Args:
            wavelengths (numpy.ndarray): Array of wavelength values.
            fwhm (numpy.ndarray): Array of full-width half-maximum values.
            geotransform (tuple, optional): GDAL-style geotransform array. Defaults to None.
            projection (str, optional): Projection string. Defaults to None.
            glt (numpy.ndarray, optional): 3d array of x and y indices. Defaults to None.
            pre_orthod (bool, optional): Whether the data is already orthod. Defaults to False.
            nodata_value (int, optional): The nodata value. Defaults to None.
        """
        self.wavelengths = wavelengths
        self.wl = wavelengths
        self.fwhm = fwhm
        self.geotransform = geotransform
        self.projection = projection
        self.glt = glt
        self.pre_orthod = False
        self.nodata_value = nodata_value

        if pre_orthod:
            self.orthoable = False
        elif self.glt is None:
            self.orthoable = False
        else:
            self.orthoable = True

    def wl_index(self, wl, buffer=None):
        """
        Finds the index of the wavelength closest to the given wavelength.

        Args:
            wl (float): The wavelength to find the index for.
            buffer (float, optional): The spectral range around the center wavelength to include. Defaults to None.

        Returns:
            int or numpy.ndarray: The index of the closest wavelength if buffer is None or 0. 
                      If buffer is provided, returns an array of indices within the buffer range.
        """
        if buffer is None or buffer == 0:
            return np.argmin(np.abs(self.wl - wl))
        else:
            return np.where(np.logical_and(self.wl >= wl - buffer, self.wl <= wl + buffer))


def load_spectral(input_file, lazy=True, load_glt=False):
    """
    Loads a spectral file and extracts the spectral metadata and data.

    Args:
        input_file (str): Path to the input spectral file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.
        load_glt (bool, optional): If True, loads the glt for orthoing. Defaults to False.

    Raises:
        ValueError: If the file type is unknown.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    if input_file.endswith(('.hdr', '.dat', '.img')) or '.' not in os.path.basename(input_file):
        return open_envi(input_file, lazy=True)
    elif input_file.endswith('.nc'):
        return open_netcdf(input_file, lazy=True, load_glt=load_glt)
    else:
        raise ValueError(f'Unknown file type for {input_file}')

def ortho_data(data, glt, glt_nodata=0):
    """
    Orthorectifies the data using the provided ground control points.

    Args:
        data (numpy.ndarray): The spectral data to orthorectify, with shape (rows, cols, bands).
        glt (numpy.ndarray): The ground control points, with shape (rows, cols, 2).

    Returns:
        numpy.ndarray: The orthorectified data.
    """
    outdata = np.zeros((glt.shape[0], glt.shape[1], data.shape[2]), dtype=data.dtype)
    valid_glt = np.all(glt != glt_nodata, axis=-1)
    if glt_nodata == 0:
        glt[valid_glt] -= 1
    outdata[valid_glt, :] = data[np.abs(glt[valid_glt, 1]), np.abs(glt[valid_glt, 0]), :]
    return outdata



def write_cog(output_file, data, meta, ortho=True, nodata_value=-9999):
    """
    Writes a Cloud Optimized GeoTIFF (COG) file.

    Args:
        output_file (str): Path to the output COG file.
        data (numpy.ndarray): The spectral data to be written, with shape (rows, cols, bands).
        meta (SpectralMetadata): The spectral metadata containing geotransform and projection information.
        ortho (bool, optional): Whether to ortho the data.  Only relevant if the data isn't natively orthod. Defaults to True.
        nodata_value (, optional): The nodata value to use. Defaults to -9999.
    """
    from osgeo import gdal
    driver = gdal.GetDriverByName('MEM')

    if ortho and meta.orthoable:
        od = ortho_data(data, meta.glt)
    else:
        od = data

    ds = driver.Create('', od.shape[1], od.shape[0], od.shape[2], gdal.GDT_Float32)
    if meta.geotransform is not None:
        ds.SetGeoTransform(meta.geotransform)
    if meta.projection is not None:
        ds.SetProjection(meta.projection)
    for i in range(od.shape[2]):
        ds.GetRasterBand(i+1).WriteArray(od[:, :, i])
        ds.GetRasterBand(i+1).SetNoDataValue(nodata_value)
    ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64, 128])
    
    tiff_driver = gdal.GetDriverByName('GTiff')
    output_dataset = tiff_driver.CreateCopy(output_file, ds, options=['COMPRESS=LZW', 'COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    ds = None
    output_dataset = None


def envi_header(input_file):
    """
    Generates the corresponding ENVI header file path for a given input file.

    Args:
        input_file (str): Path to the input file.

    Returns:
        str: Path to the corresponding ENVI header file.
    """
    return os.path.splitext(input_file)[0] + '.hdr'


def open_envi(input_file, lazy=True):
    """
    Opens an ENVI file and extracts the spectral metadata and data.

    Args:
        input_file (str): Path to the ENVI file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The data, either as a lazy-loaded memory map or a fully loaded numpy array.
    """
    header = envi_header(input_file)
    ds = envi.open(header, input_file)
    wl = np.array([float(x) for x in ds.metadata['wavelength']])
    fwhm = np.array([float(x) for x in ds.metadata['fwhm']])
    nodata_value = float(ds.metadata['data ignore value'])
    meta = SpectralMetadata(wl, fwhm, nodata_value=nodata_value)

    if lazy:
        rfl = ds.open_memmap(interleave='bip', writable=False)
    else:
        rfl = ds.open_memmap(interleave='bip').copy()
    
    return meta, rfl
    

def open_netcdf(input_file, lazy=True, load_glt=False):
    """
    Opens a NetCDF file and extracts the spectral metadata and data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.
        load_glt (bool, optional): If True, loads the glt for orthoing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    if 'EMIT' in input_file and 'RAD' in input_file:
        return open_emit_rdn(input_file, lazy=lazy, load_glt=load_glt)
    elif 'AV3' in input_file and 'RFL' in input_file:
        return open_av3_rfl(input_file, lazy=lazy)


def open_emit_rdn(input_file, lazy=True, load_glt=False):
    """
    Opens an EMIT radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the radiance data lazily. Defaults to True.
        load_glt (bool, optional): If True, loads the glt for orthoing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The radiance data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    ds = nc.Dataset(input_file)
    wl = ds['sensor_band_parameters']['wavelengths'][:]
    fwhm = ds['sensor_band_parameters']['fwhm'][:]
    trans = ds.geotransform
    proj = ds.spatial_ref
    nodata_value = ds['radiance']._FillValue

    if lazy:
        rdn = ds['radiance']
    else:
        rdn = np.array(ds['radiance'][:])
    
    glt = None
    if load_glt:
        glt = np.stack([ds['location']['glt_x'][:],ds['location']['glt_y'][:]],axis=-1)

    meta = SpectralMetadata(wl, fwhm, trans, proj, glt, pre_orthod=False, nodata_value=nodata_value)

    return meta, rdn


def open_av3_rfl(input_file, lazy=True):
    """
    Opens an EMIT radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the radiance data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The radiance data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    ds = nc.Dataset(input_file)
    wl = ds['reflectance']['wavelength'][:]
    fwhm = ds['reflectance']['fwhm'][:]
    proj = ds.variables['transverse_mercator'].spatial_ref
    trans = [float(x) for x in ds.variables['transverse_mercator'].GeoTransform.split(' ')]
    nodata_value = ds['reflectance']['reflectance']._FillValue

    if lazy:
        # This is too bad....we're forced into inconsistent handling between AV3 and EMIT
        # need to consider some clever solutions.  In the meantime, this works, but is expensive
        rfl = np.transpose(ds['reflectance']['reflectance'], (1,2,0))
    else:
        rfl = np.array(ds['reflectance']['reflectance'][:])
    
    meta = SpectralMetadata(wl, fwhm, trans, proj, glt=None, pre_orthod=True, nodata_value=nodata_value)

    return meta, rfl