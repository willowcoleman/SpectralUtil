

# SpectralUtils

This is a package for basic manipulation of imaging spectroscopy data.  It is designed to accomodate data from a variety of instruments, abstracting out the
specifics of the file delivery.  Currently, the package supports data from the following instruments / product levels:


- [AVIRIS-3 L2A Reflectance](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2357)
- [EMIT L1B Radiance](https://lpdaac.usgs.gov/products/emitl1bradv001/)
- Any data in ENVI format



## Utilities
A series of utility scripts are provided to help with common tasks, such as:

#### Standard RGB, with stretching
```
python spectral_util.py rgb EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_rgb.tif
```

#### Standard RGB, custom wavelengths
```
python spectral_util.py rgb EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_rgb.tif --red_wl 2360 --green_wl 1800 --blue_wl 1000
```

### Spectral Indices
```
python spectral_util.py nbr EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_nbr.tif
python spectral_util.py ndvi EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_ndvi.tif
```