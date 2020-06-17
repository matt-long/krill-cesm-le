import os
import shutil

from datetime import datetime

import scipy.sparse as sps
import numpy as np
import dask
import xarray as xr

import pandas as pd


project_tmpdir = '/glade/p/cgd/oce/projects/krill-cesm-le/data'


def write_ds_out(dso, file_out):
    file_out = os.path.realpath(file_out)

    os.makedirs(os.path.dirname(file_out), exist_ok=True)

    if os.path.exists(file_out):
        shutil.rmtree(file_out)
    print('-'*30)
    print(f'Writing {file_out}')
    dso.info()
    print()
    dso.to_zarr(file_out);


def pop_add_cyclic(ds):
    
    nj = ds.TLAT.shape[0]
    ni = ds.TLONG.shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon)    
    lon  = np.concatenate((tlon, tlon + 360.), 1)
    lon = lon[:, xL:xR]
    
    if ni == 320 and nj == 384:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320 and nj == 384:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
            
    return dso


def orsi_fronts():
    """Return a dictionary of dataframes with lon, lat coords of fronts 
    from Orsi et al. 1995
    
    Orsi, A. H., T. Whitworth III and W. D. Nowlin, Jr. (1995). On the 
    meridional extent and fronts of the Antarctic Circumpolar Current, 
    Deep-Sea Research I, 42, 641-673.
    """
    
    fronts = {}
    for f in ['STF','SAF','PF','SACCF','SBDY']:
        fronts[f] = pd.read_table(f'data/orsi-fronts/{f.lower()}.txt', sep='\s+',
                                  names=['lon','lat'],
                                  comment='%')
    return fronts



def infer_lat_name(ds):
    lat_names = ['latitude', 'lat']
    for n in lat_names:
        if n in ds:
            return n
    raise ValueError('could not determine lat name')    


def infer_lon_name(ds):
    lon_names = ['longitude', 'lon']
    for n in lon_names:
        if n in ds:
            return n
    raise ValueError('could not determine lon name')    


def compute_grid_area(ds, dx=None, dy=None, check_total=True):
    Re = 6.37122e6 # m, radius of Earth
    deg2rad = np.pi/180.

    lon_name = infer_lon_name(ds)
    lon = ds[lon_name].values
    
    lat_name = infer_lat_name(ds)        
    lat = ds[lat_name].values
    
    if dx is None:
        dx = np.diff(lon)
        np.testing.assert_almost_equal(dx, dx[0])
        dx = np.float(dx[0])
        
    if dy is None:        
        dy = np.diff(lat)
        np.testing.assert_almost_equal(dy, dy[0])
        dy = np.float(dy[0])
        
    ny = lat.shape[0]
    nx = lon.shape[0]

    # generated 2D arrays of cell centers
    y_center = np.broadcast_to(lat[:, None], (ny, nx))
    x_center = np.broadcast_to(lon[None, :], (ny, nx))

    # compute corner points
    y_corner = np.stack((y_center - dy / 2.,  # SW
                         y_center - dy / 2.,  # SE
                         y_center + dy / 2.,  # NE
                         y_center + dy / 2.), # NW
                        axis=2)

    x_corner = np.stack((x_center - dx / 2.,  # SW
                         x_center + dx / 2.,  # SE
                         x_center + dx / 2.,  # NE
                         x_center - dx / 2.), # NW
                        axis=2)
    
    # compute chord lengths
    y0 = np.sin(y_corner[:, :, 0] * np.pi / 180) # south
    y1 = np.sin(y_corner[:, :, 3] * np.pi / 180) # north
    x0 = x_corner[:, :, 0] * np.pi / 180         # west
    x1 = x_corner[:, :, 1] * np.pi / 180         # east
    area = (y1 - y0) * (x1 - x0) * Re**2
    
    if check_total:
        np.testing.assert_approx_equal(np.sum(area), 4 * np.pi * Re**2)
        
    ds['area'] = xr.DataArray(area, dims=(lat_name, lon_name), 
                              attrs={'units': 'm^2', 'long_name': 'area'})  

    
def ann_mean(ds, season=None, time_bnds_varname='time_bnds'):
    """Compute annual means, or optionally seasonal means"""
    ds = ds.copy(deep=True)

    group_by_year = 'time.year'
    rename = {'year': 'time'}
    
    ones = xr.full_like(ds.time, fill_value=1)
    
    if season is not None:
        season = season.upper()
        if season not in ['DJF', 'MAM', 'JJA', 'SON']:
            raise ValueError(f'unknown season: {season}')            

        ds = ds.where(ds['time.season'] == season)      
        ones = ones.where(ds['time.season'] == season) 
        ds['austral_year'] = xr.where(ds['time.month'] > 6, ds['time.year'] + 1, ds['time.year'])

        ds = ds.set_coords('austral_year')
        ones = ones.assign_coords({'austral_year': ds.austral_year})

        if season == 'DJF':
            group_by_year = 'austral_year'
            rename = {'austral_year': 'time'}

    time_wgt = ds[time_bnds_varname].diff(dim=ds[time_bnds_varname].dims[1])
    if time_wgt.dtype == '<m8[ns]':
        time_wgt = time_wgt / np.timedelta64(1, 'D')
    
    time_wgt_grouped = time_wgt.groupby(group_by_year, restore_coord_dims=False)
    time_wgt = time_wgt_grouped / time_wgt_grouped.sum(dim=xr.ALL_DIMS)
        
    nyr = len(time_wgt_grouped.groups)
         
    time_wgt = time_wgt.squeeze()

    np.testing.assert_almost_equal(time_wgt.groupby(group_by_year).sum(dim=xr.ALL_DIMS), 
                                   np.ones(nyr))

    nontime_vars = set([v for v in ds.variables if 'time' not in ds[v].dims]) - set(ds.coords)
    dsop = ds.set_coords(nontime_vars).drop(time_bnds_varname)
    
    ds_ann = (dsop * time_wgt).groupby(group_by_year, restore_coord_dims=False).sum(dim='time')
    count_ann = ones.groupby(group_by_year, restore_coord_dims=False).sum(dim='time')
    
    # copy attrs
    for v in ds_ann:
        ds_ann[v].attrs = ds[v].attrs

    # rename time
    ds_ann = ds_ann.reset_coords(nontime_vars).rename(rename)
    
    # eliminate partials
    if season is not None:
        ndx = (count_ann == 3).values
    else:
        ndx = (count_ann >= 8).values

    if not ndx.all():
        ds_ann = ds_ann.isel(time=ndx)

    return ds_ann


def linear_trend(da, dim='time'):
    da_chunk = da.chunk({dim: -1})
    trend = xr.apply_ufunc(calc_slope,
                           da_chunk,
                           vectorize=True,
                           input_core_dims=[[dim]],
                           output_core_dims=[[]],
                           output_dtypes=[np.float],
                           dask='parallelized')
    return trend


def calc_slope(y):
    """ufunc to be used by linear_trend"""
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def latlon_to_scrip(nx, ny, lon0=-180., grid_imask=None, file_out=None):
    """Generate a SCRIP grid file for a regular lat x lon grid.
    
    Parameters
    ----------
    
    nx : int
       Number of points in x (longitude).
    ny : int
       Number of points in y (latitude).
    lon0 : float, optional [default=-180]
       Longitude on lefthand grid boundary.
    grid_imask : array-like, optional [default=None]       
       If the value is set to 0 for a grid point, then that point is
       considered masked out and won't be used in the weights 
       generated by the application. 
    file_out : string, optional [default=None]
       File to which to write the grid.

    Returns
    -------
    
    ds : xarray.Dataset
       The grid file dataset.       
    """
    
    # compute coordinates of regular grid
    dx = 360. / nx
    dy = 180. / ny
    lat = np.arange(-90. + dy / 2., 90., dy)
    lon = np.arange(lon0 + dx / 2., lon0 + 360., dx)

    # make 2D
    y_center = np.broadcast_to(lat[:, None], (ny, nx))
    x_center = np.broadcast_to(lon[None, :], (ny, nx))

    # compute corner points: must be counterclockwise
    y_corner = np.stack((y_center - dy / 2.,  # SW
                         y_center - dy / 2.,  # SE
                         y_center + dy / 2.,  # NE
                         y_center + dy / 2.), # NW
                        axis=2)

    x_corner = np.stack((x_center - dx / 2.,  # SW
                         x_center + dx / 2.,  # SE
                         x_center + dx / 2.,  # NE
                         x_center - dx / 2.), # NW
                        axis=2)

    # compute area
    y0 = np.sin(y_corner[:, :, 0] * np.pi / 180.) # south
    y1 = np.sin(y_corner[:, :, 3] * np.pi / 180.) # north
    x0 = x_corner[:, :, 0] * np.pi / 180.         # west
    x1 = x_corner[:, :, 1] * np.pi / 180.         # east
    grid_area = (y1 - y0) * (x1 - x0)
    
    # sum of area should be equal to area of sphere
    np.testing.assert_allclose(grid_area.sum(), 4.*np.pi)
    
    # construct mask
    if grid_imask is None:
        grid_imask = np.ones((ny, nx), dtype=np.int32)
    
    # generate output dataset
    dso = xr.Dataset()    
    dso['grid_dims'] = xr.DataArray(np.array([nx, ny], dtype=np.int32), 
                                    dims=('grid_rank',)) 
    dso.grid_dims.encoding = {'dtype': np.int32}

    dso['grid_center_lat'] = xr.DataArray(y_center.reshape((-1,)), 
                                          dims=('grid_size'),
                                          attrs={'units': 'degrees'})

    dso['grid_center_lon'] = xr.DataArray(x_center.reshape((-1,)), 
                                          dims=('grid_size'),
                                          attrs={'units': 'degrees'})
    
    dso['grid_corner_lat'] = xr.DataArray(y_corner.reshape((-1, 4)), 
                                          dims=('grid_size', 'grid_corners'), 
                                          attrs={'units': 'degrees'})
    dso['grid_corner_lon'] = xr.DataArray(x_corner.reshape((-1, 4)), 
                                      dims=('grid_size', 'grid_corners'), 
                                      attrs={'units': 'degrees'})    

    dso['grid_imask'] = xr.DataArray(grid_imask.reshape((-1,)), 
                                     dims=('grid_size'),
                                     attrs={'units': 'unitless'})
    dso.grid_imask.encoding = {'dtype': np.int32}
    
    dso['grid_area'] = xr.DataArray(grid_area.reshape((-1,)), 
                                     dims=('grid_size'),
                                     attrs={'units': 'radians^2',
                                            'long_name': 'area weights'})
    
    # force no '_FillValue' if not specified
    for v in dso.variables:
        if '_FillValue' not in dso[v].encoding:
            dso[v].encoding['_FillValue'] = None

    dso.attrs = {'title': f'{dy} x {dx} (lat x lon) grid',
                 'created_by': 'latlon_to_scrip',
                 'date_created': f'{datetime.now()}',
                 'conventions': 'SCRIP',
                }
            
    # write output file
    if file_out is not None:
        print(f'writing {file_out}')
        dso.to_netcdf(file_out)
        
    return dso


def compute_kgp(ds, length):
    """Compute Krill Growth Potential

    Natural growth rates in Antarctic krill (Euphausia superba)
    doi: 10.4319/lo.2006.51.2.0973
    A Atkinson, RS Shreeve, AG Hirst, P Rothery, GA Tarling
    Limnol Oceanogr, 2006

    """

    # specify params
    a = -0.066
    b = 0.002
    c = -0.000061
    d = 0.385
    e = 0.328
    f = 0.0078
    g = -0.0101

    # local pointers
    sst = ds.SST
    chl = ds.Chl_surf

    # compute terms and sum
    length_term = a + (b * length) + (c * length**2)
    chl_term = (d * (chl / (e + chl)))
    sst_term = (f * sst) + (g * sst**2)

    kgp = length_term + chl_term + sst_term
    kgp.name = 'KGP'

    # mask based on SST range
    kgp = kgp.where((-1. <= sst) & (sst <= 5.)).fillna(0.).where(ds.KMT > 0)

    # add coordinates
    kgp = kgp.assign_coords({'length': length})
    kgp = kgp.assign_coords({'TLONG': ds.TLONG, 'TLAT': ds.TLAT})

    # add attrs
    kgp.attrs = {'units': 'mm d$^{-1}$', 'long_name': 'Daily growth rate'}
    ds['KGP'] = kgp
    return ds


def esmf_apply_weights(weights, indata, shape_in, shape_out):
        '''
        Apply regridding weights to data.
        Parameters
        ----------
        A : scipy sparse COO matrix
        indata : numpy array of shape ``(..., n_lat, n_lon)`` or ``(..., n_y, n_x)``.
            Should be C-ordered. Will be then tranposed to F-ordered.
        shape_in, shape_out : tuple of two integers
            Input/output data shape for unflatten operation.
            For rectilinear grid, it is just ``(n_lat, n_lon)``.
        Returns
        -------
        outdata : numpy array of shape ``(..., shape_out[0], shape_out[1])``.
            Extra dimensions are the same as `indata`.
            If input data is C-ordered, output will also be C-ordered.
        '''



        # COO matrix is fast with F-ordered array but slow with C-array, so we
        # take in a C-ordered and then transpose)
        # (CSR or CRS matrix is fast with C-ordered array but slow with F-array)
        if not indata.flags['C_CONTIGUOUS']:
            warnings.warn("Input array is not C_CONTIGUOUS. "
                          "Will affect performance.")

        # get input shape information
        shape_horiz = indata.shape[-2:]
        extra_shape = indata.shape[0:-2]

        assert shape_horiz == shape_in, (
            'The horizontal shape of input data is {}, different from that of'
            'the regridder {}!'.format(shape_horiz, shape_in)
            )

        assert shape_in[0] * shape_in[1] == weights.shape[1], (
            "ny_in * nx_in should equal to weights.shape[1]"
        )

        assert shape_out[0] * shape_out[1] == weights.shape[0], (
            "ny_out * nx_out should equal to weights.shape[0]"
        )

        # use flattened array for dot operation
        indata_flat = indata.reshape(-1, shape_in[0]*shape_in[1])
        outdata_flat = weights.dot(indata_flat.T).T

        # unflattened output array
        outdata = outdata_flat.reshape(
            [*extra_shape, shape_out[0], shape_out[1]])
        return outdata
    
class regridder(object):
    """simple class to enable regridding"""
    
    def __init__(self, src_grid_file, dst_grid_file, weight_file):
        
        # TODO: do I actually need the grid files here?
        #       shouldn't all the information be in the weight file?
        self.src_grid_file = src_grid_file
        self.dst_grid_file = dst_grid_file
        
        with xr.open_dataset(src_grid_file) as src:
            self.dims_src = tuple(src.grid_dims.values[::-1])
    
        with xr.open_dataset(dst_grid_file) as dst:
            self.dims_dst = tuple(dst.grid_dims.values[::-1])
            self.mask_dst = dst.grid_imask.values.reshape(self.dims_dst).T

        n_dst = np.prod(self.dims_dst)
        n_src = np.prod(self.dims_src)
        print(f'source grid dims: {self.dims_src}')
        print(f'destination grid dims: {self.dims_dst}')

        with xr.open_dataset(weight_file) as mf:
            row = mf.row.values - 1
            col = mf.col.values - 1
            S = mf.S.values
        self.weights = sps.coo_matrix((S, (row, col)), shape=[n_dst, n_src])

    def __repr__(self):
        return (
            f'regridder {os.path.basename(self.src_grid_file)} --> {os.path.basename(self.dst_grid_file)}'
        )
    
    def regrid_dataarray(self, da_in, renormalize=True, apply_mask=True):
        """regrid DataArray"""
        # Pull data, dims and coords from incoming DataArray
        data_src = da_in.data
        non_lateral_dims = da_in.dims[:-2]
        copy_coords = {d: da_in.coords[d] for d in non_lateral_dims if d in da_in.coords}

        # If renormalize == True, remap a field of ones
        if renormalize:
            ones_src = np.where(np.isnan(data_src), 0.0, 1.0)
            data_src = np.where(np.isnan(data_src), 0.0, data_src)

        # remap the field
        data_dst = esmf_apply_weights(
            self.weights, data_src, self.dims_src, self.dims_dst
        )

        # Renormalize to include non-missing data_src
        # TODO: it would be nice to include a threshold here,
        #       the user could specify a fraction of mapped points, 
        #       below which the value yields missing in the data_dst
        if renormalize:
            old_err_settings = np.seterr(invalid='ignore')
            ones_dst = esmf_apply_weights(
                self.weights, ones_src, self.dims_src, self.dims_dst
            )
            ones_dst = np.where(ones_dst > 0.0, ones_dst, np.nan)
            data_dst = data_dst / ones_dst
            data_dst = np.where(ones_dst > 0.0, data_dst, np.nan)
            np.seterr(**old_err_settings)

        # reform into xarray.DataArray
        da_out = xr.DataArray(
            data_dst, name=da_in.name, dims=da_in.dims, attrs=da_in.attrs, coords=copy_coords
        )

        # Apply a missing-values mask
        if apply_mask:
            da_out = da_out.where(self.mask_dst.T)

        return da_out