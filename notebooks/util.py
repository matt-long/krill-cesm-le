import numpy as np
import xarray as xr
import pandas as pd

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

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320:
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
    yc = np.broadcast_to(lat[:, None], (ny, nx))
    xc = np.broadcast_to(lon[None, :], (ny, nx))

    # generate arrays of cell vertices
    yv = np.stack((yc-dy/2., yc-dy/2., yc+dy/2., yc+dy/2.), axis=2)
    xv = np.stack((xc-dx/2., xc+dx/2., xc+dx/2., xc-dx/2.), axis=2)

    # compute chord lengths
    y0 = np.sin(yv[:, :, 0]*deg2rad) # south
    y1 = np.sin(yv[:, :, 3]*deg2rad) # north
    x0 = xv[:, :, 0] * deg2rad         # west
    x1 = xv[:, :, 1] * deg2rad         # east
    area = (y1 - y0) * (x1 - x0) * Re**2

    if check_total:
        np.testing.assert_approx_equal(np.sum(area), 4.*np.pi*Re**2.)
    
    ds['area'] = xr.DataArray(area, dims=(lat_name, lon_name), 
                              attrs={'units': 'm^2', 'long_name': 'area'})  
