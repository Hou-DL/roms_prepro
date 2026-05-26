import os
import numpy as np
import netCDF4 as nc
from datetime import datetime


SEA_SALT = 30.0
RIVER_TEMP_SPRING = 15.0
RIVER_TEMP_SUMMER = 25.0
RIVER_TEMP_AUTUMN = 18.0
RIVER_TEMP_WINTER = 8.0


def create_river_file(roms_grid_file, out_file,
                      rivers,
                      time_start=None,
                      time_end=None,
                      n_times=12,
                      time_units='days since 2000-01-01'):
    g = nc.Dataset(roms_grid_file)
    lon_rho = g.variables['lon_rho'][:]
    lat_rho = g.variables['lat_rho'][:]
    mask_rho = g.variables['mask_rho'][:]
    Lp, Mp = lon_rho.shape
    if 'Cs_r' in g.variables and g.variables['Cs_r'][:].size > 0:
        s_rho = g.variables['Cs_r'][:]
        Vtransform = int(g.variables['Vtransform'][:]) if 'Vtransform' in g.variables else 2
    else:
        s_rho = np.linspace(-0.99, -0.01, 30)
        Vtransform = 2
    g.close()

    Nr = len(s_rho)
    N = len(rivers)

    if n_times == 12:
        times = np.array([
            (datetime(2000, m, 15) - datetime(2000, 1, 1)).total_seconds()
            for m in range(1, 13)
        ])
    else:
        times = np.arange(n_times).astype('f8')

    ds = nc.Dataset(out_file, 'w', format='NETCDF3_64BIT')
    ds.title = 'ROMS river forcing'
    ds.created = datetime.now().isoformat()

    r_dim = ds.createDimension('river', N)
    t_dim = ds.createDimension('river_time', len(times))
    s_dim = ds.createDimension('s_rho', Nr)

    v = ds.createVariable('river_time', 'f8', ('river_time',))
    v.units = time_units
    v.cycle_length = 365.25 if n_times == 12 else 0
    v[:] = times

    v = ds.createVariable('river', 'i4', ('river',))
    v[:] = np.arange(N)

    v = ds.createVariable('river_Xposition', 'i4', ('river',))
    v.long_name = 'river XI-position at RHO-points'
    v[:] = np.array([r['I'] for r in rivers], dtype=int)

    v = ds.createVariable('river_Eposition', 'i4', ('river',))
    v.long_name = 'river ETA-position at RHO-points'
    v[:] = np.array([r['J'] for r in rivers], dtype=int)

    v = ds.createVariable('river_direction', 'i4', ('river',))
    v.long_name = 'river flow direction: 0=along XI, 1=along ETA'
    v[:] = np.array([r.get('direction', 0) for r in rivers], dtype=int)

    v = ds.createVariable('river_scale', 'f8', ('river',))
    v.long_name = 'river discharge scale factor'
    v[:] = np.array([r.get('scale', 1.0) for r in rivers], dtype=float)

    v = ds.createVariable('river_Vshape', 'f8', ('s_rho', 'river'))
    v.long_name = 'river vertical profile, function of s_rho'
    shape = _vertical_shape(Nr)
    for i, r in enumerate(rivers):
        v[:, i] = shape * r.get('Vshape_factor', 1.0)

    transport = np.zeros((len(times), N), dtype='f8')
    for i, r in enumerate(rivers):
        q = r.get('discharge', None)
        if q is None:
            q = np.zeros(n_times)
        elif isinstance(q, (int, float)):
            q = np.ones(n_times) * q
        elif len(q) == 1:
            q = np.ones(n_times) * q[0]
        transport[:, i] = q[:n_times]

    v = ds.createVariable('river_transport', 'f8', ('river_time', 'river'))
    v.units = 'meter3 second-1'
    v.long_name = 'river discharge'
    v[:] = transport

    temp = np.zeros((len(times), N, Nr), dtype='f8')
    salt = np.zeros((len(times), N, Nr), dtype='f8')
    for i, r in enumerate(rivers):
        t_prof = r.get('temperature', None)
        s_prof = r.get('salinity', None)
        if t_prof is None:
            t_prof = _seasonal_temperature(n_times)
        if s_prof is None:
            s_prof = np.full(n_times, r.get('salt', SEA_SALT))
        for k in range(Nr):
            temp[:, i, k] = t_prof[:n_times]
            salt[:, i, k] = s_prof[:n_times]

    v = ds.createVariable('river_temp', 'f8', ('river_time', 'river', 's_rho'))
    v.units = 'Celsius'
    v.long_name = 'river temperature'
    v[:] = temp

    v = ds.createVariable('river_salt', 'f8', ('river_time', 'river', 's_rho'))
    v.units = 'PSU'
    v.long_name = 'river salinity'
    v[:] = salt

    ds.close()
    print(f'River forcing written: {out_file} ({N} rivers, {len(times)} times)')


def _vertical_shape(Nr):
    s = np.linspace(-1, 0, Nr)
    shape = s + 1.0
    shape = shape / shape.sum()
    return shape


def _seasonal_temperature(n_times):
    cycle = np.array([8.0, 8.0, 12.0, 16.0, 22.0, 26.0,
                      28.0, 28.0, 24.0, 18.0, 14.0, 10.0])
    if n_times == 12:
        return cycle
    return np.interp(np.linspace(1, 12, n_times), np.arange(1, 13), cycle)


def find_river_mouth(lon_rho, lat_rho, mask_rho, river_lon, river_lat):
    dist = np.hypot(lon_rho - river_lon, lat_rho - river_lat)
    dist = np.where(mask_rho > 0, dist, np.inf)
    j, i = np.unravel_index(np.argmin(dist), mask_rho.shape)
    return i, j


def yangtze_river(lon_rho, lat_rho, mask_rho,
                  discharge_m3s=31000, salt_psu=0.5):
    i, j = find_river_mouth(lon_rho, lat_rho, mask_rho, 121.5, 31.5)
    monthly = np.array([28000, 24000, 26000, 31000, 42000, 52000,
                        58000, 53000, 43000, 34000, 27000, 25000])
    monthly = monthly * discharge_m3s / monthly.mean()
    return {
        'name': 'Yangtze River',
        'I': i, 'J': j,
        'discharge': monthly,
        'salt': np.full(12, salt_psu),
        'direction': 0,
    }


def huanghe_river(lon_rho, lat_rho, mask_rho,
                  discharge_m3s=1500, salt_psu=1.0):
    i, j = find_river_mouth(lon_rho, lat_rho, mask_rho, 119.0, 37.5)
    monthly = np.array([900, 700, 600, 800, 1800, 2500,
                        2800, 2600, 2200, 1500, 1000, 800])
    monthly = monthly * discharge_m3s / monthly.mean()
    return {
        'name': 'Yellow River',
        'I': i, 'J': j,
        'discharge': monthly,
        'salt': np.full(12, salt_psu),
        'direction': 0,
    }