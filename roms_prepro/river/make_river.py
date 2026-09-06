import os
import numpy as np
import netCDF4 as nc
from datetime import datetime


SEA_SALT = 30.0

# Day-of-year of the 12 monthly climatological mid-points (non-leap year)
_MONTH_MID_DAYS = np.array([
    (datetime(2000, m, 15) - datetime(2000, 1, 1)).days for m in range(1, 13)
], dtype=float)


def create_river_file(roms_grid_file, out_file,
                      rivers,
                      time_start=None,
                      time_end=None,
                      n_times=12,
                      time_units='days since 2000-01-01'):
    """
    Write a ROMS river forcing file from monthly climatological discharge.

    Two time-axis modes:
      * Default (time_start is None): 12 monthly records at the mid-month
        days of year 2000 with ``cycle_length = 365.25`` — ROMS cycles the
        climatology internally.
      * time_start/time_end given (ISO strings): the monthly cycle is
        interpolated (with periodic wrap) onto a daily axis covering the
        requested interval; ``n_times`` is ignored.

    Each river dict supports keys:
      I, J (grid indices, required), name, discharge (12 monthly values or
      scalar), salt / salinity (scalar or 12 values), temperature (12 values),
      direction (0 = inflow along +XI, 1 = along +ETA), scale, Vshape_factor.
    """
    g = nc.Dataset(roms_grid_file)
    lon_rho = g.variables['lon_rho'][:]
    lat_rho = g.variables['lat_rho'][:]
    mask_rho = g.variables['mask_rho'][:]
    Lp, Mp = lon_rho.shape
    if 'Cs_r' in g.variables and g.variables['Cs_r'][:].size > 0:
        s_rho = g.variables['Cs_r'][:]
    else:
        s_rho = np.linspace(-0.99, -0.01, 30)
    g.close()

    Nr = len(s_rho)
    N = len(rivers)

    # --- time axis (day offsets relative to the units epoch) ---
    if time_start is not None and time_end is not None:
        t0 = datetime.fromisoformat(time_start)
        t1 = datetime.fromisoformat(time_end)
        base = datetime.fromisoformat(time_units.split('since')[-1].strip())
        times = np.arange((t1 - t0).days + 1, dtype=float) + (t0 - base).days
        climatology = False
    else:
        if n_times == 12:
            times = _MONTH_MID_DAYS.copy()
        else:
            times = np.arange(n_times).astype('f8')
        climatology = True

    def _to_time_axis(q):
        """Project a monthly series (or scalar) onto the output time axis."""
        q = np.asarray(q, dtype=float)
        if q.ndim == 0:
            return np.full(len(times), float(q))
        if len(q) == 1:
            return np.full(len(times), q[0])
        # periodic linear interpolation of the monthly cycle
        x_ext = np.concatenate([_MONTH_MID_DAYS - 365.25, _MONTH_MID_DAYS,
                                _MONTH_MID_DAYS + 365.25])
        y_ext = np.concatenate([q, q, q])
        return np.interp(times, x_ext, y_ext)

    ds = nc.Dataset(out_file, 'w', format='NETCDF3_64BIT')
    ds.title = 'ROMS river forcing'
    ds.created = datetime.now().isoformat()

    r_dim = ds.createDimension('river', N)
    t_dim = ds.createDimension('river_time', len(times))
    s_dim = ds.createDimension('s_rho', Nr)
    n_dim = ds.createDimension('river_name_strlen', 32)

    v = ds.createVariable('river_time', 'f8', ('river_time',))
    v.units = time_units
    v.calendar = 'gregorian'
    v.cycle_length = 365.25 if climatology else 0
    v[:] = times

    # --- ROMS required river variables ---
    v = ds.createVariable('river_flag', 'i4', ('river',))
    v.long_name = 'river flag (1=active point source)'
    v[:] = np.ones(N, dtype=int)

    v = ds.createVariable('river_Xposition', 'i4', ('river',))
    v.long_name = 'river XI-position at RHO-points'
    v[:] = np.array([r['I'] for r in rivers], dtype=int)

    v = ds.createVariable('river_Eposition', 'i4', ('river',))
    v.long_name = 'river ETA-position at RHO-points'
    v[:] = np.array([r['J'] for r in rivers], dtype=int)

    v = ds.createVariable('river_Udirection', 'f4', ('river',))
    v.long_name = 'river U-momentum direction (+1 = +XI, -1 = -XI, 0 = none)'
    udir = [1.0 if r.get('direction', 0) == 0 else 0.0 for r in rivers]
    v[:] = np.array(udir, dtype='f4')

    v = ds.createVariable('river_Vdirection', 'f4', ('river',))
    v.long_name = 'river V-momentum direction (+1 = +ETA, -1 = -ETA, 0 = none)'
    vdir = [1.0 if r.get('direction', 0) == 1 else 0.0 for r in rivers]
    v[:] = np.array(vdir, dtype='f4')

    v = ds.createVariable('river_name', 'S1', ('river_name_strlen', 'river'))
    v.long_name = 'river name'
    for i, r in enumerate(rivers):
        name = str(r.get('name', f'river_{i}'))[:31].encode('ascii', 'replace')
        v[:len(name), i] = np.frombuffer(name, dtype='S1')

    v = ds.createVariable('river_Vshape', 'f8', ('s_rho', 'river'))
    v.long_name = 'river vertical profile, function of s_rho'
    shape = _vertical_shape(Nr)
    for i, r in enumerate(rivers):
        s_i = shape * r.get('Vshape_factor', 1.0)
        v[:, i] = s_i / s_i.sum()

    transport = np.zeros((len(times), N), dtype='f8')
    for i, r in enumerate(rivers):
        q = _to_time_axis(r.get('discharge', 0.0))
        transport[:, i] = q * r.get('scale', 1.0)

    v = ds.createVariable('river_transport', 'f8', ('river_time', 'river'))
    v.units = 'meter3 second-1'
    v.long_name = 'river discharge'
    v[:] = transport

    temp = np.zeros((len(times), N, Nr), dtype='f8')
    salt = np.zeros((len(times), N, Nr), dtype='f8')
    for i, r in enumerate(rivers):
        t_prof = r.get('temperature', None)
        if t_prof is None:
            t_prof = _seasonal_temperature(12)
        s_prof = r.get('salinity', None)
        if s_prof is None:
            s_prof = r.get('salt', SEA_SALT)
        t_prof = _to_time_axis(t_prof)
        s_prof = _to_time_axis(s_prof)
        for k in range(Nr):
            temp[:, i, k] = t_prof[:len(times)]
            salt[:, i, k] = s_prof[:len(times)]

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


def find_river_mouth(lon_rho, lat_rho, mask_rho, river_lon, river_lat,
                     max_dist_deg=5.0):
    """Locate the nearest wet grid cell to (river_lon, river_lat).

    Prints a warning when the nearest wet cell is farther than
    ``max_dist_deg`` degrees — typically means the coordinates are wrong
    or the river lies outside the grid.
    """
    dist = np.hypot(lon_rho - river_lon, lat_rho - river_lat)
    dist = np.where(mask_rho > 0, dist, np.inf)
    j, i = np.unravel_index(np.argmin(dist), mask_rho.shape)
    d_min = float(dist[j, i])
    if not np.isfinite(d_min):
        print(f'WARNING: no wet cell found for river at '
              f'({river_lon}, {river_lat}) — check the grid mask.')
    elif d_min > max_dist_deg:
        print(f'WARNING: river mouth at ({river_lon}, {river_lat}) is '
              f'{d_min:.1f} deg from the nearest wet cell (eta={j}, xi={i}) '
              f'— check the coordinates / grid coverage.')
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