import numpy as np
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm


def era5_to_roms_forcing(
    roms_grid_file=None,
    era5_files=None,
    rh_files=None,
    out_file=None,
    start_date=None,
    end_date=None,
    time_ref='days since 1900-01-01 00:00:00',
    get_lwrad=True,
    get_swrad=True,
    get_rain=True,
    get_Tair=True,
    get_Pair=True,
    get_Qair=True,
    get_Wind=True,
    interp_to_grid=False,
):
    """
    Convert ERA5 data to ROMS forcing file.

    Two modes:
      - interp_to_grid=False (default): keep ERA5 original 1D lat/lon grid.
        ROMS interpolates internally via its regridding mechanism.
        roms_grid_file is NOT needed.
      - interp_to_grid=True: bilinear interpolation to ROMS curvilinear grid.
        Requires roms_grid_file with lon_rho/lat_rho/mask_rho.
    """
    era5_files = sorted(era5_files)
    if rh_files:
        rh_files = sorted(rh_files)

    # --- load grid (only for interp mode) ---
    if interp_to_grid:
        g = nc.Dataset(roms_grid_file)
        lon_rho = g.variables['lon_rho'][:]
        lat_rho = g.variables['lat_rho'][:]
        mask = g.variables['mask_rho'][:]
        g.close()
        Mp, Lp = lon_rho.shape

    # --- peek first ERA5 file for lat/lon/time ---
    ds0 = nc.Dataset(era5_files[0])
    era5_lon = ds0.variables['longitude'][:]
    era5_lat = ds0.variables['latitude'][:]
    era5_time_raw = ds0.variables['time'][:]
    era5_time_units = ds0.variables['time'].units
    ds0.close()

    # ERA5 latitude is descending; ROMS expects ascending
    if era5_lat[0] > era5_lat[-1]:
        era5_lat = era5_lat[::-1]
        lat_reversed = True
    else:
        lat_reversed = False

    nlon = len(era5_lon)
    nlat = len(era5_lat)

    if interp_to_grid:
        from scipy.interpolate import RegularGridInterpolator
        era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat)

        if rh_files:
            ds_rh0 = nc.Dataset(rh_files[0])
            rh_lon = ds_rh0.variables['longitude'][:]
            rh_lat = ds_rh0.variables['latitude'][:]
            ds_rh0.close()
            rh_lon_2d, rh_lat_2d = np.meshgrid(rh_lon, rh_lat)

    def _has_var(filename, varname):
        d = nc.Dataset(filename)
        has = varname in d.variables
        d.close()
        return has

    def _dt_from_era5(et):
        if 'days since' in era5_time_units.lower():
            return [datetime(1900, 1, 1) + np.timedelta64(int(t * 24), 'h') for t in et]
        else:
            return [datetime(1900, 1, 1) + np.timedelta64(int(t), 'h') for t in et]

    def _read_data(filename, varname, tidx):
        d = nc.Dataset(filename)
        data = d.variables[varname][tidx]
        d.close()
        return data

    # --- count total timesteps for progress bar ---
    total_timesteps = 0
    for ef in era5_files:
        ds = nc.Dataset(ef)
        et = ds.variables['time'][:]
        ds.close()
        fd = _dt_from_era5(et)
        sd_dt = datetime.fromisoformat(start_date.replace(' ', 'T')) if start_date else None
        ed_dt = datetime.fromisoformat(end_date.replace(' ', 'T')) if end_date else None
        total_timesteps += sum(1 for f in fd if (sd_dt is None or f >= sd_dt) and (ed_dt is None or f <= ed_dt))

    all_times = []
    all_data = {k: [] for k in ['lwrad', 'lwrad_down', 'swrad', 'rain',
                                  'Tair', 'Pair', 'Qair', 'Uwind', 'Vwind']}

    pbar = tqdm(total=total_timesteps, desc='ERA5 timesteps', unit='step')
    for fi, (ef, rf) in enumerate(zip(era5_files, rh_files if rh_files else [None] * len(era5_files))):
        ds = nc.Dataset(ef)
        et = ds.variables['time'][:]
        ds.close()
        file_dates = _dt_from_era5(et)

        sd_dt = datetime.fromisoformat(start_date.replace(' ', 'T')) if start_date else None
        ed_dt = datetime.fromisoformat(end_date.replace(' ', 'T')) if end_date else None

        for ti, dt_val in enumerate(file_dates):
            if sd_dt is not None and dt_val < sd_dt:
                continue
            if ed_dt is not None and dt_val > ed_dt:
                continue

            point_data = {}

            if get_lwrad:
                down = _read_data(ef, 'msdwlwrf', ti)
                net = _read_data(ef, 'msnlwrf', ti) if _has_var(ef, 'msnlwrf') else down * 0.9
                if interp_to_grid:
                    point_data['lwrad_down'] = _interp2d(era5_lon_2d, era5_lat_2d, down, lon_rho, lat_rho)
                    point_data['lwrad'] = _interp2d(era5_lon_2d, era5_lat_2d, net, lon_rho, lat_rho)
                else:
                    point_data['lwrad'] = net
                    point_data['lwrad_down'] = down

            if get_swrad:
                sw = _read_data(ef, 'msnswrf', ti)
                if interp_to_grid:
                    point_data['swrad'] = _interp2d(era5_lon_2d, era5_lat_2d, sw, lon_rho, lat_rho)
                else:
                    point_data['swrad'] = sw

            if get_rain:
                tp = _read_data(ef, 'tp', ti) * 1000.0 / 3600.0
                if interp_to_grid:
                    point_data['rain'] = _interp2d(era5_lon_2d, era5_lat_2d, tp, lon_rho, lat_rho)
                else:
                    point_data['rain'] = tp

            if get_Tair:
                t2m = _read_data(ef, 't2m', ti) - 273.15
                if interp_to_grid:
                    point_data['Tair'] = _interp2d(era5_lon_2d, era5_lat_2d, t2m, lon_rho, lat_rho)
                else:
                    point_data['Tair'] = t2m

            if get_Pair:
                msl = _read_data(ef, 'msl', ti) * 0.01
                if interp_to_grid:
                    point_data['Pair'] = _interp2d(era5_lon_2d, era5_lat_2d, msl, lon_rho, lat_rho)
                else:
                    point_data['Pair'] = msl

            if get_Qair:
                if rf:
                    rh = _read_data(rf, 'r', ti)
                else:
                    d2m = _read_data(ef, 'd2m', ti)
                    t2m_k = _read_data(ef, 't2m', ti)
                    rh = _relative_humidity_from_dewpoint(d2m, t2m_k)
                if interp_to_grid:
                    point_data['Qair'] = _interp2d(
                        rh_lon_2d if rf else era5_lon_2d,
                        rh_lat_2d if rf else era5_lat_2d,
                        rh, lon_rho, lat_rho
                    )
                else:
                    point_data['Qair'] = rh

            if get_Wind:
                u10 = _read_data(ef, 'u10', ti)
                v10 = _read_data(ef, 'v10', ti)
                if interp_to_grid:
                    point_data['Uwind'] = _interp2d(era5_lon_2d, era5_lat_2d, u10, lon_rho, lat_rho)
                    point_data['Vwind'] = _interp2d(era5_lon_2d, era5_lat_2d, v10, lon_rho, lat_rho)
                else:
                    point_data['Uwind'] = u10
                    point_data['Vwind'] = v10

            pbar.update(1)

            all_times.append(dt_val)
            for k in all_data:
                all_data[k].append(point_data.get(k, None))

    pbar.close()
    ntimes = len(all_times)
    if ntimes == 0:
        print('No data in date range.')
        return

    roms_times = _datetime_to_roms_time(all_times, time_ref)

    out = out_file or 'romsforc_era5.nc'
    print(f'Writing {out}...')

    if interp_to_grid:
        _write_interp_output(out, roms_times, ntimes, lon_rho, lat_rho,
                             time_ref, all_data,
                             get_lwrad, get_swrad, get_rain,
                             get_Tair, get_Pair, get_Qair, get_Wind)
    else:
        _write_raw_output(out, roms_times, ntimes, era5_lon, era5_lat,
                          lat_reversed, time_ref, all_data,
                          get_lwrad, get_swrad, get_rain,
                          get_Tair, get_Pair, get_Qair, get_Wind)

    print(f'Forcing file written: {out}  ({ntimes} timesteps)')


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_interp_output(out, roms_times, ntimes, lon_rho, lat_rho,
                         time_ref, all_data,
                         get_lwrad, get_swrad, get_rain,
                         get_Tair, get_Pair, get_Qair, get_Wind):
    """Write output interpolated to ROMS grid (xi_rho, eta_rho)."""
    nc_out = nc.Dataset(out, 'w', format='NETCDF3_64BIT')
    nc_out.type = 'bulk fluxes forcing file'
    nc_out.created = datetime.now().isoformat()

    Mp, Lp = lon_rho.shape
    nc_out.createDimension('xr', Mp)
    nc_out.createDimension('er', Lp)
    nc_out.createDimension('time', ntimes)

    lon_var = nc_out.createVariable('lon', 'f8', ('xr', 'er'))
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'
    lon_var[:] = lon_rho

    lat_var = nc_out.createVariable('lat', 'f8', ('xr', 'er'))
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'
    lat_var[:] = lat_rho

    tvar = nc_out.createVariable('ocean_time', 'f8', ('time',))
    tvar.long_name = 'atmospheric forcing time'
    tvar.units = time_ref
    tvar.field = 'time, scalar, series'
    tvar.calendar = 'gregorian'
    tvar[:] = roms_times

    def _write_var(name, long_name, units, data):
        tvd = nc_out.createDimension(f'{name}_time', ntimes)
        tvar_v = nc_out.createVariable(f'{name}_time', 'f8', (f'{name}_time',))
        tvar_v.long_name = f'{name}_time'
        tvar_v.units = time_ref
        tvar_v.field = f'{name}_time, scalar, series'
        tvar_v[:] = roms_times
        v = nc_out.createVariable(name, 'f8', (f'{name}_time', 'xr', 'er'))
        v.long_name = long_name
        v.units = units
        v.field = f'{name}, scalar, series'
        v.coordinates = 'lon lat'
        v.time = f'{name}_time'
        v[:] = data
        return v

    if get_lwrad:
        _write_var('lwrad', 'net solar longwave radiation', 'Watts meter-2',
                   np.array(all_data['lwrad']))
        _write_var('lwrad_down', 'downward solar longwave radiation', 'Watts meter-2',
                   np.array(all_data['lwrad_down']))

    if get_swrad:
        _write_var('swrad', 'net solar shortwave radiation', 'Watts meter-2',
                   np.array(all_data['swrad']))

    if get_rain:
        rain_arr = np.array(all_data['rain'])
        rain_arr = np.maximum(rain_arr, 0)
        _write_var('rain', 'rain fall rate', 'kilogram meter-2 second-1', rain_arr)

    if get_Tair:
        tair_arr = np.array(all_data['Tair'])
        tair_arr = np.clip(tair_arr, -40, 45)
        _write_var('Tair', 'surface air temperature', 'Celsius', tair_arr)

    if get_Pair:
        pair_arr = np.array(all_data['Pair'])
        pair_arr = np.clip(pair_arr, 950, 1080)
        _write_var('Pair', 'surface air pressure', 'millibar', pair_arr)

    if get_Qair:
        qair_arr = np.array(all_data['Qair'])
        qair_arr = np.clip(qair_arr, 1, 100)
        _write_var('Qair', 'surface air relative humidity', 'percentage', qair_arr)

    if get_Wind:
        uwind_arr = np.array(all_data['Uwind'])
        vwind_arr = np.array(all_data['Vwind'])
        uwind_arr = np.clip(uwind_arr, -50, 50)
        vwind_arr = np.clip(vwind_arr, -50, 50)
        _write_var('Uwind', 'surface u-wind component', 'meter second-1', uwind_arr)
        _write_var('Vwind', 'surface v-wind component', 'meter second-1', vwind_arr)

    nc_out.close()


def _write_raw_output(out, roms_times, ntimes, era5_lon, era5_lat,
                      lat_reversed, time_ref, all_data,
                      get_lwrad, get_swrad, get_rain,
                      get_Tair, get_Pair, get_Qair, get_Wind):
    """Write output on ERA5's native 1D lat/lon grid (for ROMS internal regridding)."""
    nlon = len(era5_lon)
    nlat = len(era5_lat)

    nc_out = nc.Dataset(out, 'w', format='NETCDF4')
    nc_out.type = 'bulk fluxes forcing file'
    nc_out.created = datetime.now().isoformat()

    nc_out.createDimension('lon', nlon)
    nc_out.createDimension('lat', nlat)
    nc_out.createDimension('time', ntimes)

    lon_var = nc_out.createVariable('lon', 'f8', ('lon',))
    lon_var.long_name = 'Longitude'
    lon_var.units = 'degree_east'
    lon_var.standard_name = 'longitude'
    lon_var[:] = era5_lon

    lat_var = nc_out.createVariable('lat', 'f8', ('lat',))
    lat_var.long_name = 'Latitude'
    lat_var.units = 'degree_north'
    lat_var.standard_name = 'latitude'
    lat_var[:] = era5_lat

    tvar = nc_out.createVariable('ocean_time', 'f8', ('time',))
    tvar.long_name = 'atmospheric forcing time'
    tvar.units = time_ref
    tvar.field = 'time, scalar, series'
    tvar.calendar = 'gregorian'
    tvar[:] = roms_times

    def _write_raw_var(name, long_name, units, data):
        tvd = nc_out.createDimension(f'{name}_time', ntimes)
        tvar_v = nc_out.createVariable(f'{name}_time', 'f8', (f'{name}_time',))
        tvar_v.long_name = f'{name}_time'
        tvar_v.units = time_ref
        tvar_v.field = f'{name}_time, scalar, series'
        tvar_v[:] = roms_times

        v = nc_out.createVariable(name, 'f4', (f'{name}_time', 'lat', 'lon'),
                                  zlib=True, complevel=4, shuffle=True)
        v.long_name = long_name
        v.units = units
        v.field = f'{name}, scalar, series'
        v.coordinates = 'lon lat'
        v.time = f'{name}_time'
        # store data with lat ascending (match how lat variable is written)
        if lat_reversed:
            v[:] = data[:, ::-1, :]
        else:
            v[:] = data
        return v

    if get_lwrad:
        _write_raw_var('lwrad', 'net longwave radiation', 'Watts meter-2',
                       np.array(all_data['lwrad']).astype('f4'))
        _write_raw_var('lwrad_down', 'downward longwave radiation', 'Watts meter-2',
                       np.array(all_data['lwrad_down']).astype('f4'))

    if get_swrad:
        _write_raw_var('swrad', 'net shortwave radiation', 'Watts meter-2',
                       np.array(all_data['swrad']).astype('f4'))

    if get_rain:
        rain_arr = np.array(all_data['rain']).astype('f4')
        rain_arr = np.maximum(rain_arr, 0)
        _write_raw_var('rain', 'rain fall rate', 'kilogram meter-2 second-1', rain_arr)

    if get_Tair:
        tair_arr = np.array(all_data['Tair']).astype('f4')
        tair_arr = np.clip(tair_arr, -40, 45)
        _write_raw_var('Tair', 'surface air temperature', 'Celsius', tair_arr)

    if get_Pair:
        pair_arr = np.array(all_data['Pair']).astype('f4')
        pair_arr = np.clip(pair_arr, 950, 1080)
        _write_raw_var('Pair', 'surface air pressure', 'millibar', pair_arr)

    if get_Qair:
        qair_arr = np.array(all_data['Qair']).astype('f4')
        qair_arr = np.clip(qair_arr, 1, 100)
        _write_raw_var('Qair', 'surface air relative humidity', 'percentage', qair_arr)

    if get_Wind:
        uwind_arr = np.array(all_data['Uwind']).astype('f4')
        vwind_arr = np.array(all_data['Vwind']).astype('f4')
        uwind_arr = np.clip(uwind_arr, -50, 50)
        vwind_arr = np.clip(vwind_arr, -50, 50)
        _write_raw_var('Uwind', 'surface u-wind component', 'meter second-1', uwind_arr)
        _write_raw_var('Vwind', 'surface v-wind component', 'meter second-1', vwind_arr)

    nc_out.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interp2d(src_lon_2d, src_lat_2d, src_data, dst_lon, dst_lat):
    from scipy.interpolate import RegularGridInterpolator
    src_lon_1d = np.unique(src_lon_2d)
    src_lat_1d = np.unique(src_lat_2d)

    si_lon = np.argsort(src_lon_1d)
    si_lat = np.argsort(src_lat_1d)
    src_lon_1d = src_lon_1d[si_lon]
    src_lat_1d = src_lat_1d[si_lat]

    src_data_s = np.take(src_data, si_lon, axis=1)
    src_data_s = np.take(src_data_s, si_lat, axis=0)
    src_data_s = src_data_s.T

    interp = RegularGridInterpolator(
        (src_lon_1d, src_lat_1d), src_data_s,
        method='linear', bounds_error=False, fill_value=np.nan
    )

    points = np.column_stack([dst_lon.ravel(), dst_lat.ravel()])
    result = interp(points).reshape(dst_lon.shape)

    nan_mask = np.isnan(result)
    if nan_mask.any():
        interp_nn = RegularGridInterpolator(
            (src_lon_1d, src_lat_1d), src_data_s,
            method='nearest', bounds_error=False, fill_value=np.nan
        )
        result[nan_mask] = interp_nn(points[nan_mask.reshape(-1,)])

    result = np.where(np.isnan(result), 0, result)
    return result


def _relative_humidity_from_dewpoint(d2m, t2m):
    es = 6.112 * np.exp(17.67 * (t2m - 273.15) / (t2m - 273.15 + 243.5))
    e = 6.112 * np.exp(17.67 * (d2m - 273.15) / (d2m - 273.15 + 243.5))
    rh = (e / es) * 100.0
    return np.clip(rh, 0, 100)


def _datetime_to_roms_time(dates, time_ref):
    base_str = time_ref.split('since')[1].strip()
    try:
        base = datetime.strptime(base_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        base = datetime.strptime(base_str.split('UTC')[0].strip(), '%Y-%m-%d %H:%M:%S')

    if 'days' in time_ref.lower():
        return np.array([(d - base).total_seconds() / 86400.0 for d in dates])
    else:
        return np.array([(d - base).total_seconds() for d in dates])