import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime


def era5_to_roms_forcing(
    roms_grid_file,
    era5_files,
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
):
    era5_files = sorted(era5_files)
    if rh_files:
        rh_files = sorted(rh_files)

    g = nc.Dataset(roms_grid_file)
    lon_rho = g.variables['lon_rho'][:]
    lat_rho = g.variables['lat_rho'][:]
    mask = g.variables['mask_rho'][:]
    g.close()

    Mp, Lp = lon_rho.shape  # ROMS convention: (eta_rho, xi_rho)

    time_coord = time_ref.replace('days', 'seconds').replace('DAYS', 'SECONDS')

    ds0 = nc.Dataset(era5_files[0])
    era5_lon = ds0.variables['longitude'][:]
    era5_lat = ds0.variables['latitude'][:]
    era5_time_raw = ds0.variables['time'][:]
    era5_time_units = ds0.variables['time'].units
    ds0.close()

    if 'days since' in era5_time_units.lower():
        era5_dates = [datetime(1900, 1, 1) + np.timedelta64(int(t * 24), 'h') for t in era5_time_raw]
    else:
        era5_dates = [datetime(1900, 1, 1) + np.timedelta64(int(t), 'h') for t in era5_time_raw]

    all_times = []
    all_data = {k: [] for k in ['lwrad', 'lwrad_down', 'swrad', 'rain',
                                  'Tair', 'Pair', 'Qair', 'Uwind', 'Vwind']}

    era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat)

    if rh_files:
        ds_rh0 = nc.Dataset(rh_files[0])
        rh_lon = ds_rh0.variables['longitude'][:]
        rh_lat = ds_rh0.variables['latitude'][:]
        ds_rh0.close()
        rh_lon_2d, rh_lat_2d = np.meshgrid(rh_lon, rh_lat)

    print('Interpolating ERA5 data to ROMS grid...')
    for fi, (ef, rf) in enumerate(zip(era5_files, rh_files if rh_files else [None] * len(era5_files))):
        ds = nc.Dataset(ef)
        et = ds.variables['time'][:]
        if 'days since' in ds.variables['time'].units.lower():
            file_dates = [datetime(1900, 1, 1) + np.timedelta64(int(t * 24), 'h') for t in et]
        else:
            file_dates = [datetime(1900, 1, 1) + np.timedelta64(int(t), 'h') for t in et]
        ds.close()

        sd_dt = datetime.fromisoformat(start_date.replace(' ', 'T')) if start_date else None
        ed_dt = datetime.fromisoformat(end_date.replace(' ', 'T')) if end_date else None

        for ti, dt in enumerate(file_dates):
            ts = dt.strftime('%Y-%m-%d %H:%M')
            print(f'  [{fi+1}/{len(era5_files)}] {ts}')

            if sd_dt is not None and dt < sd_dt:
                continue
            if ed_dt is not None and dt > ed_dt:
                continue

            ds = nc.Dataset(ef)
            tidx = ti
            point_data = {}

            if get_lwrad:
                down = ds.variables['msdwlwrf'][tidx]
                net = ds.variables.get('msnlwrf', None)
                if net is not None:
                    net = net[tidx]
                else:
                    net = down * 0.9
                point_data['lwrad_down'] = _interp2d(era5_lon_2d, era5_lat_2d, down, lon_rho, lat_rho)
                point_data['lwrad'] = _interp2d(era5_lon_2d, era5_lat_2d, net, lon_rho, lat_rho)

            if get_swrad:
                sw = ds.variables['msnswrf'][tidx]
                point_data['swrad'] = _interp2d(era5_lon_2d, era5_lat_2d, sw, lon_rho, lat_rho)

            if get_rain:
                tp = ds.variables['tp'][tidx] * 1000.0 / 3600.0
                point_data['rain'] = _interp2d(era5_lon_2d, era5_lat_2d, tp, lon_rho, lat_rho)

            if get_Tair:
                t2m = ds.variables['t2m'][tidx] - 273.15
                point_data['Tair'] = _interp2d(era5_lon_2d, era5_lat_2d, t2m, lon_rho, lat_rho)

            if get_Pair:
                msl = ds.variables['msl'][tidx] * 0.01
                point_data['Pair'] = _interp2d(era5_lon_2d, era5_lat_2d, msl, lon_rho, lat_rho)

            if get_Qair:
                if rf:
                    dr = nc.Dataset(rf)
                    rh = dr.variables['r'][tidx]
                    dr.close()
                else:
                    d2m = ds.variables.get('d2m', None)
                    t2m_k = ds.variables['t2m'][tidx]
                    if d2m is not None:
                        d2m = d2m[tidx]
                        rh = _relative_humidity_from_dewpoint(d2m, t2m_k)
                    else:
                        rh = np.full_like(lon_rho, 70.0)
                point_data['Qair'] = _interp2d(
                    rh_lon_2d if rf else era5_lon_2d,
                    rh_lat_2d if rf else era5_lat_2d,
                    rh, lon_rho, lat_rho
                )

            if get_Wind:
                u10 = ds.variables['u10'][tidx]
                v10 = ds.variables['v10'][tidx]
                point_data['Uwind'] = _interp2d(era5_lon_2d, era5_lat_2d, u10, lon_rho, lat_rho)
                point_data['Vwind'] = _interp2d(era5_lon_2d, era5_lat_2d, v10, lon_rho, lat_rho)

            ds.close()

            all_times.append(dt)
            for k in all_data:
                all_data[k].append(point_data.get(k, None))

    ntimes = len(all_times)
    if ntimes == 0:
        print('No data in date range.')
        return

    roms_times = _datetime_to_roms_time(all_times, time_ref)

    out = out_file or 'romsforc_era5.nc'
    print(f'Writing {out}...')
    nc_out = nc.Dataset(out, 'w', format='NETCDF3_64BIT')
    nc_out.type = 'bulk fluxes forcing file'
    nc_out.created = datetime.now().isoformat()

    xr_dim = nc_out.createDimension('xr', Mp)
    er_dim = nc_out.createDimension('er', Lp)
    t_dim = nc_out.createDimension('time', ntimes)

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

    def _write_var(nc_out, name, long_name, units, data, time_dim='time'):
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
        _write_var(nc_out, 'lwrad', 'net solar longwave radiation', 'Watts meter-2',
                   np.array(all_data['lwrad']))
        _write_var(nc_out, 'lwrad_down', 'downward solar longwave radiation', 'Watts meter-2',
                   np.array(all_data['lwrad_down']))

    if get_swrad:
        _write_var(nc_out, 'swrad', 'net solar shortwave radiation', 'Watts meter-2',
                   np.array(all_data['swrad']))

    if get_rain:
        rain_arr = np.array(all_data['rain'])
        rain_arr = np.maximum(rain_arr, 0)
        _write_var(nc_out, 'rain', 'rain fall rate', 'kilogram meter-2 second-1',
                   rain_arr)

    if get_Tair:
        tair_arr = np.array(all_data['Tair'])
        tair_arr = np.clip(tair_arr, -40, 45)
        _write_var(nc_out, 'Tair', 'surface air temperature', 'Celsius',
                   tair_arr)

    if get_Pair:
        pair_arr = np.array(all_data['Pair'])
        pair_arr = np.clip(pair_arr, 950, 1080)
        _write_var(nc_out, 'Pair', 'surface air pressure', 'millibar',
                   pair_arr)

    if get_Qair:
        qair_arr = np.array(all_data['Qair'])
        qair_arr = np.clip(qair_arr, 1, 100)
        _write_var(nc_out, 'Qair', 'surface air relative humidity', 'percentage',
                   qair_arr)

    if get_Wind:
        uwind_arr = np.array(all_data['Uwind'])
        vwind_arr = np.array(all_data['Vwind'])
        uwind_arr = np.clip(uwind_arr, -50, 50)
        vwind_arr = np.clip(vwind_arr, -50, 50)
        _write_var(nc_out, 'Uwind', 'surface u-wind component', 'meter second-1',
                   uwind_arr)
        _write_var(nc_out, 'Vwind', 'surface v-wind component', 'meter second-1',
                   vwind_arr)

    nc_out.close()
    print(f'Forcing file written: {out}  ({ntimes} timesteps)')


def _interp2d(src_lon_2d, src_lat_2d, src_data, dst_lon, dst_lat):
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