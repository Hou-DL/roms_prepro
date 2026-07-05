"""
CMEMS → ROMS initial conditions.

Processing flow (matches reference d_cmems2roms_py.py):
  1. Read CMEMS variables on native grid (one time step)
  2. Range check → NaN, then fill NaN on CMEMS grid (distance_transform)
  3. Trim depth, extend if ROMS deeper than CMEMS
  4. Vertical fill (per-column nearest neighbor)
  5. Compute ocean_time from CMEMS time units
  6. Horizontal interp: RegularGridInterpolator (linear)
  7. Vertical interp: np.interp per point
  8. Rotate velocity to C-grid, compute barotropic
  9. Apply fill_value on land points
  10. Write NetCDF with _FillValue attributes
"""

import os
import numpy as np
from datetime import datetime, timedelta

from ._core import (_read_roms_grid, _read_cmems_var, _detect_cmems_files,
                    _compute_ocean_time, _detect_time_units,
                    fill_nan_2d, fill_nans_vertically,
                    interp_to_roms, mercator2roms_2d, mercator2roms_3d)

# Import reference functions from bc.roms_tools
import sys
_tools_path = os.path.join(os.path.dirname(__file__), '..', 'bc')
if _tools_path not in sys.path:
    sys.path.insert(0, _tools_path)
from roms_tools import stretching, set_depth

FILL_VAL = 1.0e+37


def cmems_to_roms_ini(roms_grid_file, zeta_file=None, temp_file=None,
                      salt_file=None, u_file=None, v_file=None,
                      data_dir=None, ini_file=None,
                      zeta_var='zos', temp_var='thetao', salt_var='so',
                      u_var='uo', v_var='vo',
                      Vtransform=2, Vstretching=3,
                      theta_s=2.5, theta_b=1.0, Tcline=25.0, N=30,
                      time_ref='1990-01-01', init_date='2025-05-01',
                      time_index=0):
    """
    Create ROMS initial conditions from CMEMS files.

    Parameters
    ----------
    data_dir : str, optional
        CMEMS data directory (auto-detect files)
    init_date : str
        Date for IC, e.g. '2025-05-01'. Auto-detects time_index from file.
    time_ref : str, optional
        ROMS time reference date, e.g. '1990-01-01'. Auto-detected if None.
    """
    print('\nReading CMEMS data ...\n')

    # Auto-detect files
    if data_dir and not all([zeta_file, temp_file, salt_file, u_file, v_file]):
        detected = _detect_cmems_files(data_dir)
        if not detected:
            raise FileNotFoundError(f'No CMEMS files in {data_dir}')
        zeta_file = zeta_file or detected.get('zos')
        temp_file = temp_file or detected.get('thetao')
        salt_file = salt_file or detected.get('so')
        u_file = u_file or detected.get('uo')
        v_file = v_file or detected.get('vo')
        for var, fpath in detected.items():
            print(f'  {var}: {os.path.basename(fpath)}')

    # Check files exist
    for name, path in [('zeta', zeta_file), ('temp', temp_file), ('salt', salt_file),
                       ('u', u_file), ('v', v_file)]:
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f'{name} file not found: {path}')

    # Read ROMS grid (using bc.roms_tools convention)
    import netCDF4 as nc
    ds = nc.Dataset(roms_grid_file)
    h = np.array(ds.variables['h'][:], dtype=float)
    lon_rho = np.array(ds.variables['lon_rho'][:], dtype=float)
    lat_rho = np.array(ds.variables['lat_rho'][:], dtype=float)
    lon_u = np.array(ds.variables['lon_u'][:], dtype=float)
    lat_u = np.array(ds.variables['lat_u'][:], dtype=float)
    lon_v = np.array(ds.variables['lon_v'][:], dtype=float)
    lat_v = np.array(ds.variables['lat_v'][:], dtype=float)
    mask_rho = np.array(ds.variables['mask_rho'][:], dtype=float)
    mask_u = np.array(ds.variables['mask_u'][:], dtype=float)
    mask_v = np.array(ds.variables['mask_v'][:], dtype=float)
    angle = np.array(ds.variables['angle'][:], dtype=float)
    ds.close()

    nlat, nlon = h.shape
    print(f'  Grid: {nlat} x {nlon}')

    # Auto-detect time_ref and time_index from init_date
    if time_ref is None or init_date is not None:
        time_units, time_values = _detect_time_units(temp_file)
        if time_units:
            idx = _match_time_idx(time_units, time_values, init_date)
            time_index = idx
            m = _re_search(r'since\s+([\d\-]+\s*[\d:]*)', time_units)
            if m:
                ref_str = m.group(1).strip()
                try:
                    ref_dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    ref_dt = datetime.strptime(ref_str, '%Y-%m-%d')
                time_ref = ref_dt.strftime('%Y-%m-%d')
                print(f'  init_date={init_date} -> time_index {time_index}')
                print(f'  Time reference: {time_ref}')

    # Read CMEMS data (native grid, one time step)
    Zeta, Zlon, Zlat, _, _ = _read_cmems_var(zeta_file, zeta_var, time_index)
    Temp, Tlon, Tlat, Tdepth, _ = _read_cmems_var(temp_file, temp_var, time_index)
    Salt, _, _, _, _ = _read_cmems_var(salt_file, salt_var, time_index)
    Uvel, Ulon, Ulat, Udepth, _ = _read_cmems_var(u_file, u_var, time_index)
    Vvel, _, _, Vdepth, _ = _read_cmems_var(v_file, v_var, time_index)

    # Transpose to (lat, lon, depth) for consistent processing
    Temp = np.ascontiguousarray(np.transpose(Temp, (1, 2, 0)))
    Salt = np.ascontiguousarray(np.transpose(Salt, (1, 2, 0)))
    Uvel = np.ascontiguousarray(np.transpose(Uvel, (1, 2, 0)))
    Vvel = np.ascontiguousarray(np.transpose(Vvel, (1, 2, 0)))

    print(f'  Data shapes: Zeta={Zeta.shape}, Temp={Temp.shape}')
    print(f'  CMEMS depths: T={len(Tdepth)}, U={len(Udepth)}, V={len(Vdepth)}')

    # === Step 1: Range check + NaN fill on CMEMS grid ===
    print('\nFilling invalid values ...')
    Temp[np.abs(Temp) > 50] = np.nan;   Temp[Temp < -100] = np.nan
    Salt[Salt < 0] = np.nan;            Salt[Salt > 50] = np.nan;  Salt[np.abs(Salt) > 100] = np.nan
    Uvel[np.abs(Uvel) > 100] = np.nan
    Vvel[np.abs(Vvel) > 100] = np.nan
    Zeta[np.abs(Zeta) > 100] = np.nan

    Temp = fill_nan_2d(Temp)
    Salt = fill_nan_2d(Salt)
    Uvel = fill_nan_2d(Uvel)
    Vvel = fill_nan_2d(Vvel)
    Zeta = fill_nan_2d(Zeta)

    # === Step 2: Process depth ===
    Tdepth = np.unique(Tdepth[Tdepth > 0])
    Udepth = np.unique(Udepth[Udepth > 0])
    Vdepth = np.unique(Vdepth[Vdepth > 0])

    hmax = h.max()
    print(f'  CMEMS max depth: {Tdepth[-1]:.0f}m, ROMS max: {hmax:.0f}m')

    # Extend depth if ROMS deeper
    if Tdepth[-1] < hmax:
        Tdepth = np.append(Tdepth, [Tdepth[-1] + 200, hmax + 200])
        Temp = np.concatenate([Temp, Temp[:, :, -1:], Temp[:, :, -1:]], axis=2)
        Salt = np.concatenate([Salt, Salt[:, :, -1:], Salt[:, :, -1:]], axis=2)
        Udepth = np.append(Udepth, [Udepth[-1] + 200, hmax + 200])
        Vdepth = np.append(Vdepth, [Vdepth[-1] + 200, hmax + 200])
        Uvel = np.concatenate([Uvel, np.zeros_like(Uvel[:, :, -1:]), np.zeros_like(Uvel[:, :, -1:])], axis=2)
        Vvel = np.concatenate([Vvel, np.zeros_like(Vvel[:, :, -1:]), np.zeros_like(Vvel[:, :, -1:])], axis=2)

    # Trim to depth
    Temp = Temp[:, :, :len(Tdepth)]
    Salt = Salt[:, :, :len(Tdepth)]
    Uvel = Uvel[:, :, :len(Udepth)]
    Vvel = Vvel[:, :, :len(Vdepth)]

    # Vertical fill
    print('  Filling NaN vertically ...')
    Temp = fill_nans_vertically(Temp)
    Salt = fill_nans_vertically(Salt)
    Uvel = fill_nans_vertically(Uvel)
    Vvel = fill_nans_vertically(Vvel)

    print(f'  After fill: Temp [{np.nanmin(Temp):.2f}, {np.nanmax(Temp):.2f}]')

    # === Step 3: Compute ocean_time ===
    time_units, time_values = _detect_time_units(temp_file)
    ocean_time = _compute_ocean_time(time_units, time_values, init_date, time_ref)
    print(f'\n  ocean_time: {ocean_time:.0f} s')

    # === Step 4: Interpolation ===
    print('\nInterpolating to ROMS grid ...')

    zeta = mercator2roms_2d(Zeta, Tlon, Tlat, lon_rho, lat_rho)
    print(f'  zeta: [{np.nanmin(zeta):.3f}, {np.nanmax(zeta):.3f}]')

    # ROMS sigma depths (using bc.roms_tools.set_depth)
    ssh = np.zeros_like(zeta)
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 1, h, ssh)
    z_u = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 3, h, ssh)
    z_v = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 4, h, ssh)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 5, h, zeta)
    Hz = z_w[:, :, 1:N+1] - z_w[:, :, 0:N]

    # 3D interpolation
    print('  Interpolating temp ...')
    temp = mercator2roms_3d(Temp, Tlon, Tlat, -Tdepth, lon_rho, lat_rho, z_r)
    print('  Interpolating salt ...')
    salt = mercator2roms_3d(Salt, Tlon, Tlat, -Tdepth, lon_rho, lat_rho, z_r)
    print('  Interpolating u ...')
    Urho = mercator2roms_3d(Uvel, Ulon, Ulat, -Udepth, lon_rho, lat_rho, z_r)
    print('  Interpolating v ...')
    Vrho = mercator2roms_3d(Vvel, Ulon, Ulat, -Vdepth, lon_rho, lat_rho, z_r)

    print(f'  Temp: [{np.nanmin(temp):.2f}, {np.nanmax(temp):.2f}]')
    print(f'  Salt: [{np.nanmin(salt):.2f}, {np.nanmax(salt):.2f}]')

    # === Step 5: Velocity rotation + C-grid + barotropic ===
    print('\nRotating velocity ...')
    angle_3d = angle[:, :, np.newaxis]
    Urot = Urho * np.cos(angle_3d) + Vrho * np.sin(angle_3d)
    Vrot = Vrho * np.cos(angle_3d) - Urho * np.sin(angle_3d)
    u = 0.5 * (Urot[:, :-1, :] + Urot[:, 1:, :])
    v = 0.5 * (Vrot[:-1, :, :] + Vrot[1:, :, :])

    Hz_u = 0.5 * (Hz[:, :-1, :] + Hz[:, 1:, :])
    Hz_v = 0.5 * (Hz[:-1, :, :] + Hz[1:, :, :])
    sum_Hz_u = np.sum(Hz_u, axis=2)
    sum_Hz_v = np.sum(Hz_v, axis=2)
    sum_Hz_u[sum_Hz_u == 0] = 1
    sum_Hz_v[sum_Hz_v == 0] = 1
    ubar = np.sum(u * Hz_u, axis=2) / sum_Hz_u
    vbar = np.sum(v * Hz_v, axis=2) / sum_Hz_v

    # === Step 6: Land masks ===
    print('Applying masks ...')
    temp[mask_rho == 0] = FILL_VAL
    salt[mask_rho == 0] = FILL_VAL
    zeta[mask_rho == 0] = FILL_VAL
    u[mask_u == 0] = FILL_VAL
    ubar[mask_u == 0] = FILL_VAL
    v[mask_v == 0] = FILL_VAL
    vbar[mask_v == 0] = FILL_VAL

    print(f'  Water temp: [{temp[mask_rho == 1].min():.2f}, {temp[mask_rho == 1].max():.2f}]')

    # === Step 7: Write NetCDF ===
    print('\nWriting NetCDF ...')
    _write_ic_netcdf(ini_file, h, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
                     ocean_time, theta_s, theta_b, Tcline, Tcline,
                     N, zeta, ubar, vbar, u, v, temp, salt, Vtransform, Vstretching)

    print('=' * 60)
    print('Done!')
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v,
            'ubar': ubar, 'vbar': vbar}


def _stretching(Vstretching, theta_s, theta_b, N, kgrid):
    """ROMS vertical stretching function (matches reference signature)."""
    ds = 1.0 / N
    if kgrid == 0:
        s = (np.arange(1, N + 1) - 0.5) * ds - 1.0
    else:
        s = np.arange(0, N + 1) * ds - 1.0

    if Vstretching == 1:
        C = (1.0 - theta_b) * np.sinh(theta_s * s) / np.sinh(theta_s) + \
            theta_b * (np.tanh(theta_s * (s + 0.5)) / (2.0 * np.tanh(0.5 * theta_s)) - 0.5)
    elif Vstretching in (2, 3, 4):
        C = (1.0 - theta_b) * np.sinh(theta_s * s) / np.sinh(theta_s) + \
            theta_b * (np.tanh(theta_s * (s + 0.5)) / (2.0 * np.tanh(0.5 * theta_s)) - 0.5)
    elif Vstretching == 5:
        alpha = 3.0; beta = 0.5
        Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
        Cbot = (np.exp(theta_b * Csur) - 1.0) / (1.0 - np.exp(-theta_b))
        C = ((1.0 - np.tanh(alpha * (s + 0.5))) / 2.0) * Cbot + \
            ((1.0 + np.tanh(alpha * s)) / 2.0) * Csur
    else:
        raise ValueError(f"Unsupported Vstretching={Vstretching}")
    return s, C


def _re_search(pattern, string):
    """Regex search shortcut."""
    import re
    return re.search(pattern, string)


def _match_time_idx(time_units, time_values, init_date):
    """Find time index closest to init_date."""
    if init_date is None or time_values is None:
        return 0
    m = _re_search(r'since\s+([\d\-]+\s*[\d:]*)', time_units)
    if not m:
        return 0
    ref_str = m.group(1).strip()
    try:
        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d')

    if 'day' in time_units.lower():
        time_seconds = time_values * 86400.0
    elif 'hour' in time_units.lower():
        time_seconds = time_values * 3600.0
    else:
        time_seconds = time_values

    time_dates = [ref_dt + timedelta(seconds=float(s)) for s in time_seconds]
    init_dt = datetime.strptime(init_date, '%Y-%m-%d')
    return min(range(len(time_dates)),
               key=lambda i: abs((time_dates[i] - init_dt).total_seconds()))


def _write_ic_netcdf(fname, h, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
                     ocean_time, theta_s, theta_b, Tcline, hc,
                     N, zeta, ubar, vbar, u, v, temp, salt,
                     Vtransform, Vstretching):
    """Write ROMS IC NetCDF using xarray with _FillValue."""
    import xarray as xr

    nlat, nlon = h.shape
    if os.path.exists(fname):
        os.remove(fname)

    ds = xr.Dataset()
    ds.attrs['type'] = 'INITIALIZATION file'

    # Stretching (matches reference d_cmems2roms_py.py signature)
    s_rho, Cs_r = _stretching(Vstretching, theta_s, theta_b, N, 0)
    s_w, Cs_w = _stretching(Vstretching, theta_s, theta_b, N, 1)

    ds['s_rho'] = xr.DataArray(s_rho, dims=['s_rho'],
        attrs={'long_name': 'S-coordinate at RHO-points',
               'valid_min': -1.0, 'valid_max': 0.0, 'positive': 'up'})
    ds['s_w'] = xr.DataArray(s_w, dims=['s_w'],
        attrs={'long_name': 'S-coordinate at W-points',
               'valid_min': -1.0, 'valid_max': 0.0, 'positive': 'up'})
    ds['Cs_r'] = xr.DataArray(Cs_r, dims=['s_rho'],
        attrs={'long_name': 'S-coordinate stretching curves at RHO-points'})
    ds['Cs_w'] = xr.DataArray(Cs_w, dims=['s_w'],
        attrs={'long_name': 'S-coordinate stretching curves at W-points'})

    # Grid
    ds['h'] = xr.DataArray(h, dims=['eta_rho', 'xi_rho'])
    ds['lon_rho'] = xr.DataArray(lon_rho, dims=['eta_rho', 'xi_rho'])
    ds['lat_rho'] = xr.DataArray(lat_rho, dims=['eta_rho', 'xi_rho'])
    ds['lon_u'] = xr.DataArray(lon_u, dims=['eta_u', 'xi_u'])
    ds['lat_u'] = xr.DataArray(lat_u, dims=['eta_u', 'xi_u'])
    ds['lon_v'] = xr.DataArray(lon_v, dims=['eta_v', 'xi_v'])
    ds['lat_v'] = xr.DataArray(lat_v, dims=['eta_v', 'xi_v'])

    # Time
    ds['ocean_time'] = xr.DataArray([ocean_time], dims=['ocean_time'],
        attrs={'long_name': 'time since initialization',
               'units': 'seconds since 1990-01-01 00:00:00',
               'calendar': 'gregorian'})

    def add_var(name, data, dims, long_name, units):
        ds[name] = xr.DataArray(data[np.newaxis, ...].astype('f4'),
            dims=['ocean_time'] + list(dims),
            attrs={'long_name': long_name, 'units': units,
                   'field': f'{name}, scalar, series', 'coordinates': 'lon lat',
                   '_FillValue': np.float32(FILL_VAL)})

    add_var('zeta', zeta, ('eta_rho', 'xi_rho'), 'free-surface', 'meter')
    add_var('ubar', ubar, ('eta_u', 'xi_u'),
            'vertically integrated u-momentum', 'meter second-1')
    add_var('vbar', vbar, ('eta_v', 'xi_v'),
            'vertically integrated v-momentum', 'meter second-1')
    add_var('temp', temp, ('eta_rho', 'xi_rho', 's_rho'),
            'potential temperature', 'Celsius')
    add_var('salt', salt, ('eta_rho', 'xi_rho', 's_rho'), 'salinity', 'PSU')
    add_var('u', u, ('eta_u', 'xi_u', 's_rho'),
            'u-momentum component', 'meter second-1')
    add_var('v', v, ('eta_v', 'xi_v', 's_rho'),
            'v-momentum component', 'meter second-1')

    # Write with h5netcdf (handles Chinese paths) or fall back to netcdf4
    try:
        ds.to_netcdf(fname, engine='h5netcdf')
    except (ImportError, ModuleNotFoundError):
        ds.to_netcdf(fname, engine='netcdf4')
    ds.close()


def mercator_to_roms_ini(roms_grid_file, source_file, ini_file,
                         Vtransform=2, Vstretching=4,
                         theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                         init_date=None, time_ref=None,
                         **source_kwargs):
    """Create ROMS IC from a single multi-variable file (Mercator/HYCOM)."""
    # Import from the old _core for single-file processing
    from ._core_old import (horizontal_interp, z_to_sigma, rotate_uv,
                            uv_to_cgrid, compute_ubar_vbar, write_ic_file,
                            _fill_nan, _parse_date, _get_time, _match_idx,
                            _read_source, _read_roms_grid)

    metrics = _read_roms_grid(roms_grid_file)

    src_time = _get_time(source_file, source_kwargs.get('time_var'))
    if init_date is not None and src_time is not None:
        idx = _match_idx(src_time, init_date)
    else:
        idx = 0

    src = _read_source(source_file, time_index=idx, **source_kwargs)
    h = metrics['h']
    mask = metrics['mask_rho']
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 1, h, np.zeros_like(h))
    spval = 1e37

    zeta = horizontal_interp(src['lon_1d'], src['lat_1d'], src['zeta'],
                              metrics['lon_rho'], metrics['lat_rho'],
                              mask=mask, fill_value=0.0)
    zeta = np.nan_to_num(zeta, nan=0.0)
    temp_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['temp'],
                                metrics['lon_rho'], metrics['lat_rho'])
    salt_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['salt'],
                                metrics['lon_rho'], metrics['lat_rho'])
    u_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['u'],
                             metrics['lon_rho'], metrics['lat_rho'])
    v_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['v'],
                             metrics['lon_rho'], metrics['lat_rho'])

    temp_h = _fill_nan(temp_h)
    salt_h = _fill_nan(salt_h)
    u_h = _fill_nan(u_h)
    v_h = _fill_nan(v_h)

    # Need to convert z_r from (eta, xi, N) to (N, eta, xi) for z_to_sigma
    z_r_t = z_r.transpose(2, 0, 1)  # (N, eta, xi)
    src_depth = src['depth']

    temp = z_to_sigma(temp_h, src_depth, z_r_t, fill_value=spval)
    salt = z_to_sigma(salt_h, src_depth, z_r_t, fill_value=spval)
    u_rho = z_to_sigma(u_h, src_depth, z_r_t, fill_value=spval)
    v_rho = z_to_sigma(v_h, src_depth, z_r_t, fill_value=spval)

    angle = metrics.get('angle', np.zeros_like(metrics['lat_rho']))
    u_rho, v_rho = rotate_uv(u_rho, v_rho, 0.0, angle)
    u, v = uv_to_cgrid(u_rho, v_rho, metrics.get('mask_u'), metrics.get('mask_v'), spval)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 5, h, zeta)
    z_w_t = z_w.transpose(2, 0, 1)  # (N+1, eta, xi)
    ubar, vbar = compute_ubar_vbar(u, v, z_w_t, metrics.get('mask_u'), metrics.get('mask_v'))

    if time_ref and src_time is not None:
        ref = _parse_date(_re_search(r'since\s+(.+)', time_ref).group(1) if 'since' in time_ref else time_ref)
        ocean_time = np.array([src_time[idx] - ref])
    else:
        ocean_time = np.array([0.0])

    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, 0)
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, 1)
    vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                    'theta_s': theta_s, 'theta_b': theta_b, 'Tcline': Tcline, 'hc': Tcline,
                    's_rho': s_rho, 'Cs_r': Cs_r, 's_w': s_w, 'Cs_w': Cs_w}

    write_ic_file(ini_file, metrics, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar)
    print(f"Written: {ini_file}")
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v}
