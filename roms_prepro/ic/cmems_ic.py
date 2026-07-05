"""
CMEMS → ROMS initial conditions.

Processing flow (matching reference d_cmems2roms_py.py):
  1. Read CMEMS variables on native grid (one time step)
  2. Range check → NaN on CMEMS grid
  3. Fill NaN using distance_transform_edt on CMEMS grid
  4. Trim depth, extend if ROMS deeper than CMEMS
  5. Vertical fill (per-column nearest neighbor)
  6. Compute ocean_time from CMEMS time + ROMS reference date
  7. Horizontal interpolation: RegularGridInterpolator per layer
  8. Vertical interpolation: np.interp per point to ROMS sigma
  9. Rotate velocity to C-grid, compute barotropic
  10. Apply fill_value on land points
  11. Write NetCDF with _FillValue attributes
"""

import os
import re
import glob
import numpy as np
import netCDF4 as nc4
from datetime import datetime, timedelta

from ..grid.vgrid import set_depth, stretching
from ._core import (_read_roms_grid, cmems_read_var, cmems_interp_2d,
                    cmems_interp_3d, cmems_fill_nan_2d, cmems_fill_nans_vertically,
                    _parse_date, _get_time, _match_idx)

FILL_VAL = 1.0e+37


def _detect_cmems_files(data_dir):
    """Auto-detect CMEMS files in directory. Returns {var: path}."""
    var_patterns = {
        'zos': ['zos', 'ssh', 'sla', 'adt', 'surf_el'],
        'thetao': ['thetao', 't', 'temperature', 'votemper', 'water_temp'],
        'so': ['so', 's', 'salinity', 'vosaline'],
        'uo': ['uo', 'u', 'water_u', 'vozocrtx'],
        'vo': ['vo', 'v', 'water_v', 'vomecrty'],
    }
    result = {}

    # Check for merged files (all variables in one file)
    for fpath in sorted(glob.glob(os.path.join(data_dir, '*.nc'))):
        try:
            ds = nc4.Dataset(fpath, 'r')
            found = {}
            for std_name, patterns in var_patterns.items():
                for p in patterns:
                    if p in ds.variables:
                        found[std_name] = p
                        break
            ds.close()
            if len(found) >= 3:
                return {k: fpath for k in found}
        except Exception:
            continue

    # Separate files per variable
    file_patterns = {
        'zos': ['cmems_zos_*.nc', '*zos*.nc', '*ssh*.nc'],
        'thetao': ['cmems_thetao_*.nc', '*thetao*.nc', '*temp*.nc'],
        'so': ['cmems_so_*.nc', '*so_*.nc', '*salt*.nc'],
        'uo': ['cmems_uo_*.nc', '*uo*.nc'],
        'vo': ['cmems_vo_*.nc', '*vo*.nc'],
    }
    for std_name, patterns in file_patterns.items():
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
            if matches:
                result[std_name] = matches[0]
                break
    return result


def _detect_time_ref_and_init_date(temp_file, init_date, time_index):
    """Auto-detect time_ref from CMEMS file and find time_index from init_date."""
    time_ref = None
    try:
        ds = nc4.Dataset(temp_file, 'r')
        for tname in ['time', 'time_counter', 't']:
            if tname in ds.variables:
                time_var = ds.variables[tname]
                units = getattr(time_var, 'units', '')
                time_values = time_var[:]
                ds.close()

                m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', units)
                if m:
                    ref_str = m.group(1).strip()
                    try:
                        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d')

                    # Determine time unit
                    if 'day' in units.lower():
                        time_seconds = time_values * 86400.0
                    elif 'hour' in units.lower():
                        time_seconds = time_values * 3600.0
                    else:
                        time_seconds = time_values

                    time_dates = [ref_dt + timedelta(seconds=float(s)) for s in time_seconds]
                    time_ref = ref_dt.strftime('%Y-%m-%d')

                    if init_date is not None:
                        init_dt = datetime.strptime(init_date, '%Y-%m-%d')
                        best_idx = min(range(len(time_dates)),
                                       key=lambda i: abs((time_dates[i] - init_dt).total_seconds()))
                        time_index = best_idx
                        print(f'  init_date={init_date} -> time_index {best_idx} ({time_dates[best_idx]})')
                return time_ref, time_index
        ds.close()
    except Exception as e:
        print(f'  Warning: time detection failed: {e}')
    return None, time_index


def cmems_to_roms_ini(roms_grid_file, zeta_file=None, temp_file=None,
                      salt_file=None, u_file=None, v_file=None,
                      data_dir=None, ini_file=None,
                      zeta_var='zos', temp_var='thetao', salt_var='so',
                      u_var='uo', v_var='vo',
                      Vtransform=2, Vstretching=3,
                      theta_s=2.5, theta_b=1.0, Tcline=25.0, N=30,
                      time_ref=None, init_date='2025-05-01', time_index=0):
    """
    Create ROMS initial conditions from CMEMS files.

    Parameters
    ----------
    data_dir : str, optional
        CMEMS data directory (auto-detect files)
    init_date : str
        Date for IC, e.g. '2025-05-01'. Auto-detects time_index from file.
    time_ref : str, optional
        ROMS time reference. Auto-detected if None.
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

    # Read ROMS grid
    metrics = _read_roms_grid(roms_grid_file)
    h = metrics['h']
    lon_rho, lat_rho = metrics['lon_rho'], metrics['lat_rho']
    lon_u, lat_u = metrics['lon_u'], metrics['lat_u']
    lon_v, lat_v = metrics['lon_v'], metrics['lat_v']
    mask_rho = metrics['mask_rho']
    mask_u = metrics.get('mask_u', np.ones_like(mask_rho))
    mask_v = metrics.get('mask_v', np.ones_like(mask_rho))
    angle = metrics.get('angle', np.zeros_like(mask_rho))

    # Auto-detect time_ref and time_index from init_date
    if time_ref is None or init_date is not None:
        detected_ref, time_index = _detect_time_ref_and_init_date(
            temp_file, init_date, time_index)
        if time_ref is None:
            time_ref = detected_ref
        if time_ref:
            print(f'  Time reference: {time_ref}')

    # Read CMEMS data (one time step, native grid)
    Zeta, Zlon, Zlat, _, _ = cmems_read_var(zeta_file, zeta_var, time_index)
    Temp, Tlon, Tlat, Tdepth, _ = cmems_read_var(temp_file, temp_var, time_index)
    Salt, _, _, _, _ = cmems_read_var(salt_file, salt_var, time_index)
    Uvel, Ulon, Ulat, Udepth, _ = cmems_read_var(u_file, u_var, time_index)
    Vvel, _, _, Vdepth, _ = cmems_read_var(v_file, v_var, time_index)

    # Transpose to (lat, lon, depth) for consistent processing
    Temp = np.ascontiguousarray(np.transpose(Temp, (1, 2, 0)))
    Salt = np.ascontiguousarray(np.transpose(Salt, (1, 2, 0)))
    Uvel = np.ascontiguousarray(np.transpose(Uvel, (1, 2, 0)))
    Vvel = np.ascontiguousarray(np.transpose(Vvel, (1, 2, 0)))

    print(f'  Data shapes: Zeta={Zeta.shape}, Temp={Temp.shape}')
    print(f'  CMEMS depths: T={len(Tdepth)}, U={len(Udepth)}, V={len(Vdepth)}')

    # === Step 1: Range check + NaN fill on CMEMS grid ===
    print('\nFilling invalid values on CMEMS grid ...')
    Temp[np.abs(Temp) > 50] = np.nan;   Temp[Temp < -100] = np.nan
    Salt[Salt < 0] = np.nan;            Salt[Salt > 50] = np.nan;  Salt[np.abs(Salt) > 100] = np.nan
    Uvel[np.abs(Uvel) > 100] = np.nan
    Vvel[np.abs(Vvel) > 100] = np.nan
    Zeta[np.abs(Zeta) > 100] = np.nan

    Temp = cmems_fill_nan_2d(Temp)
    Salt = cmems_fill_nan_2d(Salt)
    Uvel = cmems_fill_nan_2d(Uvel)
    Vvel = cmems_fill_nan_2d(Vvel)
    Zeta = cmems_fill_nan_2d(Zeta)

    # === Step 2: Process depth ===
    Tdepth = np.unique(Tdepth[Tdepth > 0])  # Keep positive
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

    # Trim data to match depth
    Temp = Temp[:, :, :len(Tdepth)]
    Salt = Salt[:, :, :len(Tdepth)]
    Uvel = Uvel[:, :, :len(Udepth)]
    Vvel = Vvel[:, :, :len(Vdepth)]

    # Vertical fill
    print('  Filling NaN vertically ...')
    Temp = cmems_fill_nans_vertically(Temp)
    Salt = cmems_fill_nans_vertically(Salt)
    Uvel = cmems_fill_nans_vertically(Uvel)
    Vvel = cmems_fill_nans_vertically(Vvel)

    print(f'  After fill: Temp [{np.nanmin(Temp):.2f}, {np.nanmax(Temp):.2f}]')

    # === Step 3: Compute ocean_time ===
    ref_dt = datetime.strptime(time_ref, '%Y-%m-%d')
    init_dt = datetime.strptime(init_date, '%Y-%m-%d')
    ocean_time = (init_dt - ref_dt).total_seconds()
    print(f'\n  ocean_time: {ocean_time:.0f} s')

    # === Step 4: Horizontal + vertical interpolation to ROMS grid ===
    print('\nInterpolating to ROMS grid ...')

    zeta_mean = np.nanmean(Zeta)
    zeta = cmems_interp_2d(Zeta - zeta_mean, Tlon, Tlat, lon_rho, lat_rho)
    zeta = cmems_fill_nan_2d(zeta)  # Fill any remaining NaN
    zeta += zeta_mean

    # ROMS depths
    ssh = np.zeros_like(zeta)
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, h, ssh, igrid=1)
    z_u = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, h, ssh, igrid=3)
    z_v = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, h, ssh, igrid=4)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, h, zeta, igrid=5)
    Hz = z_w[:, :, 1:N+1] - z_w[:, :, 0:N]

    # 3D interpolation (depths are negative, matching z_r)
    print('  Interpolating temp ...')
    temp = cmems_interp_3d(Temp, Tlon, Tlat, -Tdepth, lon_rho, lat_rho, z_r)
    print('  Interpolating salt ...')
    salt = cmems_interp_3d(Salt, Tlon, Tlat, -Tdepth, lon_rho, lat_rho, z_r)
    print('  Interpolating u ...')
    Urho = cmems_interp_3d(Uvel, Ulon, Ulat, -Udepth, lon_rho, lat_rho, z_r)
    print('  Interpolating v ...')
    Vrho = cmems_interp_3d(Vvel, Ulon, Ulat, -Vdepth, lon_rho, lat_rho, z_r)

    # === Step 5: Velocity rotation to C-grid ===
    print('\nRotating velocity ...')
    angle_3d = angle[:, :, np.newaxis]
    Urot = Urho * np.cos(angle_3d) + Vrho * np.sin(angle_3d)
    Vrot = Vrho * np.cos(angle_3d) - Urho * np.sin(angle_3d)
    u = 0.5 * (Urot[:, :-1, :] + Urot[:, 1:, :])
    v = 0.5 * (Vrot[:-1, :, :] + Vrot[1:, :, :])

    # === Step 6: Barotropic velocity ===
    Hz_u = 0.5 * (Hz[:, :-1, :] + Hz[:, 1:, :])
    Hz_v = 0.5 * (Hz[:-1, :, :] + Hz[1:, :, :])
    sum_Hz_u = np.sum(Hz_u, axis=2)
    sum_Hz_v = np.sum(Hz_v, axis=2)
    sum_Hz_u[sum_Hz_u == 0] = 1
    sum_Hz_v[sum_Hz_v == 0] = 1
    ubar = np.sum(u * Hz_u, axis=2) / sum_Hz_u
    vbar = np.sum(v * Hz_v, axis=2) / sum_Hz_v

    # === Step 7: Apply land masks ===
    print('Applying land masks ...')
    temp[mask_rho == 0] = FILL_VAL
    salt[mask_rho == 0] = FILL_VAL
    zeta[mask_rho == 0] = FILL_VAL
    u[mask_u == 0] = FILL_VAL
    ubar[mask_u == 0] = FILL_VAL
    v[mask_v == 0] = FILL_VAL
    vbar[mask_v == 0] = FILL_VAL

    print(f'  Water temp range: [{temp[mask_rho == 1].min():.2f}, {temp[mask_rho == 1].max():.2f}]')

    # === Step 8: Write NetCDF ===
    print('\nWriting initial conditions ...')
    _write_ic_netcdf(ini_file, h, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
                     ocean_time, theta_s, theta_b, Tcline, theta_s, Tcline,
                     N, zeta, ubar, vbar, u, v, temp, salt, Vtransform, Vstretching)

    print('=' * 60)
    print('Done!')
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v,
            'ubar': ubar, 'vbar': vbar}


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

    # Vertical coordinate
    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, 0)
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, 1)

    ds['s_rho'] = xr.DataArray(s_rho, dims=['s_rho'],
        attrs={'long_name': 'S-coordinate at RHO-points', 'valid_min': -1.0, 'valid_max': 0.0})
    ds['s_w'] = xr.DataArray(s_w, dims=['s_w'],
        attrs={'long_name': 'S-coordinate at W-points', 'valid_min': -1.0, 'valid_max': 0.0})
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
            attrs={'long_name': long_name, 'units': units, '_FillValue': np.float32(FILL_VAL)})

    add_var('zeta', zeta, ('eta_rho', 'xi_rho'), 'free-surface', 'meter')
    add_var('ubar', ubar, ('eta_u', 'xi_u'), 'vertically integrated u-momentum', 'meter second-1')
    add_var('vbar', vbar, ('eta_v', 'xi_v'), 'vertically integrated v-momentum', 'meter second-1')
    add_var('temp', temp, ('eta_rho', 'xi_rho', 's_rho'), 'potential temperature', 'Celsius')
    add_var('salt', salt, ('eta_rho', 'xi_rho', 's_rho'), 'salinity', 'PSU')
    add_var('u', u, ('eta_u', 'xi_u', 's_rho'), 'u-momentum component', 'meter second-1')
    add_var('v', v, ('eta_v', 'xi_v', 's_rho'), 'v-momentum component', 'meter second-1')

    ds.to_netcdf(fname, engine='h5netcdf')
    ds.close()


def mercator_to_roms_ini(roms_grid_file, source_file, ini_file,
                         Vtransform=2, Vstretching=4,
                         theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                         init_date=None, time_ref=None,
                         **source_kwargs):
    """Create ROMS IC from a single multi-variable file (Mercator/HYCOM)."""
    from ._core import horizontal_interp, z_to_sigma, rotate_uv, uv_to_cgrid, compute_ubar_vbar
    from ._core import _fill_nan, _parse_date, _get_time, _match_idx, _read_source

    metrics = _read_roms_grid(roms_grid_file)

    src_time = _get_time(source_file, source_kwargs.get('time_var'))
    if init_date is not None and src_time is not None:
        idx = _match_idx(src_time, init_date)
    else:
        idx = 0

    src = _read_source(source_file, time_index=idx, **source_kwargs)
    h = metrics['h']; mask = metrics['mask_rho']
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, h, igrid=1)
    spval = 1e37

    zeta = horizontal_interp(src['lon_1d'], src['lat_1d'], src['zeta'],
                              metrics['lon_rho'], metrics['lat_rho'], mask=mask, fill_value=0.0)
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

    temp = z_to_sigma(temp_h, src['depth'], z_r, fill_value=spval)
    salt = z_to_sigma(salt_h, src['depth'], z_r, fill_value=spval)
    u_rho = z_to_sigma(u_h, src['depth'], z_r, fill_value=spval)
    v_rho = z_to_sigma(v_h, src['depth'], z_r, fill_value=spval)

    angle = metrics.get('angle', np.zeros_like(metrics['lat_rho']))
    u_rho, v_rho = rotate_uv(u_rho, v_rho, 0.0, angle)
    u, v = uv_to_cgrid(u_rho, v_rho, metrics.get('mask_u'), metrics.get('mask_v'), spval)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, h, igrid=5)
    ubar, vbar = compute_ubar_vbar(u, v, z_w, metrics.get('mask_u'), metrics.get('mask_v'))

    if time_ref and src_time is not None:
        ref = _parse_date(re.search(r'since\s+(.+)', time_ref).group(1) if 'since' in time_ref else time_ref)
        ocean_time = np.array([src_time[idx] - ref])
    else:
        ocean_time = np.array([0.0])

    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, kgrid=0)
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, kgrid=1)
    vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                    'theta_s': theta_s, 'theta_b': theta_b, 'Tcline': Tcline, 'hc': Tcline,
                    's_rho': s_rho, 'Cs_r': Cs_r, 's_w': s_w, 'Cs_w': Cs_w}

    from ._core import write_ic_file
    write_ic_file(ini_file, metrics, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar)
    print(f"Written: {ini_file}")
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v}
