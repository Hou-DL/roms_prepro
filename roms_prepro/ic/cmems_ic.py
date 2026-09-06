"""
CMEMS -> ROMS initial conditions.

Processing flow (matches reference d_cmems2roms_py.py):
  1. Read CMEMS variables on native grid (one time step)
  2. Range check -> NaN, then fill NaN on CMEMS grid (distance_transform)
  3. Trim depth, extend if ROMS deeper than CMEMS
  4. Vertical fill (per-column nearest neighbor)
  5. Compute ocean_time from CMEMS time units
  6. Horizontal interp: RegularGridInterpolator (linear)
  7. Vertical interp: np.interp per point
  8. Rotate velocity to C-grid, compute barotropic
  9. Apply fill_value on land points
  10. Write NetCDF with _FillValue attributes

CMEMS file dates and time references are auto-read from file metadata.
"""

import os
import numpy as np

from ._core import (
    _read_roms_grid, _read_cmems_var, _detect_cmems_files,
    _compute_ocean_time, _detect_time_units, _parse_time_units,
    _resolve_vgrid_params,
    fill_nan_2d, fill_nans_vertically,
    stretching, set_depth,
    interp_to_roms, mercator2roms_2d, mercator2roms_3d,
    rotate_uv_cgrid, compute_ubar_vbar,
    _write_ic_netcdf, FILL_VAL,
    horizontal_interp, z_to_sigma,
    uv_to_cgrid, rotate_uv, compute_ubar_vbar_legacy,
    _parse_date, _get_time, _match_idx,
)


def cmems_to_roms_ini(roms_grid_file, zeta_file=None, temp_file=None,
                      salt_file=None, u_file=None, v_file=None,
                      data_dir=None, ini_file=None,
                      zeta_var='zos', temp_var='thetao', salt_var='so',
                      u_var='uo', v_var='vo',
                      Vtransform=None, Vstretching=None,
                      theta_s=None, theta_b=None, Tcline=None, N=None,
                      time_ref='1990-01-01', init_date=None,
                      time_index=None):
    """
    Create ROMS initial conditions from CMEMS files.

    Parameters
    ----------
    roms_grid_file : str
        Path to ROMS grid NetCDF file.
    zeta_file, temp_file, salt_file, u_file, v_file : str, optional
        Paths to individual CMEMS variable files.
    data_dir : str, optional
        CMEMS data directory -- auto-detect files (ignored if individual
        files are specified).
    ini_file : str, optional
        Output IC file path. Default: 'roms_ini_cmems.nc'
    zeta_var, temp_var, salt_var, u_var, v_var : str
        CMEMS variable names (or aliases). Auto-resolved.
    Vtransform, Vstretching, theta_s, theta_b, Tcline, N
        ROMS vertical grid parameters.
    time_ref : str, optional
        ROMS time reference date, e.g. '1990-01-01'.
    init_date : str, optional
        Date for the initial condition, e.g. '2025-05-01'.
        If not provided, uses the first time step.
    time_index : int, optional
        Explicit time index. Overrides init_date auto-detection.

    Returns
    -------
    dict with keys: zeta, temp, salt, u, v, ubar, vbar
    """
    print('=' * 60)
    print('CMEMS -> ROMS Initial Condition')
    print('=' * 60)

    # ---- Resolve output file ----
    if ini_file is None:
        ini_file = 'roms_ini_cmems.nc'

    # ---- Auto-detect CMEMS files ----
    print('\n[1] Locating CMEMS data ...')
    if data_dir and not all([zeta_file, temp_file, salt_file, u_file, v_file]):
        detected = _detect_cmems_files(data_dir)
        if not detected:
            raise FileNotFoundError(f'No CMEMS files found in {data_dir}')
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

    # ---- Read ROMS grid ----
    print('\n[2] Reading ROMS grid ...')
    import netCDF4 as nc
    ds = nc.Dataset(roms_grid_file)
    h         = np.array(ds.variables['h'][:], dtype=float)
    lon_rho   = np.array(ds.variables['lon_rho'][:], dtype=float)
    lat_rho   = np.array(ds.variables['lat_rho'][:], dtype=float)
    lon_u     = np.array(ds.variables['lon_u'][:], dtype=float)
    lat_u     = np.array(ds.variables['lat_u'][:], dtype=float)
    lon_v     = np.array(ds.variables['lon_v'][:], dtype=float)
    lat_v     = np.array(ds.variables['lat_v'][:], dtype=float)
    mask_rho  = np.array(ds.variables['mask_rho'][:], dtype=float)
    mask_u    = np.array(ds.variables['mask_u'][:], dtype=float)
    mask_v    = np.array(ds.variables['mask_v'][:], dtype=float)
    angle     = np.array(ds.variables['angle'][:], dtype=float)
    ds.close()

    # ---- Resolve vertical grid params (read from grid if not user-specified) ----
    _vgrid = _resolve_vgrid_params(roms_grid_file, Vtransform, Vstretching,
                                    theta_s, theta_b, Tcline, N)
    Vtransform = _vgrid['Vtransform']
    Vstretching = _vgrid['Vstretching']
    theta_s = _vgrid['theta_s']
    theta_b = _vgrid['theta_b']
    Tcline = _vgrid['Tcline']
    hc = _vgrid['hc']
    N = _vgrid['N']
    for _pn, _pv in _vgrid.items():
        print(f'  {_pn} = {_pv}')

    nlat, nlon = h.shape
    print(f'  Grid: {nlat} x {nlon}')

    # ---- Auto-detect time_ref from CMEMS files ----
    if time_ref is None:
        for fp in [temp_file, zeta_file, salt_file, u_file, v_file]:
            if fp:
                tu, tv = _detect_time_units(fp)
                if tu:
                    _, epoch = _parse_time_units(tu)
                    if epoch:
                        time_ref = epoch.strftime('%Y-%m-%d')
                        print(f'  Auto-detected time_ref from CMEMS: {time_ref}')
                        break
        if time_ref is None:
            time_ref = '1990-01-01'

    # ---- Read CMEMS data ----
    print('\n[3] Reading CMEMS data ...')

    Zeta_data, Zlon, Zlat, _, ztime_info     = _read_cmems_var(zeta_file, zeta_var, 0)
    Temp_data, Tlon, Tlat, Tdepth, ttime_info = _read_cmems_var(temp_file, temp_var, 0)
    Salt_data, _,    _,    _,      _         = _read_cmems_var(salt_file, salt_var, 0)
    Uvel_data, Ulon, Ulat, Udepth, utime_info = _read_cmems_var(u_file, u_var, 0)
    Vvel_data, _,    _,    Vdepth, _         = _read_cmems_var(v_file, v_var, 0)

    # Resolve time index
    if time_index is not None:
        tidx = time_index
    elif init_date is not None:
        tu, tv = ttime_info
        if tu is not None and tv is not None:
            _, tidx = _compute_ocean_time(ttime_info, init_date, time_ref)
        else:
            tidx = 0
        print(f'  init_date={init_date} -> time_index={tidx}')
    else:
        tidx = 0

    # Re-read with correct time index if needed
    if tidx != 0:
        Zeta_data, _, _, _, _       = _read_cmems_var(zeta_file, zeta_var, tidx)
        Temp_data, _, _, Tdepth, _  = _read_cmems_var(temp_file, temp_var, tidx)
        Salt_data, _, _, _, _       = _read_cmems_var(salt_file, salt_var, tidx)
        Uvel_data, _, _, Udepth, _  = _read_cmems_var(u_file, u_var, tidx)
        Vvel_data, _, _, Vdepth, _  = _read_cmems_var(v_file, v_var, tidx)

    # _read_cmems_var returns (lat, lon) or (lat, lon, depth) order
    print(f'  Data shapes: Zeta={Zeta_data.shape}, Temp={Temp_data.shape}')
    print(f'  CMEMS depths: T={len(Tdepth) if Tdepth is not None else 0}, '
          f'U={len(Udepth) if Udepth is not None else 0}, '
          f'V={len(Vdepth) if Vdepth is not None else 0}')

    # ---- Compute ocean_time (timestamp of the selected record) ----
    ocean_time, _ = _compute_ocean_time(ttime_info, init_date, time_ref,
                                        time_index=tidx)
    print(f'\n  ocean_time: {ocean_time:.0f} s  (ref: {time_ref})')

    # ---- Depth processing ----
    if Tdepth is None:
        Tdepth = np.array([5000.0])
    if Udepth is None:
        Udepth = Tdepth.copy()
    if Vdepth is None:
        Vdepth = Tdepth.copy()

    Tdepth = np.unique(Tdepth[Tdepth > 0])
    Udepth = np.unique(Udepth[Udepth > 0])
    Vdepth = np.unique(Vdepth[Vdepth > 0])

    # ---- Step 1: Range check + NaN fill on CMEMS grid ----
    print('\n[4] Filling invalid values ...')
    Temp_data[np.abs(Temp_data) > 50] = np.nan
    Temp_data[Temp_data < -100] = np.nan
    Salt_data[Salt_data < 0] = np.nan
    Salt_data[Salt_data > 50] = np.nan
    Salt_data[np.abs(Salt_data) > 100] = np.nan
    Uvel_data[np.abs(Uvel_data) > 100] = np.nan
    Vvel_data[np.abs(Vvel_data) > 100] = np.nan
    Zeta_data[np.abs(Zeta_data) > 100] = np.nan

    Temp_data = fill_nan_2d(Temp_data)
    Salt_data = fill_nan_2d(Salt_data)
    Uvel_data = fill_nan_2d(Uvel_data)
    Vvel_data = fill_nan_2d(Vvel_data)
    Zeta_data = fill_nan_2d(Zeta_data)

    # ---- Step 2: Extend depth if ROMS deeper than CMEMS ----
    hmax = h.max()
    print(f'  CMEMS max depth: {Tdepth[-1]:.0f}m, ROMS max depth: {hmax:.0f}m')

    if Tdepth[-1] < hmax:
        Tdepth = np.append(Tdepth, [Tdepth[-1] + 200, hmax + 200])
        Temp_data = np.concatenate([Temp_data, Temp_data[:, :, -1:],
                                    Temp_data[:, :, -1:]], axis=2)
        Salt_data = np.concatenate([Salt_data, Salt_data[:, :, -1:],
                                    Salt_data[:, :, -1:]], axis=2)
        Udepth = np.append(Udepth, [Udepth[-1] + 200, hmax + 200])
        Vdepth = np.append(Vdepth, [Vdepth[-1] + 200, hmax + 200])
        Uvel_data = np.concatenate([Uvel_data, np.zeros_like(Uvel_data[:, :, -1:]),
                                    np.zeros_like(Uvel_data[:, :, -1:])], axis=2)
        Vvel_data = np.concatenate([Vvel_data, np.zeros_like(Vvel_data[:, :, -1:]),
                                    np.zeros_like(Vvel_data[:, :, -1:])], axis=2)

    # Trim to depth
    Temp_data = Temp_data[:, :, :len(Tdepth)]
    Salt_data = Salt_data[:, :, :len(Tdepth)]
    Uvel_data = Uvel_data[:, :, :len(Udepth)]
    Vvel_data = Vvel_data[:, :, :len(Vdepth)]

    # Vertical fill
    print('  Filling NaN vertically ...')
    Temp_data = fill_nans_vertically(Temp_data)
    Salt_data = fill_nans_vertically(Salt_data)
    Uvel_data = fill_nans_vertically(Uvel_data)
    Vvel_data = fill_nans_vertically(Vvel_data)

    print(f'  After fill: Temp [{np.nanmin(Temp_data):.2f}, {np.nanmax(Temp_data):.2f}]')

    # ---- Step 3: Interpolation to ROMS grid ----
    print('\n[5] Interpolating to ROMS grid ...')

    # Use zeta lon/lat for all variables (CMEMS should use same grid)
    if Zlon is not None and Zlat is not None:
        clon, clat = Zlon, Zlat
    elif Tlon is not None:
        clon, clat = Tlon, Tlat
    else:
        clon, clat = Ulon, Ulat

    zeta = mercator2roms_2d(Zeta_data, clon, clat, lon_rho, lat_rho)
    print(f'  zeta: [{np.nanmin(zeta):.3f}, {np.nanmax(zeta):.3f}]')

    # ROMS sigma depths
    ssh = np.zeros_like(zeta)
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 1, h, ssh)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 5, h, zeta)
    Hz = z_w[:, :, 1:N + 1] - z_w[:, :, 0:N]

    # 3D interpolation
    print('  Interpolating temp ...')
    temp = mercator2roms_3d(Temp_data, clon, clat, -Tdepth, lon_rho, lat_rho, z_r)
    print('  Interpolating salt ...')
    salt = mercator2roms_3d(Salt_data, clon, clat, -Tdepth, lon_rho, lat_rho, z_r)
    print('  Interpolating u ...')
    Urho = mercator2roms_3d(Uvel_data,
                             Ulon if Ulon is not None else clon,
                             Ulat if Ulat is not None else clat,
                             -Udepth, lon_rho, lat_rho, z_r)
    print('  Interpolating v ...')
    Vrho = mercator2roms_3d(Vvel_data,
                             Ulon if Ulon is not None else clon,
                             Ulat if Ulat is not None else clat,
                             -Vdepth, lon_rho, lat_rho, z_r)

    print(f'  Temp: [{np.nanmin(temp):.2f}, {np.nanmax(temp):.2f}]')
    print(f'  Salt: [{np.nanmin(salt):.2f}, {np.nanmax(salt):.2f}]')

    # ---- Step 4: Velocity rotation + C-grid + barotropic ----
    print('\n[6] Rotating velocity ...')
    u, v = rotate_uv_cgrid(Urho, Vrho, angle)
    ubar, vbar = compute_ubar_vbar(u, v, Hz)

    # ---- Step 5: Land masks ----
    print('Applying masks ...')
    temp[mask_rho == 0] = FILL_VAL
    salt[mask_rho == 0] = FILL_VAL
    zeta[mask_rho == 0] = FILL_VAL
    u[mask_u == 0] = FILL_VAL
    ubar[mask_u == 0] = FILL_VAL
    v[mask_v == 0] = FILL_VAL
    vbar[mask_v == 0] = FILL_VAL

    print(f'  Water temp: [{temp[mask_rho == 1].min():.2f}, {temp[mask_rho == 1].max():.2f}]')

    # ---- Step 6: Write NetCDF ----
    print('\n[7] Writing NetCDF ...')
    _write_ic_netcdf(ini_file, h, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
                     ocean_time, theta_s, theta_b, Tcline, Tcline,
                     N, zeta, ubar, vbar, u, v, temp, salt,
                     Vtransform, Vstretching, time_ref)

    print('=' * 60)
    print(f'Done!  Output: {ini_file}')
    print('=' * 60)
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v,
            'ubar': ubar, 'vbar': vbar}


def mercator_to_roms_ini(roms_grid_file, source_file, ini_file,
                         Vtransform=None, Vstretching=None,
                         theta_s=None, theta_b=None, Tcline=None, N=None,
                         init_date=None, time_ref=None,
                         **source_kwargs):
    """Create ROMS IC from a single multi-variable file (Mercator/HYCOM)."""

    def _read_source(path, time_index=0):
        """Read all variables from a single source file."""
        zeta, lon_z, lat_z, _, _    = _read_cmems_var(path, 'zos', time_index)
        temp, lon_t, lat_t, depths, tinfo = _read_cmems_var(path, 'thetao', time_index)
        salt, _, _, _, _            = _read_cmems_var(path, 'so', time_index)
        u, _, _, _, _               = _read_cmems_var(path, 'uo', time_index)
        v, _, _, _, _               = _read_cmems_var(path, 'vo', time_index)

        return {
            'zeta': zeta,
            'temp': temp,
            'salt': salt,
            'u': u,
            'v': v,
            'lon_1d': lon_t,
            'lat_1d': lat_t,
            'depth': depths,
            'time_info': tinfo,
        }

    from ._core import _parse_date, _get_time, _match_idx, _fill_nan_single, _resolve_vgrid_params

    metrics = _read_roms_grid(roms_grid_file)

    # Resolve vertical grid params
    _vgrid = _resolve_vgrid_params(roms_grid_file, Vtransform, Vstretching,
                                    theta_s, theta_b, Tcline, N)
    Vtransform = _vgrid['Vtransform']
    Vstretching = _vgrid['Vstretching']
    theta_s = _vgrid['theta_s']
    theta_b = _vgrid['theta_b']
    Tcline = _vgrid['Tcline']
    hc = _vgrid['hc']
    N = _vgrid['N']

    # Time matching
    src_time = _get_time(source_file, source_kwargs.get('time_var'))
    if init_date is not None and src_time is not None:
        idx = _match_idx(src_time, init_date)
    else:
        idx = 0

    # Read source
    src = _read_source(source_file, time_index=idx)
    h = metrics['h']
    mask = metrics['mask_rho']

    # ROMS sigma depths
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 1, h, np.zeros_like(h))
    spval = FILL_VAL

    # Horizontal interpolation -> ROMS grid
    zeta = horizontal_interp(src['lon_1d'], src['lat_1d'], src['zeta'],
                              metrics['lon_rho'], metrics['lat_rho'],
                              mask=mask, fill_value=0.0)
    zeta = np.nan_to_num(zeta, nan=0.0)

    # 3D: src is (lat, lon, depth); legacy horiz interp expects (depth, lat, lon)
    temp_t = src['temp'].transpose(2, 0, 1) if src['temp'].ndim == 3 else src['temp']
    salt_t = src['salt'].transpose(2, 0, 1) if src['salt'].ndim == 3 else src['salt']
    u_t = src['u'].transpose(2, 0, 1) if src['u'].ndim == 3 else src['u']
    v_t = src['v'].transpose(2, 0, 1) if src['v'].ndim == 3 else src['v']

    temp_h = horizontal_interp(src['lon_1d'], src['lat_1d'], temp_t,
                                metrics['lon_rho'], metrics['lat_rho'])
    salt_h = horizontal_interp(src['lon_1d'], src['lat_1d'], salt_t,
                                metrics['lon_rho'], metrics['lat_rho'])
    u_h = horizontal_interp(src['lon_1d'], src['lat_1d'], u_t,
                             metrics['lon_rho'], metrics['lat_rho'])
    v_h = horizontal_interp(src['lon_1d'], src['lat_1d'], v_t,
                             metrics['lon_rho'], metrics['lat_rho'])

    # NaN fill
    temp_h = np.nan_to_num(temp_h, nan=0.0)
    salt_h = np.nan_to_num(salt_h, nan=0.0)
    u_h = np.nan_to_num(u_h, nan=0.0)
    v_h = np.nan_to_num(v_h, nan=0.0)

    # Vertical interpolation: source depths -> ROMS sigma
    # z_to_sigma expects z_levels negative-up; source depths are positive-down
    src_depth = -np.asarray(src['depth'], dtype=float)
    z_r_t = z_r.transpose(2, 0, 1)  # (N, nlat, nlon) for legacy interpy

    temp = z_to_sigma(temp_h, src_depth, z_r_t, fill_value=spval)
    salt = z_to_sigma(salt_h, src_depth, z_r_t, fill_value=spval)
    u_rho = z_to_sigma(u_h, src_depth, z_r_t, fill_value=spval)
    v_rho = z_to_sigma(v_h, src_depth, z_r_t, fill_value=spval)

    # Velocity rotation + C-grid + barotropic
    angle = metrics.get('angle', np.zeros_like(metrics['lat_rho']))
    u_rho, v_rho = rotate_uv(u_rho, v_rho, 0.0, angle)

    u, v = uv_to_cgrid(u_rho, v_rho, metrics.get('mask_u'), metrics.get('mask_v'), spval)

    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 5, h, zeta)
    z_w_t = z_w.transpose(2, 0, 1)
    ubar, vbar = compute_ubar_vbar_legacy(u, v, z_w_t,
                                           metrics.get('mask_u'), metrics.get('mask_v'))

    # Ocean time — must be seconds relative to the SAME reference the
    # writer declares (units = 'seconds since {ref} 00:00:00')
    write_ref = time_ref or '1990-01-01'
    tinfo = src.get('time_info')
    if time_ref and tinfo and tinfo[0] is not None:
        ocean_time, _ = _compute_ocean_time(tinfo, init_date, time_ref)
    elif src_time is not None:
        m = __import__('re').search(r'since\s+(.+)', time_ref) if time_ref else None
        ref = _parse_date(m.group(1)) if m else _parse_date(write_ref)
        if ref is None:
            ref = _parse_date('1990-01-01') or 0.0
        ocean_time = float(src_time[idx] - ref)
    else:
        ocean_time = 0.0

    # Write — _write_ic_netcdf expects 3-D fields as (eta, xi, s_rho)
    _write_ic_netcdf(ini_file, h, metrics['lon_rho'], metrics['lat_rho'],
                     metrics['lon_u'], metrics['lat_u'],
                     metrics['lon_v'], metrics['lat_v'],
                     ocean_time, theta_s, theta_b, Tcline, Tcline,
                     N, zeta, ubar, vbar,
                     u.transpose(1, 2, 0), v.transpose(1, 2, 0),
                     temp.transpose(1, 2, 0), salt.transpose(1, 2, 0),
                     Vtransform, Vstretching, write_ref)
    print(f"Written: {ini_file}")
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v,
            'ubar': ubar, 'vbar': vbar}
