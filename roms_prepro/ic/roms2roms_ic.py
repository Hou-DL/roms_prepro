"""
ROMS -> ROMS initial conditions via intermediate standard-z grid.

Source ROMS history file should contain: temp, salt, zeta, u, v, angle
and dimensions: ocean_time, s_rho, eta_rho, xi_rho, etc.

Supports single file path or directory (auto-search *.nc).
Time reference auto-read from source ocean_time units.
"""

import os
import re
import glob
from datetime import datetime as _dt
import numpy as np
import netCDF4 as nc4

from ._core import (
    _read_roms_grid, _resolve_vgrid_params, set_depth, stretching,
    horizontal_interp, sigma_to_z, z_to_sigma,
    rotate_uv, uv_to_cgrid, compute_ubar_vbar_legacy,
    _detect_time_units, _parse_date, _get_time, _match_idx,
    _write_ic_netcdf, FILL_VAL,
)

DEFAULT_Z = np.array([
    -7500, -7000, -6500, -6000, -5500, -5000, -4500, -4000, -3500,
    -3000, -2500, -2000, -1750, -1500, -1250, -1000, -900, -800, -700,
    -600, -500, -400, -300, -250, -200, -175, -150, -125, -100, -90,
    -80, -70, -60, -50, -45, -40, -35, -30, -25, -20, -17.5,
    -15, -12.5, -10, -7.5, -5, -2.5, 0
])


def _get_var(filename, varname, time_idx=0):
    """Read one variable from a ROMS file."""
    ds = nc4.Dataset(filename)
    arr = ds.variables[varname][:]
    if arr.ndim == 4:
        arr = arr[time_idx]
    ds.close()
    return arr


def _remap_var(var_name, src_file, src_grd, src_z_r, src_z_w,
               dst_grd, dst_z_r, dst_z_w, z_levels,
               src_time=0, spval=1e37):
    """
    Remap one variable from source ROMS to target ROMS via intermediate Z.

    src_z_r, src_z_w: sigma depths for source (nlat, nlon, N) and (nlat, nlon, N+1)
    dst_z_r, dst_z_w: same for target
    z_levels: 1D array of standard z-levels (negative, descending)
    """
    src_var = _get_var(src_file, var_name, src_time)
    ndim = 3 if src_var.ndim == 3 else 2

    src_lon = src_grd['lon_rho']
    src_lat = src_grd['lat_rho']
    dst_lon = dst_grd['lon_rho']
    dst_lat = dst_grd['lat_rho']
    mask = dst_grd['mask_rho']

    if ndim == 2:
        return horizontal_interp(src_lon, src_lat, src_var,
                                 dst_lon, dst_lat, mask=mask,
                                 fill_value=spval)

    # 3D: sigma -> z -> horizontal -> sigma
    # src_z_r is (nlat_src, nlon_src, N), need (N, nlat_src, nlon_src)
    src_z_r_t = src_z_r.transpose(2, 0, 1)  # (N, nlat, nlon)
    dst_z_r_t = dst_z_r.transpose(2, 0, 1)

    var_z = sigma_to_z(src_var, src_z_r_t, z_levels, fill_value=spval)
    var_z_dst = np.zeros((len(z_levels), dst_lon.shape[0], dst_lon.shape[1]))
    for k in range(len(z_levels)):
        var_z_dst[k] = horizontal_interp(src_lon, src_lat, var_z[k],
                                          dst_lon, dst_lat, mask=mask,
                                          fill_value=spval)
    return z_to_sigma(var_z_dst, z_levels, dst_z_r_t, fill_value=spval)


def _detect_roms_time_ref(src_file):
    """Auto-detect time reference from ROMS ocean_time variable."""
    try:
        ds = nc4.Dataset(src_file, 'r')
        time_var = None
        for candidate in ['ocean_time', 'time', 'time_counter']:
            if candidate in ds.variables:
                time_var = ds.variables[candidate]
                break
        if time_var is None:
            ds.close()
            return None
        units = getattr(time_var, 'units', '')
        ds.close()
        m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', units)
        if m:
            ref_str = m.group(1).strip()
            try:
                dt = __import__('datetime').datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    dt = __import__('datetime').datetime.strptime(ref_str, '%Y-%m-%d')
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    return ref_str
        return None
    except Exception:
        return None


def _detect_roms_files(src_hist_file=None, src_hist_dir=None):
    """Auto-detect ROMS history files."""
    if src_hist_file and os.path.isfile(src_hist_file):
        return [src_hist_file]
    if src_hist_dir and os.path.isdir(src_hist_dir):
        patterns = ['*.nc', 'ocean_his_*.nc', 'ocean_avg_*.nc', 'roms_his_*.nc']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(src_hist_dir, pattern)))
        if files:
            return sorted(set(files))
    return []


def roms_to_roms_ini(src_grid_file, src_hist_file=None, dst_grid_file=None,
                     ini_file=None, src_time=0,
                     Vtransform=None, Vstretching=None,
                     theta_s=None, theta_b=None, Tcline=None, N=None,
                     z_levels=None, var_mapping=None,
                     init_date=None, time_ref=None,
                     src_hist_dir=None):
    """
    Create ROMS IC from another ROMS run (nesting).

    Parameters
    ----------
    src_grid_file : str
        Source (parent) ROMS grid file.
    src_hist_file : str, optional
        Source ROMS history file path.
    src_hist_dir : str, optional
        Source ROMS history directory (auto-search).
    dst_grid_file : str
        Target (child) ROMS grid file.
    ini_file : str
        Output IC file path.
    src_time : int
        Time index in source file.
    Vtransform, Vstretching, theta_s, theta_b, Tcline, N
        Target ROMS vertical grid parameters.
    z_levels : ndarray, optional
        Intermediate z-levels. Default: DEFAULT_Z.
    var_mapping : dict, optional
        Variable name mapping {'temp': 'temperature', ...}.
    init_date : str, optional
        Which date's IC to make, e.g. '2025-05-01'.
    time_ref : str, optional
        Time reference, auto-read from source file if None.
    """
    if z_levels is None:
        z_levels = DEFAULT_Z

    if ini_file is None:
        ini_file = 'roms_ini_nested.nc'

    # Auto-detect files
    src_files = _detect_roms_files(src_hist_file, src_hist_dir)
    if not src_files:
        raise FileNotFoundError('No ROMS history files found')
    first_file = src_files[0]

    # Auto-detect time reference
    if time_ref is None:
        time_ref = _detect_roms_time_ref(first_file)
        if time_ref:
            print(f'  Auto-detected time reference: {time_ref}')
        else:
            time_ref = '1970-01-01'

    # Read grids
    src_grd = _read_roms_grid(src_grid_file)
    dst_grd = _read_roms_grid(dst_grid_file)

    # Resolve vertical grid params from dst grid if not user-specified
    _vgrid = _resolve_vgrid_params(dst_grid_file, Vtransform, Vstretching,
                                    theta_s, theta_b, Tcline, N)
    Vtransform = _vgrid['Vtransform']
    Vstretching = _vgrid['Vstretching']
    theta_s = _vgrid['theta_s']
    theta_b = _vgrid['theta_b']
    Tcline = _vgrid['Tcline']
    hc = _vgrid['hc']
    N = _vgrid['N']

    # Time matching
    src_times = _get_time(first_file)
    if init_date is not None and src_times is not None:
        src_time = _match_idx(src_times, init_date)
        print(f"  init_date={init_date} -> time index {src_time}")
    ocean_time_val = src_times[src_time] if (init_date and src_times is not None) else 0.0

    # Compute ocean_time
    from datetime import datetime
    m = re.search(r'since\s+(.+)', time_ref)
    if m:
        ref_epoch = _parse_date(m.group(1))
    else:
        ref_epoch = _parse_date(time_ref)
    if ref_epoch is not None and src_times is not None:
        ocean_time_val = float(src_times[src_time] - ref_epoch)

    ocean_time = float(ocean_time_val)

    # Source vertical grid
    src_N = N
    ds_tmp = nc4.Dataset(first_file)
    if 's_rho' in ds_tmp.dimensions:
        src_N = len(ds_tmp.dimensions['s_rho'])
    ds_tmp.close()

    src_z_w = set_depth(2, 4,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N, 5,
                        src_grd['h'], np.zeros_like(src_grd['h']))
    src_z_r = set_depth(2, 4,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N, 1,
                        src_grd['h'], np.zeros_like(src_grd['h']))

    # Target vertical grid
    dst_z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 5,
                        dst_grd['h'], np.zeros_like(dst_grd['h']))
    dst_z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N, 1,
                        dst_grd['h'], np.zeros_like(dst_grd['h']))

    spval = FILL_VAL

    # Remap source angle for velocity rotation
    parent_angle = _remap_var('angle', src_grid_file, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    target_angle = dst_grd.get('angle', np.zeros_like(dst_grd['lat_rho']))

    vmap = var_mapping or {}
    v = lambda k: vmap.get(k, k)

    print("Remapping 3D ...")
    temp = _remap_var(v('temp'), src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    salt = _remap_var(v('salt'), src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    zeta = _remap_var(v('zeta'), src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)

    print("Remapping velocity ...")
    u_rho = _remap_var(v('u'), src_hist_file, src_grd, src_z_r, src_z_w,
                       dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    v_rho = _remap_var(v('v'), src_hist_file, src_grd, src_z_r, src_z_w,
                       dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)

    # Rotate: source -> true N/E -> target
    u_rho, v_rho = rotate_uv(u_rho, v_rho, parent_angle, target_angle)
    u, v = uv_to_cgrid(u_rho, v_rho, dst_grd.get('mask_u'),
                       dst_grd.get('mask_v'), spval)
    ubar, vbar = compute_ubar_vbar_legacy(u, v, dst_z_w.transpose(2, 0, 1),
                                           dst_grd.get('mask_u'),
                                           dst_grd.get('mask_v'))

    # Land masks
    N_target = dst_z_r.shape[2]
    wm3d_rho = np.tile(dst_grd['mask_rho'], (N_target, 1, 1))
    temp[wm3d_rho == 0] = spval
    salt[wm3d_rho == 0] = spval
    zeta[dst_grd['mask_rho'] == 0] = spval

    wm3d_u = np.tile(dst_grd.get('mask_u', np.ones_like(dst_grd['mask_rho'])),
                     (N_target, 1, 1))
    u[wm3d_u == 0] = spval
    ubar[dst_grd.get('mask_u', np.ones_like(dst_grd['mask_rho'])) == 0] = spval

    wm3d_v = np.tile(dst_grd.get('mask_v', np.ones_like(dst_grd['mask_rho'])),
                     (N_target, 1, 1))
    v[wm3d_v == 0] = spval
    vbar[dst_grd.get('mask_v', np.ones_like(dst_grd['mask_rho'])) == 0] = spval

    # Write output — normalize time_ref for writer (it appends ' 00:00:00')
    try:
        _ref_dt = _dt.strptime(time_ref, '%Y-%m-%d %H:%M:%S')
        time_ref_out = _ref_dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        try:
            _ref_dt = _dt.strptime(time_ref, '%Y-%m-%d')
            time_ref_out = time_ref
        except (ValueError, TypeError):
            time_ref_out = '1990-01-01'

    _write_ic_netcdf(ini_file, dst_grd['h'],
                     dst_grd['lon_rho'], dst_grd['lat_rho'],
                     dst_grd['lon_u'], dst_grd['lat_u'],
                     dst_grd['lon_v'], dst_grd['lat_v'],
                     ocean_time, theta_s, theta_b, Tcline, Tcline,
                     N, zeta, ubar, vbar, u, v, temp, salt,
                     Vtransform, Vstretching, time_ref_out)
    print(f"Written: {ini_file}")
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v,
            'ubar': ubar, 'vbar': vbar}
