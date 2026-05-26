"""
ROMS → ROMS initial and boundary conditions.

Uses an intermediate standard-z grid for conservative vertical remapping.
"""

import os, re
import numpy as np
import netCDF4 as nc4

from ._core import (horizontal_interp, sigma_to_z, z_to_sigma,
                    rotate_uv, uv_to_cgrid, compute_ubar_vbar,
                    write_ic_file, write_bry_file, EDGE_NAMES)
from .cmems_icbc import _parse_date, _get_time, _match_idx, _filter_files

DEFAULT_Z = np.array([
    -7500, -7000, -6500, -6000, -5500, -5000, -4500, -4000, -3500,
    -3000, -2500, -2000, -1750, -1500, -1250, -1000, -900, -800, -700,
    -600, -500, -400, -300, -250, -200, -175, -150, -125, -100, -90,
    -80, -70, -60, -50, -45, -40, -35, -30, -25, -20, -17.5,
    -15, -12.5, -10, -7.5, -5, -2.5, 0
])


def _read_roms_grid(grid_file, hist_file=None):
    """Read ROMS grid into dict."""
    g = nc4.Dataset(grid_file)
    grd = {}
    for v in ['h', 'lon_rho', 'lat_rho', 'mask_rho', 'mask_u', 'mask_v',
              'angle', 'pm', 'pn']:
        if v in g.variables:
            grd[v] = g.variables[v][:]
    grd['mask_rho'] = grd.get('mask_rho', np.ones(grd['h'].shape))
    for p in ['Vtransform', 'Vstretching', 'theta_s', 'theta_b', 'Tcline',
              'hc']:
        if p in g.variables:
            grd[p] = float(g.variables[p][:])
    g.close()

    data = {'N': grd.get('N', 30)}
    if hist_file:
        h = nc4.Dataset(hist_file)
        if 'ocean_time' in h.variables:
            data['ocean_time'] = h.variables['ocean_time'][:]
        if 's_rho' in h.dimensions:
            data['N'] = len(h.dimensions['s_rho'])
        h.close()
    return grd, data


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

    The variable is remapped at rho-points regardless of its native C-grid
    position (u, v, temp, salt all go through rho-point remapping).
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

    var_z = sigma_to_z(src_var, src_z_r, z_levels, fill_value=spval)
    var_z_dst = np.zeros((len(z_levels), dst_lon.shape[0], dst_lon.shape[1]))
    for k in range(len(z_levels)):
        var_z_dst[k] = horizontal_interp(src_lon, src_lat, var_z[k],
                                          dst_lon, dst_lat, mask=mask,
                                          fill_value=spval)
    return z_to_sigma(var_z_dst, z_levels, dst_z_r, fill_value=spval)


# ---------------------------------------------------------------------------
# IC driver
# ---------------------------------------------------------------------------

def roms_to_roms_ini(src_grid_file, src_hist_file, dst_grid_file,
                     ini_file, src_time=0,
                     Vtransform=2, Vstretching=4,
                     theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                     z_levels=None, var_mapping=None,
                     init_date=None, time_ref=None):
    """Create ROMS IC from another ROMS run.

    Parameters
    ----------
    init_date : str or None
        If given, find the closest time step to this date (e.g. '2020-01-15').
        Overrides ``src_time``.
    time_ref : str or None
        Output reference, e.g. ``'seconds since 2000-01-01 00:00:00'``.
    """
    from ..grid import set_depth

    if z_levels is None:
        z_levels = DEFAULT_Z

    src_grd, src_data = _read_roms_grid(src_grid_file, src_hist_file)
    dst_grd, _ = _read_roms_grid(dst_grid_file)

    # Resolve time
    if init_date is not None:
        src_times = _get_time(src_hist_file)
        if src_times is not None:
            src_time = _match_idx(src_times, init_date)
            print(f"  init_date={init_date} → time index {src_time}")
    ocean_time_val = src_times[src_time] if (init_date and src_times is not None) else 0.0

    # Resolve reference epoch for output
    if time_ref:
        m = re.search(r'since\s+(.+)', time_ref)
        ref_epoch = _parse_date(m.group(1)) if m else 0.0
        ocean_time_val = ocean_time_val - ref_epoch
    ocean_time = np.array([ocean_time_val])

    # Source depths
    src_Vt = int(src_grd.get('Vtransform', 1))
    src_Vs = int(src_grd.get('Vstretching', 1))
    src_N = src_data['N']
    src_z_w = set_depth(src_Vt, src_Vs,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N,
                        src_grd['h'], igrid=5)
    src_z_r = set_depth(src_Vt, src_Vs,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N,
                        src_grd['h'], igrid=1)

    # Target depths
    dst_z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=5)
    dst_z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=1)

    spval = 1e37

    # Resolve variable names (allow custom mapping)
    vmap = var_mapping or {}
    v = lambda k: vmap.get(k, k)

    # ---- Remap source rotation angle to target grid first ----
    parent_angle = _remap_var(v('angle'), src_grid_file, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    target_angle = dst_grd.get('angle', np.zeros_like(dst_grd['lat_rho']))

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
    target_angle = dst_grd.get('angle', np.zeros_like(dst_grd['lat_rho']))

    print("Remapping 3D ...")
    temp = _remap_var('temp', src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    salt = _remap_var('salt', src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    zeta = _remap_var('zeta', src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)

    print("Remapping velocity ...")
    u_rho = _remap_var('u', src_hist_file, src_grd, src_z_r, src_z_w,
                       dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    v_rho = _remap_var('v', src_hist_file, src_grd, src_z_r, src_z_w,
                       dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)

    # Rotate: source XI/ETA → true N/E (using remapped parent angle),
    #          then true N/E → target XI/ETA (using target angle)
    u_rho, v_rho = rotate_uv(u_rho, v_rho, parent_angle, target_angle)
    u, v = uv_to_cgrid(u_rho, v_rho, dst_grd.get('mask_u'),
                       dst_grd.get('mask_v'), spval)
    ubar, vbar = compute_ubar_vbar(u, v, dst_z_w, dst_grd.get('mask_u'),
                                   dst_grd.get('mask_v'))

    vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching}
    write_ic_file(ini_file, dst_grd, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar)
    print(f"Written: {ini_file}")


# ---------------------------------------------------------------------------
# BC driver
# ---------------------------------------------------------------------------

def roms_to_roms_bry(src_grid_file, src_hist_files, dst_grid_file,
                     bry_file,
                     Vtransform=2, Vstretching=4,
                     theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                     boundaries=(False, True, True, True),
                     z_levels=None, var_mapping=None,
                     start_date=None, end_date=None,
                     time_ref='seconds since 1970-01-01 00:00:00'):
    """Create ROMS time-dependent BC from another ROMS run.

    Parameters
    ----------
    start_date, end_date : str or None
        Filter time range, e.g. ``start_date='2020-01-01'``.
    time_ref : str
        Reference string for output bry_time.
    """
    from ..grid import set_depth

    if z_levels is None:
        z_levels = DEFAULT_Z

    # Filter by date
    filtered = _filter_files(src_hist_files, start_date, end_date,
                             'ocean_time')
    if not filtered:
        raise ValueError("No source files in specified time range.")

    src_grd, src_data = _read_roms_grid(src_grid_file,
                                        filtered[0][0] if filtered else None)
    dst_grd, _ = _read_roms_grid(dst_grid_file)

    src_Vt = int(src_grd.get('Vtransform', 1))
    src_Vs = int(src_grd.get('Vstretching', 1))
    src_N = src_data['N']
    src_z_w = set_depth(src_Vt, src_Vs,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N,
                        src_grd['h'], igrid=5)
    src_z_r = set_depth(src_Vt, src_Vs,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N,
                        src_grd['h'], igrid=1)

    dst_z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=5)
    dst_z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=1)

    spval = 1e37

    # Resolve variable names
    vmap = var_mapping or {}
    v = lambda k: vmap.get(k, k)

    # ---- Remap source rotation angle to target grid first ----
    parent_angle = _remap_var(v('angle'), src_grid_file, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)
    target_angle = dst_grd.get('angle', np.zeros_like(dst_grd['lat_rho']))
    # Resolve time reference
    m = re.search(r'since\s+(.+)', time_ref) if time_ref else None
    ref_epoch = _parse_date(m.group(1)) if m else 0.0

    # Count and pre-allocate
    total_steps = sum(len(idxs) for _, idxs in filtered)
    bry_time = np.zeros(total_steps)
    t = 0

    for fp, idxs in filtered:
        src_t = _get_time(fp, 'ocean_time')
        for idx in idxs:
            ti = src_t[idx] if src_t is not None else float(t)
            bry_sec = ti - ref_epoch if ref_epoch != 0 else ti
            print(f"\n[{t + 1}/{total_steps}] {os.path.basename(fp)} "
                  f"idx={idx}, time={ti:.0f}s since 1970")

            temp = _remap_var(v('temp'), fp, src_grd, src_z_r, src_z_w,
                          dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)
        salt = _remap_var(v('salt'), src_file, src_grd, src_z_r, src_z_w,
                          dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)
        zeta = _remap_var(v('zeta'), src_file, src_grd, src_z_r, src_z_w,
                          dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)
        u_rho = _remap_var(v('u'), src_file, src_grd, src_z_r, src_z_w,
                           dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)
        v_rho = _remap_var(v('v'), src_file, src_grd, src_z_r, src_z_w,
                           dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)

        u_rho, v_rho = rotate_uv(u_rho, v_rho, parent_angle, target_angle)
        u, v = uv_to_cgrid(u_rho, v_rho, dst_grd.get('mask_u'),
                           dst_grd.get('mask_v'), spval)
        ubar, vbar = compute_ubar_vbar(u, v, dst_z_w,
                                       dst_grd.get('mask_u'),
                                       dst_grd.get('mask_v'))

        # Extract edges
        edge_data = {}
        for name in ['temp', 'salt']:
            f3d = locals()[name]
            edge_data[name] = [None] * 4
            for b, active in enumerate(boundaries):
                if not active:
                    continue
                edge = EDGE_NAMES[b]
                if edge in ('west', 'east'):
                    idx = 0 if edge == 'west' else -1
                    edge_data[name][b] = f3d[:, :, idx]
                else:
                    idx = 0 if edge == 'south' else -1
                    edge_data[name][b] = f3d[:, idx, :]

        for name, f3d, vname in [('u', u, 'u'), ('v', v, 'v')]:
            edge_data[vname] = [None] * 4
            for b, active in enumerate(boundaries):
                if not active:
                    continue
                edge = EDGE_NAMES[b]
                if edge in ('west', 'east'):
                    idx = 0 if edge == 'west' else -1
                    edge_data[vname][b] = f3d[:, :, idx]
                else:
                    idx = 0 if edge == 'south' else -1
                    edge_data[vname][b] = f3d[:, idx, :]

        for name, f2d in [('zeta', zeta), ('ubar', ubar), ('vbar', vbar)]:
            edge_data[name] = [None] * 4
            for b, active in enumerate(boundaries):
                if not active:
                    continue
                edge = EDGE_NAMES[b]
                if edge in ('west', 'east'):
                    idx = 0 if edge == 'west' else -1
                    edge_data[name][b] = f2d[:, idx]
                else:
                    idx = 0 if edge == 'south' else -1
                    edge_data[name][b] = f2d[idx, :]

        # Get time from source
        src_time = _get_var(src_file, 'ocean_time', 0)
        bry_time[t] = float(np.atleast_1d(src_time)[0])

        if t == 0:
            vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching}
            write_bry_file(bry_file, dst_grd, vgrid_params, boundaries,
                           edge_data, bry_time[:t + 1])
        else:
            ds = nc4.Dataset(bry_file, 'a')
            ds.variables['bry_time'][t] = bry_time[t]
            for var_name in edge_data:
                for b, active in enumerate(boundaries):
                    if not active or edge_data[var_name][b] is None:
                        continue
                    edge = EDGE_NAMES[b]
                    key = f'{var_name}_{edge}'
                    ds.variables[key][t] = edge_data[var_name][b]
            ds.close()

    print(f"\nWritten: {bry_file}")