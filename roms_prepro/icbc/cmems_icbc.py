"""
Mercator / HYCOM / CMEMS reanalysis → ROMS initial and boundary conditions.

Usage
-----
IC:  mercator_to_roms_ini(grid, source, ini, init_date='2020-01-15', N=30)
BC:  mercator_to_roms_bry(grid, source_files, bry,
                          start_date='2020-01-01', end_date='2020-01-31', N=30)
"""

import os, re
import numpy as np
import netCDF4 as nc4
from datetime import datetime
from tqdm import tqdm

from ._core import (horizontal_interp, z_to_sigma,
                    rotate_uv, uv_to_cgrid, compute_ubar_vbar,
                    write_ic_file, write_bry_file, EDGE_NAMES)


# ===========================================================================
# Helpers
# ===========================================================================

def _fill_nan_2d(arr, max_pass=5):
    """Replace NaN cells with the nearest valid neighbour (KDTree)."""
    from scipy.spatial import cKDTree
    arr = arr.copy()
    bad = np.isnan(arr)
    if not bad.any():
        return arr
    ok = ~bad
    ny, nx = arr.shape
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    ok_pts = np.column_stack((ii[ok], jj[ok]))
    bad_pts = np.column_stack((ii[bad], jj[bad]))
    if len(ok_pts) == 0:
        return arr
    tree = cKDTree(ok_pts)
    _, idx = tree.query(bad_pts)
    arr[bad] = arr[ok][idx]
    return arr


def _fill_nan(arr):
    """Fill NaN in 3D arrays: horizontal KDTree per level,
    then vertical copy for remaining NaN cells."""
    if arr.ndim == 2:
        return _fill_nan_2d(arr)
    nk, ny, nx = arr.shape
    out = np.zeros_like(arr)
    for k in range(nk):
        out[k] = _fill_nan_2d(arr[k])
    # Vertical fill: any remaining NaN → copy nearest valid above/below
    for k in range(nk):
        layer = out[k]
        bad = np.isnan(layer)
        if not bad.any():
            continue
        ji, ii = np.where(bad)
        for j, i in zip(ji, ii):
            # Search nearest non-NaN level in this column
            for dk in range(1, nk):
                for sign in [-1, 1]:
                    kk = k + dk * sign
                    if 0 <= kk < nk and not np.isnan(out[kk, j, i]):
                        out[k, j, i] = out[kk, j, i]
                        break
                else:
                    continue
                break
    return out

def _parse_date(t):
    """Convert date string / datetime / numeric to seconds since 1970-01-01."""
    if t is None:
        return None
    if isinstance(t, (int, float, np.floating, np.integer)):
        return float(t)
    if isinstance(t, datetime):
        return t.timestamp()
    if isinstance(t, str):
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d-%b-%Y']:
            try:
                dt = datetime.strptime(t, fmt)
                # On Windows, strptime may fail for pre-1970 dates;
                # use manual calculation as fallback
                try:
                    return dt.timestamp()
                except OSError:
                    return (dt - datetime(1970, 1, 1)).total_seconds()
            except ValueError:
                continue
    raise ValueError(f"Cannot parse date: {t}")


def _get_time(file_path, time_var=None):
    """Read time array from a source file. Returns seconds since 1970."""
    ds = nc4.Dataset(file_path, 'r')
    candidates = [time_var] if time_var else ['time', 'time_counter',
                                               'ocean_time']
    t_name = None
    for c in candidates:
        if c and c in ds.variables:
            t_name = c; break
    if t_name is None:
        ds.close(); return None

    t = np.atleast_1d(ds.variables[t_name][:]).astype(float)
    units = getattr(ds.variables[t_name], 'units', '')
    ds.close()

    # Convert to seconds since 1970 if units available
    m = re.search(r'since\s+([\d\-]+\s[\d\:]+|[\d\-]+)', str(units))
    if m:
        ref_epoch = _parse_date(m.group(1))
        if 'days' in str(units).lower():
            t = t * 86400.0 + ref_epoch
        elif 'hours' in str(units).lower():
            t = t * 3600.0 + ref_epoch
        else:
            t = t + ref_epoch
    return t


def _match_idx(source_times, target_date):
    """Find closest time index to target_date."""
    target = _parse_date(target_date)
    if target is None:
        return 0
    return int(np.argmin(np.abs(source_times - target)))


def _filter_files(source_files, start_date=None, end_date=None,
                  time_var=None):
    """Filter files to [start_date, end_date]. Returns [(fp, [indices])]."""
    t0 = _parse_date(start_date) if start_date else None
    t1 = _parse_date(end_date) if end_date else None
    filtered = []
    for fp in sorted(source_files):
        st = _get_time(fp, time_var)
        if st is None:
            filtered.append((fp, [0])); continue
        idxs = [i for i, ti in enumerate(st)
                if (t0 is None or ti >= t0) and (t1 is None or ti <= t1)]
        if idxs:
            filtered.append((fp, idxs))
    return filtered


# ===========================================================================
# Source reader
# ===========================================================================

def _read_source(file_path,
                 lon_var=None, lat_var=None, depth_var=None,
                 time_var=None, time_index=0,
                 zeta_var=None, temp_var=None, salt_var=None,
                 u_var=None, v_var=None):
    """Read a Mercator/HYCOM/CMEMS NetCDF file with auto-detection."""
    ds = nc4.Dataset(file_path, 'r')

    def f(candidates, override):
        if override and override in ds.variables:
            return override
        for c in candidates:
            if c in ds.variables:
                return c
        raise KeyError(f"none of {candidates} found")

    out = {}
    out['lon'] = ds.variables[f(['longitude','lon'], lon_var)][:]
    out['lat'] = ds.variables[f(['latitude','lat'], lat_var)][:]
    out['depth'] = ds.variables[f(['depth','lev','deptht'], depth_var)][:]
    # Mercator/HYCOM/CMEMS store depth as positive downward; ROMS uses
    # negative values (below mean sea level). Flip sign here.
    out['depth'] = -np.abs(np.asarray(out['depth'], dtype=float))

    # Keep 1-D axes for fast RegularGridInterpolator
    out['lon_1d'] = np.asarray(out['lon'], dtype=float).ravel()
    out['lat_1d'] = np.asarray(out['lat'], dtype=float).ravel()

    for vn, cands, ov in [
                ('zeta', ['zos','ssh','surf_el'], zeta_var),
                ('temp', ['thetao','water_temp','votemper','temperature'], temp_var),
                ('salt', ['so','salinity','vosaline'], salt_var),
                ('u',    ['uo','water_u','vozocrtx'], u_var),
                ('v',    ['vo','water_v','vomecrty'], v_var),
                ]:
            vname = f(cands, ov)
            arr = ds.variables[vname][:]
            var_dims = ds.variables[vname].dimensions
            time_on_first = (
                len(var_dims) >= 3 and
                var_dims[0] in ds.dimensions and
                'time' in var_dims[0].lower()
            )
            if time_on_first and arr.shape[0] > 1:
                arr = arr[min(time_index, arr.shape[0] - 1)]
            # Convert masked arrays to plain arrays (mask → NaN)
            if hasattr(arr, 'mask'):
                arr = arr.filled(np.nan)
            out[vn] = arr

    ds.close()
    return out


# ===========================================================================
# Grid reader
# ===========================================================================

def _read_roms_grid(grid_file):
    g = nc4.Dataset(grid_file)
    m = {}
    for v in ['h','lon_rho','lat_rho','mask_rho','mask_u','mask_v','angle']:
        if v in g.variables:
            m[v] = g.variables[v][:]
    ny, nx = m['lon_rho'].shape
    for e, i in [('west',0),('east',-1)]:
        m[f'lon_rho_{e}'] = m['lon_rho'][:,i]
        m[f'lat_rho_{e}'] = m['lat_rho'][:,i]
    for e, i in [('south',0),('north',-1)]:
        m[f'lon_rho_{e}'] = m['lon_rho'][i,:]
        m[f'lat_rho_{e}'] = m['lat_rho'][i,:]
    m['mask_rho'] = m.get('mask_rho', np.ones(m['h'].shape))
    g.close()
    return m


# ===========================================================================
# IC driver
# ===========================================================================

def mercator_to_roms_ini(roms_grid_file, source_file, ini_file,
                         Vtransform=2, Vstretching=4,
                         theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                         init_date=None, time_ref=None,
                         **source_kwargs):
    """Create ROMS IC from a Mercator/HYCOM/CMEMS file.

    Parameters
    ----------
    init_date : str or datetime or None
        Date for the IC, e.g. ``'2020-01-15'``. If None, uses the first
        time step in the source file.
    time_ref : str or None
        Output time reference, e.g. ``'seconds since 2000-01-01 00:00:00'``.
        If None, ocean_time is set to 0 (relative to source time).

    Source variable override kwargs:
    ``temp_var='thetao'``, ``lon_var='nav_lon'``, etc.
    """
    from ..grid import set_depth
    metrics = _read_roms_grid(roms_grid_file)

    # Resolve time index
    src_time = _get_time(source_file, source_kwargs.get('time_var'))
    if init_date is not None:
        idx = _match_idx(src_time, init_date)
        print(f"  init_date={init_date} → time index {idx} "
              f"(source time = {src_time[idx]:.0f} s since 1970)")
    else:
        idx = 0
        print(f"  using first time step (index 0)")

    src = _read_source(source_file, time_index=idx, **source_kwargs)
    h = metrics['h']; mask = metrics['mask_rho']
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, igrid=1)
    spval = 1e37

    zeta = horizontal_interp(src['lon_1d'], src['lat_1d'], src['zeta'],
                              metrics['lon_rho'], metrics['lat_rho'],
                              mask=mask, fill_value=0.0)
    temp_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['temp'],
                                metrics['lon_rho'], metrics['lat_rho'])
    salt_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['salt'],
                                metrics['lon_rho'], metrics['lat_rho'])
    u_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['u'],
                             metrics['lon_rho'], metrics['lat_rho'])
    v_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['v'],
                             metrics['lon_rho'], metrics['lat_rho'])

    # Fill NaN near coasts
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
    u, v = uv_to_cgrid(u_rho, v_rho, metrics.get('mask_u'),
                       metrics.get('mask_v'), spval)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, igrid=5)
    ubar, vbar = compute_ubar_vbar(u, v, z_w, metrics.get('mask_u'),
                                   metrics.get('mask_v'))

    # Ocean time
    if time_ref and src_time is not None:
        ref = _parse_date(re.search(r'since\s+(.+)', time_ref).group(1)
                          if 'since' in time_ref else time_ref)
        ocean_time = np.array([src_time[idx] - ref])
    else:
        ocean_time = np.array([0.0])

    from ..grid import stretching
    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, kgrid=0)
    vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                    'theta_s': theta_s, 'theta_b': theta_b,
                    'Tcline': Tcline, 'hc': Tcline,
                    's_rho': s_rho, 'Cs_r': Cs_r}
    write_ic_file(ini_file, metrics, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar)
    print(f"Written: {ini_file}")
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v}


# ===========================================================================
# BC driver
# ===========================================================================

def mercator_to_roms_bry(roms_grid_file, source_files, bry_file,
                         Vtransform=2, Vstretching=4,
                         theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                         boundaries=(False, True, True, True),
                         start_date=None, end_date=None,
                         time_ref='seconds since 1970-01-01 00:00:00',
                         **source_kwargs):
    """Create ROMS time-dependent BC from Mercator/HYCOM files.

    Parameters
    ----------
    source_files : list of str
        Source file paths (will be filtered by date if start/end given).
    start_date, end_date : str or None
        Filter time range, e.g. ``start_date='2020-01-01'``.
    time_ref : str
        Reference for output bry_time, e.g.
        ``'seconds since 2000-01-01 00:00:00'``.
    """
    from ..grid import set_depth
    metrics = _read_roms_grid(roms_grid_file)
    h = metrics['h']; mask = metrics['mask_rho']
    angle = metrics.get('angle', np.zeros_like(metrics['lat_rho']))
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, igrid=1)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, igrid=5)
    spval = 1e37

    # Filter by date
    tv = source_kwargs.get('time_var', None)
    filtered = _filter_files(source_files, start_date, end_date, tv)
    if not filtered:
        raise ValueError("No source files in the specified time range.")

    # Resolve reference epoch
    m = re.search(r'since\s+(.+)', time_ref) if time_ref else None
    ref_epoch = _parse_date(m.group(1)) if m else 0.0

    # Count total time steps
    total_steps = sum(len(idxs) for _, idxs in filtered)
    t = 0
    bry_time = np.zeros(total_steps)

    pbar = tqdm(total=total_steps, desc='BC time steps', unit='step')
    for fp, idxs in filtered:
        src_t = _get_time(fp, tv)
        for idx in idxs:
            ti = src_t[idx] if src_t is not None else float(t)
            bry_sec = ti - ref_epoch if ref_epoch != 0 else ti
            pbar.set_postfix_str(f'{os.path.basename(fp)} idx={idx}')

            src = _read_source(fp, time_index=idx, time_var=tv,
                               lon_var=source_kwargs.get('lon_var'),
                               lat_var=source_kwargs.get('lat_var'),
                               depth_var=source_kwargs.get('depth_var'),
                               zeta_var=source_kwargs.get('zeta_var'),
                               temp_var=source_kwargs.get('temp_var'),
                               salt_var=source_kwargs.get('salt_var'),
                               u_var=source_kwargs.get('u_var'),
                               v_var=source_kwargs.get('v_var'))

            zeta = horizontal_interp(src['lon_1d'], src['lat_1d'], src['zeta'],
                                      metrics['lon_rho'], metrics['lat_rho'],
                                      mask=mask, fill_value=0.0)
            temp_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['temp'],
                                        metrics['lon_rho'], metrics['lat_rho'])
            salt_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['salt'],
                                        metrics['lon_rho'], metrics['lat_rho'])
            u_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['u'],
                                     metrics['lon_rho'], metrics['lat_rho'])
            v_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['v'],
                                      metrics['lon_rho'], metrics['lat_rho'])
            # Fill NaN near coasts
            temp_h = _fill_nan(temp_h)
            salt_h = _fill_nan(salt_h)
            u_h = _fill_nan(u_h)
            v_h = _fill_nan(v_h)
            temp_s = z_to_sigma(temp_h, src['depth'], z_r, fill_value=spval)
            salt_s = z_to_sigma(salt_h, src['depth'], z_r, fill_value=spval)
            u_rho = z_to_sigma(u_h, src['depth'], z_r, fill_value=spval)
            v_rho = z_to_sigma(v_h, src['depth'], z_r, fill_value=spval)

            u_rho, v_rho = rotate_uv(u_rho, v_rho, 0.0, angle)
            u_cg, v_cg = uv_to_cgrid(u_rho, v_rho,
                                      metrics.get('mask_u'),
                                      metrics.get('mask_v'), spval)
            ubar, vbar = compute_ubar_vbar(u_cg, v_cg, z_w,
                                           metrics.get('mask_u'),
                                           metrics.get('mask_v'))

            # Extract edges
            edge_data = {}
            for name, data in [
                    ('temp', temp_s), ('salt', salt_s),
                    ('u', u_cg), ('v', v_cg),
                    ('zeta', zeta), ('ubar', ubar), ('vbar', vbar)]:
                edge_data[name] = [None] * 4
                for b, active in enumerate(boundaries):
                    if not active: continue
                    edge = EDGE_NAMES[b]
                    if edge in ('west', 'east'):
                        ix = 0 if edge == 'west' else -1
                        edge_data[name][b] = data[:, :, ix] if data.ndim == 3 else data[:, ix]
                    else:
                        ix = 0 if edge == 'south' else -1
                        edge_data[name][b] = data[:, ix, :] if data.ndim == 3 else data[ix, :]

            bry_time[t] = bry_sec

            if t == 0:
                # Compute S-coordinate parameters for the NC file
                from ..grid import stretching
                s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, kgrid=0)
                vp = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                      'theta_s': theta_s, 'theta_b': theta_b,
                      'Tcline': Tcline, 'hc': Tcline,
                      's_rho': s_rho, 'Cs_r': Cs_r}
                write_bry_file(bry_file, metrics, vp, boundaries,
                               edge_data, bry_time[:t + 1])
            else:
                ds = nc4.Dataset(bry_file, 'a')
                ds.variables['bry_time'][t] = bry_time[t]
                for vn in edge_data:
                    for b, active in enumerate(boundaries):
                        if not active or edge_data[vn][b] is None: continue
                        ds.variables[f'{vn}_{EDGE_NAMES[b]}'][t] = edge_data[vn][b]
                ds.close()
            t += 1
            pbar.update(1)

    pbar.close()
    print(f"Written: {bry_file}  ({total_steps} time steps)")