"""
Core interpolation routines for CMEMS → ROMS.

Uses bc.roms_tools functions (matching reference d_cmems2roms_py.py)
to ensure consistent behavior.
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Grid reading (thin wrapper around bc.roms_tools)
# ---------------------------------------------------------------------------

def _read_roms_grid(grid_file):
    """Read ROMS grid into dict."""
    import netCDF4 as nc4
    g = nc4.Dataset(grid_file)
    m = {}
    for v in ['h', 'lon_rho', 'lat_rho', 'mask_rho', 'mask_u', 'mask_v',
              'angle', 'lon_u', 'lat_u', 'lon_v', 'lat_v']:
        if v in g.variables:
            m[v] = g.variables[v][:]
    g.close()
    return m


# ---------------------------------------------------------------------------
# CMEMS reading
# ---------------------------------------------------------------------------

def _read_cmems_var(file_path, var_name, time_index):
    """
    Read a single variable from CMEMS NetCDF.

    Returns
    -------
    data : ndarray (nlat, nlon) or (nlat, nlon, ndepth)
    lon_1d : ndarray
    lat_1d : ndarray
    depth : ndarray or None (positive depths, or None for 2D)
    time_units : str or None
    """
    import netCDF4 as nc4
    ds = nc4.Dataset(file_path, 'r')

    lon_name = next((n for n in ['longitude', 'lon', 'nav_lon'] if n in ds.dimensions), None)
    lat_name = next((n for n in ['latitude', 'lat', 'nav_lat'] if n in ds.dimensions), None)
    depth_name = next((n for n in ['depth', 'lev', 'deptht'] if n in ds.dimensions), None)
    time_name = next((n for n in ['time', 'time_counter', 't'] if n in ds.dimensions), None)

    lon_1d = np.asarray(ds.variables[lon_name][:], dtype=float) if lon_name else None
    lat_1d = np.asarray(ds.variables[lat_name][:], dtype=float) if lat_name else None

    # Auto-detect variable name
    if var_name not in ds.variables:
        for aliases in [['zos', 'sla', 'adt', 'ssh', 'surf_el'],
                        ['thetao', 't', 'temperature', 'votemper'],
                        ['so', 's', 'salinity', 'vosaline'],
                        ['uo', 'u', 'water_u', 'vozocrtx'],
                        ['vo', 'v', 'water_v', 'vomecrty']]:
            if var_name in aliases:
                for alias in aliases:
                    if alias in ds.variables:
                        var_name = alias
                        break
                break

    v = ds.variables[var_name]

    # Read one time step
    dims = v.dimensions
    has_time = (time_name and time_name in dims and len(dims) >= 3)
    if has_time and v.shape[0] > 1:
        data = np.array(v[min(time_index, v.shape[0] - 1)], dtype=float)
    else:
        data = np.array(v[:], dtype=float)

    # Handle _FillValue / scale_factor / add_offset
    fv = getattr(v, '_FillValue', None)
    if fv is not None:
        data[data == fv] = np.nan
    sf = getattr(v, 'scale_factor', None)
    so = getattr(v, 'add_offset', None)
    if sf is not None:
        data = data * (sf if sf != 0 else 1.0) + (so if so is not None else 0.0)

    # Read depth (keep positive)
    depth = None
    if depth_name and depth_name in ds.variables:
        depth = np.abs(np.asarray(ds.variables[depth_name][:], dtype=float))

    # Read time units
    time_units = None
    if time_name and time_name in ds.variables:
        time_units = getattr(ds.variables[time_name], 'units', None)

    ds.close()
    return data, lon_1d, lat_1d, depth, time_units


# ---------------------------------------------------------------------------
# CMEMS NaN filling (reference implementation)
# ---------------------------------------------------------------------------

def _fill_nan_single(layer):
    """Fill single layer NaN using distance_transform_edt."""
    nan_mask = np.isnan(layer)
    if not nan_mask.any() or nan_mask.all():
        return layer
    _, idx = distance_transform_edt(~nan_mask, return_indices=True)
    result = layer.copy()
    result[nan_mask] = layer[idx[0][nan_mask], idx[1][nan_mask]]
    return result


def fill_nan_2d(data):
    """Fill NaN in 2D/3D data using distance_transform_edt per layer."""
    result = data.copy()
    if result.ndim == 2:
        return _fill_nan_single(result)
    for k in range(result.shape[2]):
        result[:, :, k] = _fill_nan_single(result[:, :, k].copy())
    return result


def fill_nans_vertically(data):
    """Fill NaN/0 columns vertically with nearest valid neighbor."""
    nlat, nlon, ndepth = data.shape
    for i in range(nlat):
        for j in range(nlon):
            col = data[i, j, :]
            valid = np.where(~np.isnan(col) & (col != 0))[0]
            nan_idx = np.where(np.isnan(col) | (col == 0))[0]
            if len(valid) > 0 and len(nan_idx) > 0:
                for k in nan_idx:
                    nearest = valid[np.argmin(np.abs(valid - k))]
                    col[k] = col[nearest]
                data[i, j, :] = col
    return data


# ---------------------------------------------------------------------------
# Interpolation (reference implementation)
# ---------------------------------------------------------------------------

def interp_to_roms(data, lon_1d, lat_1d, lon_rho, lat_rho):
    """
    Linear interpolation: CMEMS regular grid → ROMS grid.
    data: (nlat_src, nlon_src)
    Returns: same shape as lon_rho / lat_rho.
    """
    lat_s, lon_s = lat_1d.copy(), lon_1d.copy()
    d = data.copy()
    if lat_s[0] > lat_s[-1]:
        lat_s = lat_s[::-1]
        d = d[::-1, :]
    if lon_s[0] > lon_s[-1]:
        lon_s = lon_s[::-1]
        d = d[:, ::-1]

    interp = RegularGridInterpolator(
        (lat_s, lon_s), d,
        method='linear', bounds_error=False, fill_value=np.nan)
    return interp((lat_rho, lon_rho))


def mercator2roms_2d(Zeta, lon_1d, lat_1d, lon_rho, lat_rho):
    """2D interpolation with de-meaning."""
    zeta_mean = np.nanmean(Zeta)
    Zeta_dm = Zeta - zeta_mean
    Fout = interp_to_roms(Zeta_dm, lon_1d, lat_1d, lon_rho, lat_rho)
    Fout = _fill_nan_single(Fout)
    Fout += zeta_mean
    return Fout


def mercator2roms_3d(Finp, lon_1d, lat_1d, depth_in, lon_rho, lat_rho, z_r):
    """
    3D interpolation: horizontal per layer, vertical per point.

    Parameters
    ----------
    Finp : (nlat_src, nlon_src, ndepth_src)
    depth_in : (ndepth_src,) negative depths
    z_r : (nlat_roms, nlon_roms, N) ROMS sigma depths
    """
    nlat_r, nlon_r, N = z_r.shape
    ndepth_src = len(depth_in)

    # 1. Horizontal interpolation per layer
    Flev = np.full((nlat_r, nlon_r, ndepth_src), np.nan)
    for k in range(ndepth_src):
        Flev[:, :, k] = interp_to_roms(Finp[:, :, k], lon_1d, lat_1d, lon_rho, lat_rho)

    # 2. Vertical interpolation per point
    Fout = np.zeros((nlat_r, nlon_r, N))
    for i in range(nlat_r):
        for j in range(nlon_r):
            src = Flev[i, j, :]
            valid = ~np.isnan(src)
            if np.sum(valid) < 2:
                continue
            src_z = depth_in[valid]
            src_v = src[valid]
            target_z = z_r[i, j, :]
            target_z = np.clip(target_z, src_z[-1], src_z[0])
            Fout[i, j, :] = np.interp(target_z, src_z, src_v)

    return Fout


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_cmems_files(data_dir):
    """Auto-detect CMEMS files in directory. Returns {var: path}."""
    result = {}
    patterns = {
        'zos': ['cmems_zos_*.nc', '*zos*.nc', '*ssh*.nc'],
        'thetao': ['cmems_thetao_*.nc', '*thetao*.nc', '*temp*.nc'],
        'so': ['cmems_so_*.nc', '*so_*.nc', '*salt*.nc'],
        'uo': ['cmems_uo_*.nc', '*uo*.nc'],
        'vo': ['cmems_vo_*.nc', '*vo*.nc'],
    }
    for std_name, pats in patterns.items():
        for pat in pats:
            matches = sorted(__import__('glob').glob(os.path.join(data_dir, pat)))
            if matches:
                result[std_name] = matches[0]
                break
    return result


def _detect_time_units(temp_file):
    """Read time units and values from CMEMS file."""
    import netCDF4 as nc4
    try:
        ds = nc4.Dataset(temp_file, 'r')
        for tname in ['time', 'time_counter', 't']:
            if tname in ds.variables:
                units = getattr(ds.variables[tname], 'units', '')
                values = ds.variables[tname][:]
                ds.close()
                return units, values
        ds.close()
    except Exception:
        pass
    return None, None


def _compute_ocean_time(time_units, time_values, init_date, roms_time_ref):
    """Compute ocean_time in seconds from CMEMS time and ROMS reference."""
    if time_units is None or time_values is None:
        return 0.0

    m = __import__('re').search(r'since\s+([\d\-]+\s*[\d:]*)', time_units)
    if not m:
        return 0.0

    ref_str = m.group(1).strip()
    try:
        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d')

    idx = _match_time_idx(time_units, time_values, init_date)
    current_dt = ref_dt + timedelta(seconds=float(time_values[idx]))

    ref_dt = datetime.strptime(roms_time_ref, '%Y-%m-%d')
    return (current_dt - ref_dt).total_seconds()


def _match_time_idx(time_units, time_values, init_date):
    """Find time index closest to init_date."""
    if init_date is None:
        return 0

    m = __import__('re').search(r'since\s+([\d\-]+\s*[\d:]*)', time_units)
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


def _parse_date(date_str):
    """Parse date string to seconds since epoch."""
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
    return (dt - datetime(1970, 1, 1)).total_seconds()


def _find_dim_name(ds, candidates):
    """Find matching dimension name in dataset."""
    for name in candidates:
        if name in ds.dimensions:
            return name
    return None


def _find_var_name(ds, candidates):
    """Find matching variable name in dataset."""
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _get_time(file_path, time_var=None):
    """Read time array from file. Returns seconds since 1970."""
    import netCDF4 as nc4
    try:
        ds = nc4.Dataset(file_path, 'r')
        candidates = [time_var] if time_var else ['time', 'time_counter', 'ocean_time']
        t_name = None
        for c in candidates:
            if c and c in ds.variables:
                t_name = c
                break
        if t_name is None:
            ds.close()
            return None

        t = np.atleast_1d(ds.variables[t_name][:]).astype(float)
        units = getattr(ds.variables[t_name], 'units', '')
        ds.close()

        m = __import__('re').search(r'since\s+([\d\-]+\s*[\d:]*)', str(units))
        if m:
            ref_epoch = _parse_date(m.group(1))
            if 'days' in str(units).lower():
                t = t * 86400.0 + ref_epoch
            elif 'hours' in str(units).lower():
                t = t * 3600.0 + ref_epoch
            else:
                t = t + ref_epoch
        return t
    except Exception:
        return None


def _match_idx(source_times, target_date):
    """Find closest time index to target_date."""
    target = _parse_date(target_date) if target_date else None
    if target is None:
        return 0
    return int(np.argmin(np.abs(source_times - target)))
