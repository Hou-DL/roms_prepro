"""
Core IC routines — ported from reference d_cmems2roms_py.py.

Provides:
  - stretching / set_depth (vertical coordinate)
  - NaN filling: fill_nan_2d, fill_nans_vertically, _fill_nan_single
  - Interpolation: interp_to_roms, mercator2roms_2d, mercator2roms_3d
  - CMEMS enhanced reader: _read_cmems_var, _detect_cmems_files
  - Time helpers: _parse_time_units, _compute_ocean_time
  - NetCDF writer: _write_ic_netcdf
  - Legacy helpers for roms2roms: horizontal_interp, sigma_to_z, z_to_sigma,
    rotate_uv, uv_to_cgrid, compute_ubar_vbar

Axis convention: IC data uses (lat, lon) or (lat, lon, depth).
FILL_VAL = 1e37 for land masking.
"""

import os
import re
import glob as _glob
import sys
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

FILL_VAL = 1.0e+37

# =====================================================================
# Constants — dimension/variable aliases for auto-detection
# =====================================================================

_TIME_ALIASES  = ['time', 'time_counter', 't', 'ocean_time', 'juld']
_DEPTH_ALIASES = ['depth', 'deptht', 'depthu', 'depthv', 'lev', 'z', 'level', 'sigma']
_LAT_ALIASES   = ['latitude', 'lat', 'nav_lat', 'y', 'Latitude']
_LON_ALIASES   = ['longitude', 'lon', 'nav_lon', 'x', 'Longitude']

_VAR_ALIASES = {
    'zos':    ['zos', 'sla', 'adt', 'ssh', 'surf_el', 'zeta'],
    'thetao': ['thetao', 'temperature', 'temp', 't', 'votemper', 'TEMP'],
    'so':     ['so', 'salinity', 'salt', 's', 'vosaline', 'SALT'],
    'uo':     ['uo', 'u', 'water_u', 'vozocrtx', 'u_eastward', 'eastward_sea_water_velocity'],
    'vo':     ['vo', 'v', 'water_v', 'vomecrty', 'v_northward', 'northward_sea_water_velocity'],
}


# =====================================================================
# Vertical coordinate functions
# stretching is delegated to grid.vgrid (numerically verified against
# pyroms / ROMS set_scoord for Vstretching 1-5). The previous local copy
# used the Song & Haidvogel (1994) curve for every Vstretching value,
# which misplaced the vertical levels for Vstretching >= 2.
# =====================================================================

try:
    from ..grid.vgrid import stretching  # noqa: F401  (re-exported)
except ImportError:  # direct-script mode
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from grid.vgrid import stretching  # noqa: F401


def set_depth(Vtransform, Vstretching, theta_s, theta_b, hc, N, igrid, h, zeta):
    """
    Compute ROMS S-coordinate depths.
    igrid: 1=RHO, 3=U, 4=V, 5=W
    Returns: (nlat, nlon, N) or (nlat, nlon, N+1)
    """
    nlat, nlon = h.shape

    if igrid == 1:
        s, C = stretching(Vstretching, theta_s, theta_b, N, 0)
        h2d = h; zeta2d = zeta
    elif igrid == 3:
        s, C = stretching(Vstretching, theta_s, theta_b, N, 0)
        h2d = 0.5 * (h[:, :-1] + h[:, 1:])
        zeta2d = 0.5 * (zeta[:, :-1] + zeta[:, 1:])
    elif igrid == 4:
        s, C = stretching(Vstretching, theta_s, theta_b, N, 0)
        h2d = 0.5 * (h[:-1, :] + h[1:, :])
        zeta2d = 0.5 * (zeta[:-1, :] + zeta[1:, :])
    elif igrid == 5:
        s, C = stretching(Vstretching, theta_s, theta_b, N, 1)
        h2d = h; zeta2d = zeta
    else:
        raise ValueError(f"Unsupported igrid={igrid}")

    if Vtransform == 2:
        S = s[:, np.newaxis, np.newaxis]
        C2 = C[:, np.newaxis, np.newaxis]
        H = h2d[np.newaxis, :, :]
        Z = zeta2d[np.newaxis, :, :]
        z = Z + (Z + H) * (hc * S + C2 * H) / (hc + H)
        z = z.transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported Vtransform={Vtransform}")

    return z


# =====================================================================
# NaN filling (from d_cmems2roms_py.py — EXACT)
# =====================================================================

def _fill_nan_single(layer):
    """2D nearest-neighbor NaN fill via distance_transform."""
    nan_mask = np.isnan(layer)
    if not np.any(nan_mask) or np.all(nan_mask):
        return layer
    _, indices = distance_transform_edt(nan_mask, return_indices=True)
    layer[nan_mask] = layer[indices[0][nan_mask], indices[1][nan_mask]]
    return layer


def fill_nan_2d(data):
    """Fill NaN in 2D or 3D (lat,lon[,depth]) arrays. Copies."""
    result = np.copy(data)
    if result.ndim == 2:
        return _fill_nan_single(result)
    for k in range(result.shape[2]):
        result[:, :, k] = _fill_nan_single(result[:, :, k].copy())
    return result


def fill_nans_vertically(data):
    """Per-column nearest-neighbor fill of NaN/0 (shallow-water artifice)."""
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


# =====================================================================
# Interpolation (from d_cmems2roms_py.py — EXACT)
# =====================================================================

def interp_to_roms(data, lon_1d, lat_1d, lon_rho, lat_rho):
    """Linear interpolation: CMEMS regular grid -> ROMS grid."""
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
    """2D interpolation (zeta) with de-mean, fill, re-mean."""
    zeta_mean = np.nanmean(Zeta)
    Zeta_dm = Zeta - zeta_mean
    Fout = interp_to_roms(Zeta_dm, lon_1d, lat_1d, lon_rho, lat_rho)
    Fout = _fill_nan_single(Fout)
    Fout = Fout + zeta_mean
    return Fout


def mercator2roms_3d(Finp, lon_1d, lat_1d, depth_in, lon_rho, lat_rho, z_r):
    """
    3D interpolation: horizontal per layer, then vertical per point.

    Finp: (nlat_src, nlon_src, ndepth_src)
    depth_in: (ndepth_src,) negative values (downward negative)
    z_r: (nlat_roms, nlon_roms, N) ROMS sigma depths
    """
    nlat_r, nlon_r, N = z_r.shape
    ndepth_src = len(depth_in)

    # Horizontal interpolation per layer
    Flev = np.full((nlat_r, nlon_r, ndepth_src), np.nan)
    for k in range(ndepth_src):
        Flev[:, :, k] = interp_to_roms(Finp[:, :, k], lon_1d, lat_1d, lon_rho, lat_rho)

    # Vertical interpolation per point
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
            # Ensure src_z is monotonically increasing for np.interp
            sort_idx = np.argsort(src_z)
            src_z = src_z[sort_idx]
            src_v = src_v[sort_idx]
            target_z = np.clip(target_z, src_z[0], src_z[-1])
            Fout[i, j, :] = np.interp(target_z, src_z, src_v)

    return Fout


# =====================================================================
# Velocity processing (from d_cmems2roms_py.py — EXACT)
# =====================================================================

def rotate_uv_cgrid(Urho, Vrho, angle):
    """
    Rotate Urho/Vrho to ROMS C-grid and compute barotropic.

    angle: (nlat, nlon) — returns dicts with u, v, ubar, vbar already
    staggered to U/V masks.

    NOTE: For IC, the caller must provide pre-staggered Hz_u, Hz_v.
    This returns the rotated 3D velocities; use compute_ubar_vbar() for barotropic.
    """
    angle_3d = angle[:, :, np.newaxis]
    Urot = Urho * np.cos(angle_3d) + Vrho * np.sin(angle_3d)
    Vrot = Vrho * np.cos(angle_3d) - Urho * np.sin(angle_3d)
    u = 0.5 * (Urot[:, :-1, :] + Urot[:, 1:, :])
    v = 0.5 * (Vrot[:-1, :, :] + Vrot[1:, :, :])
    return u, v


def compute_ubar_vbar(u, v, Hz):
    """
    Compute barotropic velocity from 3D u,v and layer thicknesses.

    u: (nlat, nlon-1, N), v: (nlat-1, nlon, N)
    Hz: (nlat, nlon, N) — RHO-grid layer thicknesses.
    """
    Hz_u = 0.5 * (Hz[:, :-1, :] + Hz[:, 1:, :])
    Hz_v = 0.5 * (Hz[:-1, :, :] + Hz[1:, :, :])

    sum_Hz_u = np.sum(Hz_u, axis=2)
    sum_Hz_v = np.sum(Hz_v, axis=2)
    sum_Hz_u[sum_Hz_u == 0] = 1
    sum_Hz_v[sum_Hz_v == 0] = 1

    ubar = np.sum(u * Hz_u, axis=2) / sum_Hz_u
    vbar = np.sum(v * Hz_v, axis=2) / sum_Hz_v
    return ubar, vbar


# =====================================================================
# NetCDF writer (adapted from d_cmems2roms_py.py:write_roms_ini — EXACT)
# =====================================================================

def _write_ic_netcdf(fname, h, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
                     ocean_time, theta_s, theta_b, Tcline, hc,
                     N, zeta, ubar, vbar, u, v, temp, salt,
                     Vtransform, Vstretching, time_ref='1990-01-01'):
    """Write ROMS IC NetCDF (NETCDF3_CLASSIC format, MATLAB-compatible)."""
    import netCDF4 as nc4

    nlat, nlon = h.shape
    if os.path.exists(fname):
        os.remove(fname)

    ds = nc4.Dataset(fname, 'w', format='NETCDF3_CLASSIC')
    ds.type = 'INITIALIZATION file'

    # Dimensions — ocean_time unlimited, tracer fixed at 2
    ds.createDimension('xi_rho', nlon)
    ds.createDimension('xi_u', nlon - 1)
    ds.createDimension('xi_v', nlon)
    ds.createDimension('eta_rho', nlat)
    ds.createDimension('eta_u', nlat)
    ds.createDimension('eta_v', nlat - 1)
    ds.createDimension('s_rho', N)
    ds.createDimension('s_w', N + 1)
    ds.createDimension('tracer', 2)
    ds.createDimension('ocean_time', None)  # unlimited

    # Scalar variables
    ncvar = ds.createVariable('spherical', 'i4', ())
    ncvar.long_name = 'grid type logical switch'
    ncvar[:] = 1

    for key, val in [('Vtransform', int(Vtransform)), ('Vstretching', int(Vstretching))]:
        ncvar = ds.createVariable(key, 'i4', ())
        ncvar[:] = val

    for key, val in [('theta_s', float(theta_s)), ('theta_b', float(theta_b)),
                     ('Tcline', float(Tcline)), ('hc', float(hc))]:
        ncvar = ds.createVariable(key, 'f8', ())
        if key == 'theta_s':
            ncvar.long_name = 'S-coordinate surface control parameter'
        elif key == 'theta_b':
            ncvar.long_name = 'S-coordinate bottom control parameter'
        elif key == 'Tcline':
            ncvar.long_name = 'S-coordinate surface/bottom layer width'
            ncvar.units = 'meter'
        elif key == 'hc':
            ncvar.long_name = 'S-coordinate parameter, critical depth'
            ncvar.units = 'meter'
        ncvar[:] = val

    # Vertical coordinate variables
    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, 0)
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, 1)

    s_data = {'s_rho': s_rho, 's_w': s_w}
    for key, lon in [
        ('s_rho', 'S-coordinate at RHO-points'),
        ('s_w', 'S-coordinate at W-points'),
    ]:
        ncvar = ds.createVariable(key, 'f8', (key,))
        ncvar.long_name = lon
        ncvar.valid_min = -1.0
        ncvar.valid_max = 0.0
        ncvar.positive = 'up'
        ncvar.standard_name = 'ocean_s_coordinate_g2'
        ncvar[:] = s_data[key]

    for key, dim, lon in [
        ('Cs_r', 's_rho', 'S-coordinate stretching curves at RHO-points'),
        ('Cs_w', 's_w', 'S-coordinate stretching curves at W-points'),
    ]:
        ncvar = ds.createVariable(key, 'f8', (dim,))
        ncvar.long_name = lon
        ncvar.valid_min = -1.0
        ncvar.valid_max = 0.0
        ncvar[:] = {'Cs_r': Cs_r, 'Cs_w': Cs_w}[key]

    # Grid coordinates
    for vname, data, dims, long_name, units in [
        ('h', h, ('eta_rho', 'xi_rho'), 'bathymetry at RHO-points', 'meter'),
        ('lon_rho', lon_rho, ('eta_rho', 'xi_rho'), 'longitude of RHO-points', 'degree_east'),
        ('lat_rho', lat_rho, ('eta_rho', 'xi_rho'), 'latitude of RHO-points', 'degree_north'),
        ('lon_u', lon_u, ('eta_u', 'xi_u'), 'longitude of U-points', 'degree_east'),
        ('lat_u', lat_u, ('eta_u', 'xi_u'), 'latitude of U-points', 'degree_north'),
        ('lon_v', lon_v, ('eta_v', 'xi_v'), 'longitude of V-points', 'degree_east'),
        ('lat_v', lat_v, ('eta_v', 'xi_v'), 'latitude of V-points', 'degree_north'),
    ]:
        ncvar = ds.createVariable(vname, 'f8', dims)
        ncvar.long_name = long_name
        ncvar.units = units
        ncvar[:] = data

    # ocean_time
    ncvar = ds.createVariable('ocean_time', 'f8', ('ocean_time',))
    ncvar.long_name = 'time since initialization'
    ncvar.units = f'seconds since {time_ref} 00:00:00'
    ncvar.calendar = 'gregorian'
    ncvar[:] = ocean_time

    # Data variables — dimension order matches reference:
    # (ocean_time, s_rho, eta_*, xi_*) for 3D, (ocean_time, eta_*, xi_*) for 2D
    def _def_var(name, array_data, dims, long_name, units, spval=FILL_VAL):
        ncvar = ds.createVariable(name, 'f8', dims, fill_value=spval)
        ncvar.long_name = long_name
        ncvar.units = units
        ncvar.time = 'ocean_time'
        coords = ' '.join(d for d in dims if d != 'ocean_time' and d != 's_rho')
        if name in ('u', 'ubar'):
            coords = coords.replace('eta_u xi_u', 'lon_u lat_u').replace('eta_rho xi_rho', 'lon_u lat_u')
        elif name in ('v', 'vbar'):
            coords = coords.replace('eta_v xi_v', 'lon_v lat_v').replace('eta_rho xi_rho', 'lon_v lat_v')
        else:
            coords = coords.replace('eta_rho xi_rho', 'lon_rho lat_rho')
        if 's_rho' in dims:
            coords = coords + ' s_rho'
        coords = coords + ' ocean_time'
        ncvar.coordinates = coords
        if array_data.ndim == 3:
            array_data = array_data.transpose(2, 0, 1)  # (eta,xi,s) -> (s,eta,xi)
        array_data = array_data[np.newaxis, ...].astype('f8')
        ncvar[:] = array_data

    _def_var('zeta', zeta, ('ocean_time', 'eta_rho', 'xi_rho'),
             'free-surface', 'meter')
    _def_var('ubar', ubar, ('ocean_time', 'eta_u', 'xi_u'),
             'vertically integrated u-momentum component', 'meter second-1')
    _def_var('vbar', vbar, ('ocean_time', 'eta_v', 'xi_v'),
             'vertically integrated v-momentum component', 'meter second-1')
    _def_var('u', u, ('ocean_time', 's_rho', 'eta_u', 'xi_u'),
             'u-momentum component', 'meter second-1')
    _def_var('v', v, ('ocean_time', 's_rho', 'eta_v', 'xi_v'),
             'v-momentum component', 'meter second-1')
    _def_var('temp', temp, ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
             'potential temperature', 'Celsius')
    _def_var('salt', salt, ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
             'salinity', 'PSU')

    ds.close()


# =====================================================================
# CMEMS file reading & auto-detection (from existing _core.py)
# =====================================================================

def _find_in(ds, candidates, attr='dimensions'):
    """Return first matching key from candidates in ds dimensions/variables."""
    if attr == 'dimensions':
        for name in candidates:
            if name in ds.dimensions:
                return name
    else:
        for name in candidates:
            if name in ds.variables:
                return name
    return None


def _find_dim(ds, candidates):
    return _find_in(ds, candidates, 'dimensions')


def _find_var(ds, candidates):
    return _find_in(ds, candidates, 'variables')


def _resolve_variable(ds, var_name):
    """Find variable by canonical name or alias."""
    if var_name in ds.variables:
        return ds.variables[var_name], var_name

    for canon, aliases in _VAR_ALIASES.items():
        if var_name == canon or var_name in aliases:
            for alias in aliases:
                if alias in ds.variables:
                    return ds.variables[alias], alias
            break

    for vname, vobj in ds.variables.items():
        if vname in ds.dimensions:
            continue
        return vobj, vname

    raise KeyError(f"Variable '{var_name}' not found in {list(ds.variables.keys())}")


def _classify_dims(variable_dims):
    """Classify dims of a variable. Returns (time_pos, depth_pos, lat_pos, lon_pos)."""
    ndim = len(variable_dims)
    time_pos = depth_pos = lat_pos = lon_pos = None

    for i, d in enumerate(variable_dims):
        d_lower = d.lower()
        if any(t in d_lower for t in ['time', 'juld']):
            time_pos = i
        elif any(t in d_lower for t in ['depth', 'lev', 'level', 'sigma', 'z']):
            depth_pos = i
        elif any(t in d_lower for t in ['lat', 'y']):
            lat_pos = i
        elif any(t in d_lower for t in ['lon', 'x']):
            lon_pos = i

    assigned = sum(p is not None for p in (time_pos, depth_pos, lat_pos, lon_pos))
    if assigned < ndim:
        unassigned = [i for i in range(ndim)
                      if i not in (time_pos, depth_pos, lat_pos, lon_pos)]
        if lat_pos is None and len(unassigned) >= 2:
            lat_pos = unassigned[-2]
            lon_pos = unassigned[-1]
        elif lon_pos is None and len(unassigned) >= 1:
            lon_pos = unassigned[-1]

    return time_pos, depth_pos, lat_pos, lon_pos


def _read_1d_coord(ds, dim_name):
    """Read 1D coordinate from dataset, handling 2D->1D fallback."""
    if dim_name is None or dim_name not in ds.variables:
        return None
    arr = np.asarray(ds.variables[dim_name][:], dtype=float)
    if arr.ndim == 2:
        dim_lower = dim_name.lower()
        if any(t in dim_lower for t in ['lon', 'x']):
            arr = arr[0, :] if arr.shape[0] > 0 else arr[:, 0]
        else:
            arr = arr[:, 0] if arr.shape[1] > 0 else arr[0, :]
    return np.asarray(arr, dtype=float)


def _read_time(ds, time_pos):
    """Read time variable and units from dataset."""
    if time_pos is None:
        return None, None
    tvar = _find_var(ds, _TIME_ALIASES)
    if tvar is None or tvar not in ds.variables:
        return None, None
    tv = ds.variables[tvar]
    units = getattr(tv, 'units', None)
    values = np.asarray(tv[:], dtype=float)
    return units, values


def _read_cmems_var(file_path, var_name, time_index=0):
    """
    Read a single variable from CMEMS NetCDF with maximum compatibility.

    Returns
    -------
    data : ndarray, (nlat, nlon) or (nlat, nlon, ndepth)
    lon_1d : 1D longitude
    lat_1d : 1D latitude
    depth : 1D positive depths (None for 2D)
    (time_units, time_values) : tuple
    """
    import netCDF4 as nc4
    ds = nc4.Dataset(file_path, 'r')

    var_obj, actual_name = _resolve_variable(ds, var_name)
    time_pos, depth_pos, lat_pos, lon_pos = _classify_dims(var_obj.dimensions)

    lon_dim = _find_dim(ds, _LON_ALIASES)
    lat_dim = _find_dim(ds, _LAT_ALIASES)
    depth_dim = _find_dim(ds, _DEPTH_ALIASES)

    lon_1d = _read_1d_coord(ds, lon_dim)
    lat_1d = _read_1d_coord(ds, lat_dim)
    depth = _read_1d_coord(ds, depth_dim)

    # Disable netCDF4 auto-scale to handle fill/scale/offset manually
    var_obj.set_auto_scale(False)
    raw = np.array(var_obj[:], dtype=float)
    var_obj.set_auto_scale(True)

    if time_pos is not None and raw.shape[time_pos] > 1:
        time_index = min(time_index, raw.shape[time_pos] - 1)

    if time_pos is not None and raw.ndim >= 3:
        raw = np.take(raw, time_index, axis=time_pos)

    fv = getattr(var_obj, '_FillValue', None)
    mv = getattr(var_obj, 'missing_value', None)
    sf = getattr(var_obj, 'scale_factor', None)
    ao = getattr(var_obj, 'add_offset', None)
    if fv is not None:
        raw[raw == float(fv)] = np.nan
    if mv is not None:
        raw[raw == float(mv)] = np.nan
    if sf is not None:
        raw = raw * float(sf)
    if ao is not None:
        raw = raw + float(ao)

    ndim = raw.ndim
    remaining_dims = list(var_obj.dimensions)
    if time_pos is not None:
        remaining_dims.pop(time_pos)

    depth_pos2 = lat_pos2 = lon_pos2 = None
    for idx, dname in enumerate(remaining_dims):
        dl = dname.lower()
        if any(t in dl for t in ['depth', 'lev', 'level', 'sigma']):
            depth_pos2 = idx
        elif any(t in dl for t in ['lat', 'y']):
            lat_pos2 = idx
        elif any(t in dl for t in ['lon', 'x']):
            lon_pos2 = idx

    if ndim == 3 and depth_pos2 is None and lat_pos2 is not None and lon_pos2 is not None:
        remaining = [i for i in range(ndim) if i not in (lat_pos2, lon_pos2)]
        if remaining:
            depth_pos2 = remaining[0]
    if ndim == 3 and lat_pos2 is None and depth_pos2 is not None and lon_pos2 is not None:
        lat_pos2 = 0

    if lat_pos2 is None:
        lat_pos2 = -2 if ndim >= 2 else 0
    if lon_pos2 is None:
        lon_pos2 = -1 if ndim >= 2 else (0 if ndim == 1 else 1)

    if ndim == 3:
        target_order = [lat_pos2, lon_pos2]
        if depth_pos2 is not None:
            target_order.append(depth_pos2)
        else:
            remaining = [i for i in range(ndim) if i not in (lat_pos2, lon_pos2)]
            target_order.extend(remaining)
        data = np.transpose(raw, target_order)
    elif ndim == 2:
        data = np.transpose(raw, [lat_pos2, lon_pos2])
    else:
        data = raw

    time_units, time_values = _read_time(ds, time_pos)
    ds.close()

    if depth is not None and len(depth) > 0:
        if depth[0] > depth[-1]:
            depth = depth[::-1]
            if data.ndim == 3 and depth_dim:
                data = data[:, :, ::-1]

    return data, lon_1d, lat_1d, depth, (time_units, time_values)


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


def _read_vgrid_params(grid_file):
    """Read S-coordinate parameters from ROMS grid file global attributes.

    Grid files store these as ``var NAME = VALUE`` global attributes.
    Returns a dict with keys: Vtransform, Vstretching, theta_s, theta_b,
    Tcline, hc, N.  Missing keys are absent from the dict.
    """
    import netCDF4 as nc4
    params = {}
    with nc4.Dataset(grid_file, 'r') as g:
        # 1) Read from scalar variables
        for key in ('Vtransform', 'Vstretching', 'theta_s', 'theta_b', 'Tcline', 'hc'):
            if key in g.variables:
                val = g.variables[key][()]
                val = val.item() if hasattr(val, 'item') else val
                if key in ('Vtransform', 'Vstretching'):
                    params[key] = int(val)
                else:
                    params[key] = float(val)
        # 2) Fall back to global 'var NAME = VALUE' attributes (e.g. NSCS grid)
        if not any(k in params for k in ('Vtransform', 'Vstretching', 'theta_s', 'theta_b', 'Tcline')):
            for attr in g.ncattrs():
                val = g.getncattr(attr)
                name = attr.replace('var ', '').strip()
                if name in ('Vtransform', 'Vstretching') and isinstance(val, (int, float)):
                    params[name] = int(val)
                elif name in ('theta_s', 'theta_b', 'Tcline', 'hc') and isinstance(val, (int, float)):
                    params[name] = float(val)
        # 3) N from s_rho dimension
        if 's_rho' in g.dimensions:
            params['N'] = len(g.dimensions['s_rho'])
    return params


def _resolve_vgrid_params(grid_file, Vtransform, Vstretching, theta_s, theta_b, Tcline, N):
    """Fill in any None ROMS grid parameter from the grid file.

    Parameters that are already set (non-None) are kept as-is.
    Parameters that are None are read from the grid file if possible.
    Remaining None values get ROMS-standard defaults (Vtransform=2,
    Vstretching=2, theta_s=2.5, theta_b=1.0, Tcline=25.0, hc=Tcline, N=30).
    """
    file_params = _read_vgrid_params(grid_file)
    # Final defaults for anything still missing
    defaults = {
        'Vtransform': 2,
        'Vstretching': 2,
        'theta_s': 2.5,
        'theta_b': 1.0,
        'Tcline': 25.0,
        'N': 30,
    }
    resolved = {}
    for key in ('Vtransform', 'Vstretching', 'theta_s', 'theta_b', 'Tcline', 'N'):
        user_val = locals()[key] if key in locals() else None
        if user_val is not None:
            resolved[key] = user_val
        elif key in file_params:
            resolved[key] = file_params[key]
        else:
            resolved[key] = defaults[key]
    resolved['hc'] = resolved['Tcline']
    return resolved


def _detect_cmems_files(data_dir):
    """Auto-detect CMEMS files in directory. Returns {var: path}."""
    result = {}
    patterns = {
        'zos':    ['cmems_zos_*.nc', '*zos*.nc', '*ssh*.nc', '*.zos.*.nc'],
        'thetao': ['cmems_thetao_*.nc', '*thetao*.nc', '*temp*.nc', '*temperature*.nc'],
        'so':     ['cmems_so_*.nc', '*so_*.nc', '*salt*.nc', '*salinity*.nc'],
        'uo':     ['cmems_uo_*.nc', '*uo*.nc'],
        'vo':     ['cmems_vo_*.nc', '*vo*.nc'],
    }
    for std_name, pats in patterns.items():
        for pat in pats:
            matches = sorted(_glob.glob(os.path.join(data_dir, pat)))
            if matches:
                result[std_name] = matches[0]
                break
    return result


def _detect_time_units(temp_file):
    """Read time units and values from a CMEMS file."""
    import netCDF4 as nc4
    try:
        ds = nc4.Dataset(temp_file, 'r')
        tvar = _find_var(ds, _TIME_ALIASES)
        if tvar and tvar in ds.variables:
            units = getattr(ds.variables[tvar], 'units', '')
            values = ds.variables[tvar][:]
            ds.close()
            return units, values
        ds.close()
    except Exception:
        pass
    return None, None


def _parse_time_units(units_str):
    """Parse 'hours since 1950-01-01 00:00:00' -> (multiplier, epoch_datetime)."""
    if not units_str:
        return None, None
    m = re.search(r'(hours|days|seconds)\s+since\s+([\d\-]+\s*[\d:]*)', units_str, re.I)
    if not m:
        return None, None

    unit_str = m.group(1).lower()
    ref_str = m.group(2).strip()

    if unit_str.startswith('hour'):
        multiplier = 3600.0
    elif unit_str.startswith('day'):
        multiplier = 86400.0
    else:
        multiplier = 1.0

    try:
        epoch = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            epoch = datetime.strptime(ref_str, '%Y-%m-%d')
        except ValueError:
            epoch = datetime.strptime(ref_str, '%Y-%m-%dT%H:%M:%S')

    return multiplier, epoch


def _compute_ocean_time(time_info, init_date, roms_time_ref, time_index=None):
    """
    Compute ocean_time (seconds) from CMEMS time metadata.

    Returns (ocean_time_seconds, time_index_int).
    ``time_index`` is used when ``init_date`` is None so that an explicitly
    selected record is also timestamped with that record's time.
    """
    time_units, time_values = time_info
    if time_units is None or time_values is None:
        return 0.0, 0

    multiplier, epoch = _parse_time_units(time_units)
    if multiplier is None:
        return 0.0, 0

    if init_date is not None:
        init_dt = datetime.strptime(init_date, '%Y-%m-%d')
        time_dates = [epoch + timedelta(seconds=float(v) * multiplier) for v in time_values]
        idx = min(range(len(time_dates)),
                  key=lambda i: abs((time_dates[i] - init_dt).total_seconds()))
    elif time_index is not None:
        idx = int(min(time_index, len(time_values) - 1))
    else:
        idx = 0

    current_dt = epoch + timedelta(seconds=float(time_values[idx]) * multiplier)
    roms_ref_dt = datetime.strptime(roms_time_ref.split()[0], '%Y-%m-%d')
    ocean_time = (current_dt - roms_ref_dt).total_seconds()

    return ocean_time, idx


def _parse_date(date_str):
    """Parse date string to seconds since 1970-01-01."""
    if date_str is None:
        return None
    if isinstance(date_str, (int, float)):
        return float(date_str)
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.strptime(date_str, fmt)
            return (dt - datetime(1970, 1, 1)).total_seconds()
        except ValueError:
            continue
    return None


def _get_time(file_path, time_var=None):
    """Read time array from file. Returns seconds since 1970 or None."""
    import netCDF4 as nc4
    try:
        ds = nc4.Dataset(file_path, 'r')
        candidates = [time_var] if time_var else _TIME_ALIASES
        t_name = _find_var(ds, candidates)
        if t_name is None:
            ds.close()
            return None
        t = np.atleast_1d(ds.variables[t_name][:]).astype(float)
        units = getattr(ds.variables[t_name], 'units', '')
        ds.close()
        multiplier, epoch = _parse_time_units(units)
        if multiplier and epoch:
            ref_epoch = (epoch - datetime(1970, 1, 1)).total_seconds()
            t = t * multiplier + ref_epoch
        return t
    except Exception:
        return None


def _match_idx(source_times, target_date):
    """Find closest time index to target_date."""
    target = _parse_date(target_date) if target_date else None
    if target is None:
        return 0
    return int(np.argmin(np.abs(source_times - target)))


# =====================================================================
# Legacy interpolation helpers for ROMS-to-ROMS nesting (from _core_old.py)
# =====================================================================

def horizontal_interp(src_lon, src_lat, src_var, dst_lon, dst_lat,
                      mask=None, fill_value=np.nan):
    """Interpolate from source grid to destination grid (single-file mode)."""
    src_var = np.asarray(src_var, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)
    src_lat = np.asarray(src_lat, dtype=float)

    is_regular = (src_lon.ndim == 1 and src_lat.ndim == 1)
    if not is_regular:
        is_regular = (src_lon.shape[0] > 1 and src_lat.shape[0] > 1 and
                      np.allclose(src_lon[1:, :] - src_lon[:-1, :], 0) and
                      np.allclose(src_lat[:, 1:] - src_lat[:, :-1], 0))

    if is_regular:
        if src_lon.ndim == 1:
            x = np.asarray(src_lon, dtype=float)
            y = np.asarray(src_lat, dtype=float)
        else:
            x = np.asarray(src_lon[0, :], dtype=float)
            y = np.asarray(src_lat[:, 0], dtype=float)

        z_src = np.asarray(src_var, dtype=float)
        if x[0] > x[-1]:
            x = x[::-1]
            if z_src.ndim == 3:
                z_src = z_src[:, :, ::-1]
            elif z_src.ndim == 2:
                z_src = z_src[:, ::-1]
        if y[0] > y[-1]:
            y = y[::-1]
            if z_src.ndim == 3:
                z_src = z_src[:, ::-1, :]
            elif z_src.ndim == 2:
                z_src = z_src[::-1, :]

        def _fast_interp(var_2d):
            interp = RegularGridInterpolator(
                (y, x), var_2d, bounds_error=False, fill_value=fill_value)
            pts = np.column_stack((dst_lat.ravel(), dst_lon.ravel()))
            return interp(pts).reshape(dst_lon.shape)

        if src_var.ndim == 2:
            return _fast_interp(src_var)
        nlev = src_var.shape[0]
        result = np.zeros((nlev, dst_lon.shape[0], dst_lon.shape[1]))
        for k in range(nlev):
            result[k] = _fast_interp(z_src[k])
        if mask is not None:
            result[:, mask < 0.5] = fill_value
        return result

    if src_var.ndim == 2:
        return _interp_2d(src_lon, src_lat, src_var, dst_lon, dst_lat, mask)
    nlev = src_var.shape[0]
    result = np.zeros((nlev, dst_lon.shape[0], dst_lon.shape[1]))
    for k in range(nlev):
        result[k] = _interp_2d(src_lon, src_lat, src_var[k], dst_lon, dst_lat)
    if mask is not None:
        result[:, mask < 0.5] = fill_value
    return result


def _interp_2d(src_lon, src_lat, src_var, dst_lon, dst_lat, mask=None):
    """Interpolate a 2D field from scattered source to destination grid."""
    src_lon = np.asarray(src_lon, dtype=float).ravel()
    src_lat = np.asarray(src_lat, dtype=float).ravel()
    src_var = np.asarray(src_var, dtype=float).ravel()
    dst_lon = np.asarray(dst_lon, dtype=float)
    dst_lat = np.asarray(dst_lat, dtype=float)

    interp = LinearNDInterpolator(
        np.column_stack((src_lon, src_lat)), src_var)
    result = interp(dst_lon, dst_lat)

    bad = np.isnan(result)
    if bad.any() and (~bad).any():
        ok_pts = np.column_stack((dst_lon[~bad], dst_lat[~bad]))
        ok_vals = result[~bad]
        bad_pts = np.column_stack((dst_lon[bad], dst_lat[bad]))
        tree = cKDTree(ok_pts)
        _, idx = tree.query(bad_pts)
        result[bad] = ok_vals[idx]

    if mask is not None:
        result[mask < 0.5] = np.nan
    return result


def z_to_sigma(var_z, z_levels, sigma_depth, fill_value=np.nan):
    """Interpolate from z-levels to ROMS sigma coordinates."""
    var_z = np.asarray(var_z, dtype=float)
    z_lev = np.asarray(z_levels, dtype=float).ravel()
    s_dep = np.asarray(sigma_depth, dtype=float)
    ns, eta, xi = s_dep.shape
    nz = len(z_lev)

    if z_lev[0] > z_lev[-1]:
        z_lev = z_lev[::-1]
        var_z = var_z[::-1]

    var_flat = var_z.reshape(nz, -1)
    s_flat = s_dep.reshape(ns, -1)
    result = np.full_like(s_flat, fill_value)

    for c in range(var_flat.shape[1]):
        col = var_flat[:, c]
        valid = np.isfinite(col)
        if valid.sum() < 2:
            continue
        z_v, v_v = z_lev[valid], col[valid]
        result[:, c] = np.interp(s_flat[:, c], z_v, v_v,
                                  left=v_v[0], right=v_v[-1])

    return result.reshape(ns, eta, xi)


def sigma_to_z(var_sigma, sigma_depth, z_levels, fill_value=np.nan):
    """Interpolate from ROMS sigma to z-levels."""
    ns, eta, xi = var_sigma.shape
    nz = len(z_levels)
    result = np.full((nz, eta, xi), fill_value, dtype=float)

    for j in range(eta):
        for i in range(xi):
            s_dep = sigma_depth[:, j, i]
            v_col = var_sigma[:, j, i]
            valid = np.isfinite(v_col)
            if valid.sum() < 2:
                continue
            z_s = s_dep[valid]
            v_s = v_col[valid]
            if z_s[0] > z_s[-1]:
                z_s = z_s[::-1]
                v_s = v_s[::-1]
            result[:, j, i] = np.interp(z_levels, z_s, v_s,
                                         left=fill_value, right=fill_value)
    return result


def rotate_uv(u, v, src_angle, dst_angle):
    """Rotate u,v between grids."""
    angle = np.asarray(dst_angle) - np.asarray(src_angle)
    ndim_u = u.ndim - 2
    angle = angle.reshape((1,) * ndim_u + angle.shape)
    c = np.cos(angle)
    s = np.sin(angle)
    u_rot = u * c + v * s
    v_rot = v * c - u * s
    return u_rot, v_rot


def uv_to_cgrid(u_rho, v_rho, mask_u=None, mask_v=None, spval=1e37):
    """Average velocity to C-grid staggered positions."""
    u = 0.5 * (u_rho[..., :-1] + u_rho[..., 1:])
    v = 0.5 * (v_rho[..., :-1, :] + v_rho[..., 1:, :])

    if mask_u is not None:
        mask_u_bc = mask_u.reshape((1,) * (u.ndim - 2) + mask_u.shape)
        u = np.where(mask_u_bc > 0.5, u, spval)
    if mask_v is not None:
        mask_v_bc = mask_v.reshape((1,) * (v.ndim - 2) + mask_v.shape)
        v = np.where(mask_v_bc > 0.5, v, spval)
    return u, v


def compute_ubar_vbar_legacy(u, v, z_w, mask_u=None, mask_v=None):
    """Compute barotropic velocity (legacy, used by roms2roms)."""
    dz_w = np.abs(z_w[1:] - z_w[:-1])
    dz_u = 0.5 * (dz_w[:, :, :-1] + dz_w[:, :, 1:])
    dz_v = 0.5 * (dz_w[:, :-1, :] + dz_w[:, 1:, :])

    ubar = np.sum(u * dz_u, axis=0) / np.sum(dz_u, axis=0)
    vbar = np.sum(v * dz_v, axis=0) / np.sum(dz_v, axis=0)

    if mask_u is not None:
        ubar[mask_u < 0.5] = 0.0
    if mask_v is not None:
        vbar[mask_v < 0.5] = 0.0
    return ubar, vbar
