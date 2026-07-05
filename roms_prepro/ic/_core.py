"""
Core interpolation routines and IC file writer for ROMS initial conditions.

Provides:
- horizontal_interp: source lon/lat → target lon/lat (2D and 3D)
- z_to_sigma: standard z-levels → ROMS sigma coordinates
- sigma_to_z: ROMS sigma → standard z-levels
- rotate_uv: rotate u,v from source angle to target angle
- uv_to_cgrid: average rotated vectors to ROMS C-grid positions
- compute_ubar_vbar: compute barotropic velocity
- write_ic_file: write standard ROMS IC NetCDF
"""

import re
import numpy as np
from datetime import datetime
import netCDF4 as nc4
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Horizontal interpolation
# ---------------------------------------------------------------------------

def _interp_2d(src_lon, src_lat, src_var, dst_lon, dst_lat, mask=None):
    """Interpolate a 2-D field from source to destination grid."""
    src_lon = np.asarray(src_lon, dtype=float).ravel()
    src_lat = np.asarray(src_lat, dtype=float).ravel()
    src_var = np.asarray(src_var, dtype=float).ravel()
    dst_lon = np.asarray(dst_lon, dtype=float)
    dst_lat = np.asarray(dst_lat, dtype=float)

    interp = LinearNDInterpolator(
        np.column_stack((src_lon, src_lat)), src_var
    )
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


def horizontal_interp(src_lon, src_lat, src_var, dst_lon, dst_lat,
                      mask=None, fill_value=np.nan):
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
            result = _fast_interp(src_var)
        else:
            nlev = src_var.shape[0]
            result = np.zeros((nlev, dst_lon.shape[0], dst_lon.shape[1]))
            for k in range(nlev):
                result[k] = _fast_interp(z_src[k])
            if mask is not None:
                result[:, mask < 0.5] = fill_value
        return result

    if src_var.ndim == 2:
        return _interp_2d(src_lon, src_lat, src_var, dst_lon, dst_lat, mask)
    else:
        nlev = src_var.shape[0]
        result = np.zeros((nlev, dst_lon.shape[0], dst_lon.shape[1]))
        for k in range(nlev):
            result[k] = _interp_2d(src_lon, src_lat, src_var[k],
                                    dst_lon, dst_lat)
        if mask is not None:
            result[:, mask < 0.5] = fill_value
        return result


# ---------------------------------------------------------------------------
# Vertical interpolation
# ---------------------------------------------------------------------------

def z_to_sigma(var_z, z_levels, sigma_depth, fill_value=np.nan):
    """
    Interpolate from standard z-levels to ROMS sigma coordinates
    (vectorised over spatial dimensions).

    Parameters
    ----------
    var_z : ndarray (nz, eta, xi)
    z_levels : ndarray (nz,) — monotonic depth array (negative downward)
    sigma_depth : ndarray (ns, eta, xi)
    fill_value : float

    Returns
    -------
    var_sigma : ndarray (ns, eta, xi)
    """
    var_z = np.asarray(var_z, dtype=float)
    z_lev = np.asarray(z_levels, dtype=float).ravel()
    s_dep = np.asarray(sigma_depth, dtype=float)
    ns, eta, xi = s_dep.shape
    nz = len(z_lev)

    # Ensure z_lev is increasing (shallow → deep)
    if z_lev[0] > z_lev[-1]:
        z_lev = z_lev[::-1]
        var_z = var_z[::-1]

    # Flatten spatial dims: (nz, eta*xi)
    var_flat = var_z.reshape(nz, -1)
    s_flat = s_dep.reshape(ns, -1)
    result = np.full_like(s_flat, fill_value)

    cols = var_flat.shape[1]
    for c in range(cols):
        col = var_flat[:, c]
        valid = np.isfinite(col)
        if valid.sum() < 2:
            continue
        z_v, v_v = z_lev[valid], col[valid]
        result[:, c] = np.interp(s_flat[:, c], z_v, v_v,
                                  left=v_v[0], right=v_v[-1])

    return result.reshape(ns, eta, xi)


def sigma_to_z(var_sigma, sigma_depth, z_levels, fill_value=np.nan):
    """
    Interpolate from ROMS sigma coordinates to standard z-levels.

    Parameters
    ----------
    var_sigma : ndarray (ns, eta, xi)
        Data on sigma levels.
    sigma_depth : ndarray (ns, eta, xi)
        Sigma-level depths.
    z_levels : ndarray (nz,)
        Target z-levels (negative downward).
    fill_value : float

    Returns
    -------
    var_z : ndarray (nz, eta, xi)
    """
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


# ---------------------------------------------------------------------------
# Velocity rotation
# ---------------------------------------------------------------------------

def rotate_uv(u, v, src_angle, dst_angle):
    """
    Rotate u,v from source grid to target grid.

    Default src_angle=0 means source is eastward/northward (ERA5 convention).
    """
    if src_angle is None:
        src_angle = 0.0
    if dst_angle is None:
        dst_angle = 0.0
    angle = np.asarray(dst_angle) - np.asarray(src_angle)
    ndim_u = u.ndim - 2
    angle = angle.reshape((1,) * ndim_u + angle.shape)
    c = np.cos(angle)
    s = np.sin(angle)
    u_rot = u * c + v * s
    v_rot = v * c - u * s
    return u_rot, v_rot


def uv_to_cgrid(u_rho, v_rho, mask_u=None, mask_v=None, spval=1e37):
    """
    Average velocity from RHO-points to U/V staggered C-grid positions.

    Parameters
    ----------
    u_rho, v_rho : ndarray (..., eta_rho, xi_rho)
        Velocity components at RHO-points.
    mask_u, mask_v : ndarray or None
    spval : float
        Fill value for land cells.

    Returns
    -------
    u, v : ndarray
        (..., eta_u, xi_u) and (..., eta_v, xi_v)
    """
    shape_u = u_rho.shape[:-2] + (u_rho.shape[-2], u_rho.shape[-1] - 1)
    shape_v = v_rho.shape[:-2] + (v_rho.shape[-2] - 1, v_rho.shape[-1])

    u = 0.5 * (u_rho[..., :-1] + u_rho[..., 1:])
    v = 0.5 * (v_rho[..., :-1, :] + v_rho[..., 1:, :])

    if mask_u is not None:
        mask_u_bc = mask_u.reshape((1,) * (u.ndim - 2) + mask_u.shape)
        u = np.where(mask_u_bc > 0.5, u, spval)
    if mask_v is not None:
        mask_v_bc = mask_v.reshape((1,) * (v.ndim - 2) + mask_v.shape)
        v = np.where(mask_v_bc > 0.5, v, spval)

    return u, v


def compute_ubar_vbar(u, v, z_w, mask_u=None, mask_v=None):
    """
    Compute barotropic velocity by vertically integrating 3-D u/v.

    Parameters
    ----------
    u, v : ndarray (N, eta_u, xi_u) / (N, eta_v, xi_v)
    z_w : ndarray (N+1, eta_rho, xi_rho)
        W-level depths (from set_depth, igrid=5).
    mask_u, mask_v : ndarray

    Returns
    -------
    ubar, vbar : ndarray (eta_u, xi_u) / (eta_v, xi_v)
    """
    # Layer thickness at u and v points
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


# ---------------------------------------------------------------------------
# NC file writer
# ---------------------------------------------------------------------------

def _put_var(ds, name, data, dims, **attrs):
    """Create and write a variable to a NetCDF dataset."""
    fill = attrs.pop('_FillValue', None)
    missing = attrs.pop('missing_value', None)
    v = ds.createVariable(name, 'f8', dims, fill_value=fill if fill is not None else None)
    if missing is not None:
        setattr(v, 'missing_value', missing)
    for k, val in attrs.items():
        setattr(v, k, val)
    v[:] = data


def write_ic_file(filename, metrics, vgrid_params, ocean_time,
                  zeta, temp, salt, u_mom, v_mom, ubar, vbar,
                  time_ref='2025-01-01'):
    """
    Write a ROMS initial conditions NetCDF file matching the exact
    convention used by ROMS Matlab toolbox (c_initial.m).

    Parameters
    ----------
    time_ref : str
        Reference date for ocean_time units, e.g. '2025-01-01'.
        Output units will be 'seconds since {time_ref} 00:00:00'.
    """
    spval = 1e37
    ds = nc4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    ds.type = 'ROMS INITIALIZATION file'
    ds.title = 'ROMS initial conditions from roms_prepro'
    ds.data_source = 'CMEMS reanalysis'

    ny, nx = metrics['lon_rho'].shape
    N = temp.shape[0]
    mask = metrics.get('mask_rho', np.ones((ny, nx)))

    # --- dimensions ---
    for d in [('xi_rho', nx), ('xi_u', nx - 1), ('xi_v', nx),
              ('xi_psi', nx - 1)]:
        ds.createDimension(d[0], d[1])
    for d in [('eta_rho', ny), ('eta_u', ny), ('eta_v', ny - 1),
              ('eta_psi', ny - 1)]:
        ds.createDimension(d[0], d[1])
    ds.createDimension('s_rho', N)
    ds.createDimension('s_w', N + 1)
    ds.createDimension('tracer', 2)
    ds.createDimension('ocean_time', None)

    # --- scalar params ---
    v = ds.createVariable('spherical', 'i4', ())
    v.long_name = 'grid type logical switch'
    v.flag_values = np.array([0, 1], 'i4')
    v.flag_meanings = 'Cartesian spherical'
    v[:] = 1

    for key in ('Vtransform', 'Vstretching'):
        if key in vgrid_params:
            v = ds.createVariable(key, 'i4', ())
            v.long_name = f'vertical terrain-following {"transformation equation" if key == "Vtransform" else "stretching function"}'
            v[:] = int(vgrid_params[key])

    # --- h ---
    v = ds.createVariable('h', 'f8', ('eta_rho', 'xi_rho'))
    v.long_name = 'bathymetry at RHO-points'
    v.units = 'meter'
    v.coordinates = 'lon_rho lat_rho'
    v[:] = metrics['h']

    # --- grid coordinates ---
    for vname, long_name, units, dims in [
        ('lon_rho', 'longitude of RHO-points', 'degree_east', ('eta_rho', 'xi_rho')),
        ('lat_rho', 'latitute of RHO-points', 'degree_north', ('eta_rho', 'xi_rho')),
        ('lon_u', 'longitude of U-points', 'degree_east', ('eta_u', 'xi_u')),
        ('lat_u', 'latitute of U-points', 'degree_north', ('eta_u', 'xi_u')),
        ('lon_v', 'longitude of V-points', 'degree_east', ('eta_v', 'xi_v')),
        ('lat_v', 'latitute of V-points', 'degree_north', ('eta_v', 'xi_v')),
    ]:
        if vname in metrics:
            v = ds.createVariable(vname, 'f8', dims)
            v.long_name = long_name
            v.units = units
            v.standard_name = {'degree_east': 'longitude', 'degree_north': 'latitude'}.get(units, units)
            v[:] = metrics[vname]

    # --- s-coordinate params ---
    for key, value, long_name, units in [
        ('theta_s', vgrid_params.get('theta_s', 5.0),
         'S-coordinate surface control parameter', ''),
        ('theta_b', vgrid_params.get('theta_b', 0.4),
         'S-coordinate bottom control parameter', ''),
        ('Tcline', vgrid_params.get('Tcline', 10.0),
         'S-coordinate surface/bottom layer width', 'meter'),
        ('hc', vgrid_params.get('hc', 10.0),
         'S-coordinate parameter, critical depth', 'meter'),
    ]:
        v = ds.createVariable(key, 'f8', ())
        v.long_name = long_name
        if units:
            v.units = units
        v[:] = float(value)

    # --- s_rho, s_w, Cs_r, Cs_w ---
    for key, dim, lon, valid_min, valid_max in [
        ('s_rho', 's_rho', 'S-coordinate at RHO-points', -1.0, 0.0),
        ('s_w', 's_w', 'S-coordinate at W-points', -1.0, 0.0),
    ]:
        if key in vgrid_params:
            v = ds.createVariable(key, 'f8', (dim,))
            v.long_name = lon
            v.valid_min = valid_min
            v.valid_max = valid_max
            v.positive = 'up'
            v.standard_name = 'ocean_s_coordinate_g2'
            v.formula_terms = f's: {key} C: Cs_{key[-1]} eta: zeta depth: h depth_c: hc'
            v[:] = np.asarray(vgrid_params[key], 'f8')

    for key, dim, lon in [
        ('Cs_r', 's_rho', 'S-coordinate stretching function at RHO-points'),
        ('Cs_w', 's_w', 'S-coordinate stretching function at W-points'),
    ]:
        if key in vgrid_params:
            v = ds.createVariable(key, 'f8', (dim,))
            v.long_name = lon
            v.valid_min = -1.0
            v.valid_max = 0.0
            v[:] = np.asarray(vgrid_params[key], 'f8')

    # --- ocean_time ---
    v = ds.createVariable('ocean_time', 'f8', ('ocean_time',))
    v.long_name = 'time since initialization'
    v.units = f'seconds since {time_ref} 00:00:00'
    v.calendar = 'gregorian'
    v[:] = ocean_time

    # --- data variables ---
    def _def_var(name, data, dims, long_name, units):
        """Define a data variable matching reference IC convention."""
        v = ds.createVariable(name, 'f8', dims)
        v.long_name = long_name
        v.units = units
        v.time = 'ocean_time'
        # coordinates must include the time dimension name at the end
        coords = ' '.join(d for d in dims if d != 'ocean_time' and d != 's_rho')
        if name in ('u', 'ubar'):
            coords = coords.replace('eta_u xi_u', 'lon_u lat_u').replace('eta_rho xi_rho', 'lon_u lat_u').replace('eta_v xi_v', 'lon_v lat_v')
        elif name in ('v', 'vbar'):
            coords = coords.replace('eta_v xi_v', 'lon_v lat_v')
        else:
            coords = coords.replace('eta_rho xi_rho', 'lon_rho lat_rho')
        # Append s_rho for 3D and ocean_time for all
        if 's_rho' in dims:
            coords = coords + ' s_rho'
        coords = coords + ' ocean_time'
        v.coordinates = coords
        # Replace NaN and spval with 0 for ROMS compatibility
        data = np.asarray(data, 'f8').copy()
        data[data > 1e30] = np.nan
        data = np.nan_to_num(data, nan=0.0)
        v[:] = data

    _def_var('zeta', zeta[None], ('ocean_time', 'eta_rho', 'xi_rho'),
             'free-surface', 'meter')
    _def_var('temp', temp[None], ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
             'potential temperature', 'Celsius')
    _def_var('salt', salt[None], ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
             'salinity', 'PSU')
    _def_var('u', u_mom[None], ('ocean_time', 's_rho', 'eta_u', 'xi_u'),
             'u-momentum component', 'meter second-1')
    _def_var('v', v_mom[None], ('ocean_time', 's_rho', 'eta_v', 'xi_v'),
             'v-momentum component', 'meter second-1')
    _def_var('ubar', ubar[None], ('ocean_time', 'eta_u', 'xi_u'),
             'vertically integrated u-momentum component', 'meter second-1')
    _def_var('vbar', vbar[None], ('ocean_time', 'eta_v', 'xi_v'),
             'vertically integrated v-momentum component', 'meter second-1')

    ds.close()


# ---------------------------------------------------------------------------
# Helpers shared across IC modules
# ---------------------------------------------------------------------------

def _fill_nan_2d(arr, max_pass=5):
    """Replace NaN cells with the nearest valid neighbour (distance transform).

    Uses scipy.ndimage.distance_transform_edt for fast nearest-neighbor
    filling on regular grids. Much faster than KDTree for this use case.
    """
    from scipy.ndimage import distance_transform_edt

    arr = arr.copy()
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    if np.all(nan_mask):
        return arr

    # distance_transform_edt returns indices of nearest non-NaN pixel
    _, idx = distance_transform_edt(~nan_mask, return_indices=True)
    arr[nan_mask] = arr[idx[0][nan_mask], idx[1][nan_mask]]
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
    """Convert date string / datetime / numeric to seconds since 1970-01-01 (UTC)."""
    import calendar
    if t is None:
        return None
    if isinstance(t, (int, float, np.floating, np.integer)):
        return float(t)
    if isinstance(t, datetime):
        return calendar.timegm(t.timetuple()) + t.microsecond / 1e6
    if isinstance(t, str):
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%d-%b-%Y']:
            try:
                dt = datetime.strptime(t, fmt)
                return calendar.timegm(dt.timetuple())
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
    out['depth'] = -np.abs(np.asarray(out['depth'], dtype=float))

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
            if hasattr(arr, 'mask'):
                arr = arr.filled(np.nan)
            out[vn] = arr

    ds.close()
    return out


def _read_roms_grid(grid_file):
    """Read ROMS grid into dict."""
    g = nc4.Dataset(grid_file)
    m = {}
    for v in ['h','lon_rho','lat_rho','mask_rho','mask_u','mask_v','angle',
              'lon_u','lat_u','lon_v','lat_v']:
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


# ---------------------------------------------------------------------------
# CMEMS-specific helpers (regular-grid interpolation)
# ---------------------------------------------------------------------------

def cmems_read_var(file_path, var_name, time_index=0):
    """
    Read a single variable from CMEMS NetCDF, returning data on native grid.

    Returns
    -------
    data : ndarray (nlat, nlon) or (nlat, nlon, ndepth)
    lon_1d : ndarray (nlon,)
    lat_1d : ndarray (nlat,)
    depth : ndarray (ndepth,) or None (negative depths for 3D vars)
    time_units : str or None
    """
    ds = nc4.Dataset(file_path, 'r')

    lon_name = next((n for n in ['longitude', 'lon', 'nav_lon'] if n in ds.dimensions), None)
    lat_name = next((n for n in ['latitude', 'lat', 'nav_lat'] if n in ds.dimensions), None)
    depth_name = next((n for n in ['depth', 'lev', 'deptht'] if n in ds.dimensions), None)
    time_name = next((n for n in ['time', 'time_counter', 't'] if n in ds.dimensions), None)

    lon_1d = np.asarray(ds.variables[lon_name][:], dtype=float) if lon_name else None
    lat_1d = np.asarray(ds.variables[lat_name][:], dtype=float) if lat_name else None

    if var_name not in ds.variables:
        aliases = {
            'zos': ['zos', 'sla', 'adt', 'ssh', 'surf_el'],
            'thetao': ['thetao', 't', 'temperature', 'votemper'],
            'so': ['so', 's', 'salinity', 'vosaline'],
            'uo': ['uo', 'u', 'water_u', 'vozocrtx'],
            'vo': ['vo', 'v', 'water_v', 'vomecrty'],
        }
        for aliases_list in aliases.values():
            if var_name in aliases_list:
                for alias in aliases_list:
                    if alias in ds.variables:
                        var_name = alias
                        break
                break

    v = ds.variables[var_name]
    var_dims = v.dimensions
    has_time = (time_name and time_name in var_dims and len(var_dims) >= 3)
    if has_time and v.shape[0] > 1:
        data = np.array(v[min(time_index, v.shape[0] - 1)], dtype=float)
    else:
        data = np.array(v[:], dtype=float)

    fv = getattr(v, '_FillValue', None)
    if fv is not None:
        data[data == fv] = np.nan
    sf = getattr(v, 'scale_factor', None)
    soff = getattr(v, 'add_offset', None)
    if sf is not None:
        data = data * (sf if sf != 0 else 1.0) + (soff if soff is not None else 0.0)

    depth = None
    if depth_name and depth_name in ds.variables:
        depth = -np.abs(np.asarray(ds.variables[depth_name][:], dtype=float))

    time_units = None
    if time_name and time_name in ds.variables:
        time_units = getattr(ds.variables[time_name], 'units', None)

    ds.close()
    return data, lon_1d, lat_1d, depth, time_units


def cmems_interp_2d(data, lon_1d, lat_1d, lon_rho, lat_rho):
    """Interpolate 2D CMEMS field to ROMS grid using RegularGridInterpolator."""
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


def cmems_interp_3d(data, lon_1d, lat_1d, depth_vals, lon_rho, lat_rho, z_r):
    """Interpolate 3D CMEMS field: horizontal per layer, vertical per point."""
    nlat_r, nlon_r, N = z_r.shape
    ndepth_src = data.shape[2]

    Flev = np.full((nlat_r, nlon_r, ndepth_src), np.nan)
    for k in range(ndepth_src):
        Flev[:, :, k] = cmems_interp_2d(data[:, :, k], lon_1d, lat_1d, lon_rho, lat_rho)

    Fout = np.full((nlat_r, nlon_r, N), np.nan)
    for i in range(nlat_r):
        for j in range(nlon_r):
            src = Flev[i, j, :]
            valid = ~np.isnan(src)
            if np.sum(valid) < 2:
                continue
            src_z = depth_vals[valid]
            src_v = src[valid]
            target_z = z_r[i, j, :]
            target_z_clipped = np.clip(target_z, src_z.min(), src_z.max())
            Fout[i, j, :] = np.interp(target_z_clipped, src_z, src_v)

    return Fout


def cmems_fill_nan_2d(data):
    """Fill NaN using distance_transform_edt on each layer."""
    from scipy.ndimage import distance_transform_edt
    result = data.copy()
    if result.ndim == 2:
        nan_mask = np.isnan(result)
        if nan_mask.any() and not nan_mask.all():
            _, idx = distance_transform_edt(~nan_mask, return_indices=True)
            result[nan_mask] = result[idx[0][nan_mask], idx[1][nan_mask]]
        return result
    for k in range(result.shape[2]):
        layer = result[:, :, k]
        nan_mask = np.isnan(layer)
        if nan_mask.any() and not nan_mask.all():
            _, idx = distance_transform_edt(~nan_mask, return_indices=True)
            layer[nan_mask] = layer[idx[0][nan_mask], idx[1][nan_mask]]
            result[:, :, k] = layer
    return result


def cmems_fill_nans_vertically(data):
    """Fill NaN/0 columns vertically with nearest valid neighbor (per column)."""
    result = data.copy()
    nlat, nlon, ndepth = result.shape
    for i in range(nlat):
        for j in range(nlon):
            col = result[i, j, :]
            valid = np.where(~np.isnan(col) & (col != 0))[0]
            nan_idx = np.where(np.isnan(col) | (col == 0))[0]
            if len(valid) > 0 and len(nan_idx) > 0:
                for k in nan_idx:
                    nearest = valid[np.argmin(np.abs(valid - k))]
                    col[k] = col[nearest]
                result[i, j, :] = col
    return result
