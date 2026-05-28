"""
Core interpolation routines for ROMS initial/boundary conditions.

Provides:
- horizontal_interp: source lon/lat → target lon/lat (2D and 3D)
- z_to_sigma: standard z-levels → ROMS sigma coordinates
- sigma_to_z: ROMS sigma → standard z-levels
- rotate_uv: rotate u,v from source angle to target angle
- uv_to_cgrid: average rotated vectors to ROMS C-grid positions
- extract_boundary: extract N/S/E/W edges from full fields
- write_bry_file: write standard ROMS BC NetCDF
- write_ic_file: write standard ROMS IC NetCDF
"""

import numpy as np
from datetime import datetime
import netCDF4 as nc4
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Horizontal interpolation
# ---------------------------------------------------------------------------

def _interp_2d(src_lon, src_lat, src_var, dst_lon, dst_lat, mask=None):
    """
    Interpolate a 2-D field from source to destination grid.
    """
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
# Boundary extraction
# ---------------------------------------------------------------------------

EDGE_NAMES = ['west', 'east', 'south', 'north']


def extract_boundary(field_2d, field_3d_u, field_3d_v, boundaries):
    """
    Extract N/S/E/W boundary segments from full 2D and 3D fields.

    Parameters
    ----------
    field_2d : dict
        Dict of 2D fields: {'zeta': (eta, xi), 'ubar': (eta_u, xi_u), ...}
    field_3d_u : ndarray (N, eta_u, xi_u)
        3D u-velocity.
    field_3d_v : ndarray (N, eta_v, xi_v)
        3D v-velocity.
    boundaries : list of bool [W, E, S, N]
        Which boundaries are active.

    Returns
    -------
    bry_data : dict of dict
        bry_data['temp'][0] = temp_west (N, len_west)
        bry_data['temp'][1] = temp_east, etc.
    """
    edges = {}
    for b, active in enumerate(boundaries):
        if not active:
            continue
        edge = EDGE_NAMES[b]
        edges[edge] = b

    result = {}
    for vname, f2d in field_2d.items():
        result[vname] = [None] * 4
        for edge, b in edges.items():
            if edge in ('west', 'east'):
                idx = 0 if edge == 'west' else -1
                result[vname][b] = f2d[:, idx] if f2d.ndim == 2 else f2d[idx]
            else:
                idx = 0 if edge == 'south' else -1
                result[vname][b] = f2d[idx, :] if f2d.ndim == 2 else f2d[:, idx]

    # 3D U: extracted at u-point edges
    result['u_3d'] = [None] * 4
    for edge, b in edges.items():
        if edge in ('west', 'east'):
            idx = 0 if edge == 'west' else -1
            result['u_3d'][b] = field_3d_u[:, :, idx]  # (N, eta_u)
        else:
            idx = 0 if edge == 'south' else -1
            result['u_3d'][b] = field_3d_u[:, idx, :]  # (N, xi_u)

    result['v_3d'] = [None] * 4
    for edge, b in edges.items():
        if edge in ('west', 'east'):
            idx = 0 if edge == 'west' else -1
            result['v_3d'][b] = field_3d_v[:, :, idx]  # (N, eta_v)
        else:
            idx = 0 if edge == 'south' else -1
            result['v_3d'][b] = field_3d_v[:, idx, :]  # (N, xi_v)

    return result


# ---------------------------------------------------------------------------
# NC file writers
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
                  zeta, temp, salt, u, v, ubar, vbar):
    """Write a standard ROMS initial conditions NetCDF file."""
    spval = 1e37
    ds = nc4.Dataset(filename, 'w', format='NETCDF3_64BIT')
    ds.Description = 'ROMS initial conditions'
    ds.Author = 'roms_prepro.icbc'
    ds.Created = datetime.now().isoformat()

    ny, nx = metrics['lon_rho'].shape
    N = temp.shape[0]
    mask = metrics.get('mask_rho', np.ones((ny, nx)))

    for d in [('xi_rho', nx), ('xi_u', nx - 1), ('xi_v', nx),
              ('xi_psi', nx - 1)]:
        ds.createDimension(d[0], d[1])
    for d in [('eta_rho', ny), ('eta_u', ny), ('eta_v', ny - 1),
              ('eta_psi', ny - 1)]:
        ds.createDimension(d[0], d[1])
    ds.createDimension('s_rho', N)
    ds.createDimension('ocean_time', None)

    ds.createVariable('spherical', 'c')[:] = 'T'
    for key in ('Vtransform', 'Vstretching'):
        if key in vgrid_params:
            ds.createVariable(key, 'i4', ())[:] = vgrid_params[key]

    _put_var(ds, 'h', metrics['h'], ('eta_rho', 'xi_rho'),
             long_name='bathymetry', units='meter')
    _put_var(ds, 'lon_rho', metrics['lon_rho'], ('eta_rho', 'xi_rho'),
             long_name='longitude', units='degree_east',
             coordinates='lon_rho lat_rho', field='longitude, scalar')
    _put_var(ds, 'lat_rho', metrics['lat_rho'], ('eta_rho', 'xi_rho'),
             long_name='latitude', units='degree_north',
             coordinates='lon_rho lat_rho', field='latitude, scalar')
    _put_var(ds, 'mask_rho', metrics['mask_rho'], ('eta_rho', 'xi_rho'))

    # Staggered-grid coordinates (required by ROMS for coordinates attribute references)
    if 'lon_u' in metrics:
        _put_var(ds, 'lon_u', metrics['lon_u'], ('eta_u', 'xi_u'),
                 long_name='longitude at u-points', units='degree_east')
    if 'lat_u' in metrics:
        _put_var(ds, 'lat_u', metrics['lat_u'], ('eta_u', 'xi_u'),
                 long_name='latitude at u-points', units='degree_north')
    if 'lon_v' in metrics:
        _put_var(ds, 'lon_v', metrics['lon_v'], ('eta_v', 'xi_v'),
                 long_name='longitude at v-points', units='degree_east')
    if 'lat_v' in metrics:
        _put_var(ds, 'lat_v', metrics['lat_v'], ('eta_v', 'xi_v'),
                 long_name='latitude at v-points', units='degree_north')

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
        ds.createVariable(key, 'f8', ())
        setattr(ds.variables[key], 'long_name', long_name)
        if units:
            setattr(ds.variables[key], 'units', units)
        ds.variables[key][:] = float(value)
    if 's_rho' in vgrid_params:
        _put_var(ds, 's_rho', np.asarray(vgrid_params['s_rho'], 'f8'),
                 ('s_rho',), long_name='S-coordinate at RHO-points')
    if 'Cs_r' in vgrid_params:
        _put_var(ds, 'Cs_r', np.asarray(vgrid_params['Cs_r'], 'f8'),
                 ('s_rho',), long_name='S-coordinate stretching at RHO-points')

    ds.createVariable('ocean_time', 'f8', ('ocean_time',))
    ds.variables['ocean_time'][:] = ocean_time
    ds.variables['ocean_time'].units = 'seconds since 2000-01-01 00:00:00'
    ds.variables['ocean_time'].field = 'time, scalar'
    _put_var(ds, 'zeta', zeta[None, :, :], ('ocean_time', 'eta_rho', 'xi_rho'),
             long_name='free surface', units='meter',
             time='ocean_time', coordinates='lon_rho lat_rho',
             field='free-surface, scalar')
    temp_arr = temp[None].copy()
    temp_arr[temp_arr == spval] = np.nan
    temp_arr[:, :, mask < 0.5] = spval
    temp_arr = np.where(np.isnan(temp_arr), spval, temp_arr)
    _put_var(ds, 'temp', temp_arr, ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
             long_name='potential temperature', units='Celsius',
             time='ocean_time', coordinates='lon_rho lat_rho',
             field='temperature, scalar', _FillValue=spval, missing_value=spval)
    salt_arr = salt[None].copy()
    salt_arr = np.where(np.isnan(salt_arr), spval, salt_arr)
    salt_arr[:, :, mask < 0.5] = spval
    _put_var(ds, 'salt', salt_arr, ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
             long_name='salinity', units='PSU',
             time='ocean_time', coordinates='lon_rho lat_rho',
             field='salinity, scalar', _FillValue=spval, missing_value=spval)
    _put_var(ds, 'u', u[None], ('ocean_time', 's_rho', 'eta_u', 'xi_u'),
             long_name='u-momentum', units='m/s',
             time='ocean_time', coordinates='lon_u lat_u',
             field='u-velocity, scalar', _FillValue=spval, missing_value=spval)
    _put_var(ds, 'v', v[None], ('ocean_time', 's_rho', 'eta_v', 'xi_v'),
             long_name='v-momentum', units='m/s',
             time='ocean_time', coordinates='lon_v lat_v',
             field='v-velocity, scalar', _FillValue=spval, missing_value=spval)
    _put_var(ds, 'ubar', ubar[None], ('ocean_time', 'eta_u', 'xi_u'),
             long_name='barotropic u', units='m/s',
             time='ocean_time', coordinates='lon_u lat_u',
             field='ubar-velocity, scalar', _FillValue=spval, missing_value=spval)
    _put_var(ds, 'vbar', vbar[None], ('ocean_time', 'eta_v', 'xi_v'),
             long_name='barotropic v', units='m/s',
             time='ocean_time', coordinates='lon_v lat_v',
             field='vbar-velocity, scalar', _FillValue=spval, missing_value=spval)
    ds.close()


def write_bry_file(filename, metrics, vgrid_params, boundaries,
                   bry_data, bry_time):
    """
    Write a standard ROMS boundary conditions NetCDF file.

    Parameters
    ----------
    filename : str
    metrics : dict
    vgrid_params : dict
    boundaries : list of bool [W, E, S, N]
    bry_data : dict
        Keys: 'zeta', 'temp', 'salt', 'u', 'v', 'ubar', 'vbar'.
        Each is a list of length 4: [west, east, south, north].
    bry_time : ndarray (ntime,)
    """
    ds = nc4.Dataset(filename, 'w', format='NETCDF3_64BIT')
    ds.Description = 'ROMS boundary conditions'
    ds.Author = 'roms_prepro.icbc'
    ds.Created = datetime.now().isoformat()

    ny, nx = metrics['lon_rho'].shape

    # Determine N from any 3D variable (data is (ntime, N, ...))
    N = 1
    for v in ['temp', 'salt', 'u', 'v']:
        if v in bry_data:
            for arr in bry_data[v]:
                if arr is not None and arr.ndim >= 2:
                    N = arr.shape[1]
                    break
            if N > 1:
                break

    ntime = len(bry_time)

    # Global dimensions
    ds.createDimension('xi_rho', nx)
    ds.createDimension('eta_rho', ny)
    ds.createDimension('s_rho', N)

    # ROMS expects per-variable time dimensions for boundary files.
    # Use fixed size since we know ntime upfront.
    # NETCDF3_64BIT only supports one unlimited dimension, so use fixed.
    ds.createDimension('zeta_time', ntime)
    ds.createDimension('v2d_time', ntime)
    ds.createDimension('v3d_time', ntime)
    ds.createDimension('temp_time', ntime)
    ds.createDimension('salt_time', ntime)
    ds.createVariable('spherical', 'c')[:] = 'T'
    for key in ('Vtransform', 'Vstretching'):
        if key in vgrid_params:
            ds.createVariable(key, 'i4', ())[:] = vgrid_params[key]

    _put_var(ds, 'h', metrics['h'], ('eta_rho', 'xi_rho'),
             long_name='bathymetry', units='meter')

    # S-coordinate parameters (required by ROMS for vertical grid)
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
        ds.createVariable(key, 'f8', ())
        setattr(ds.variables[key], 'long_name', long_name)
        if units:
            setattr(ds.variables[key], 'units', units)
        ds.variables[key][:] = float(value)

    # s_rho, Cs_r from stretching function
    if 's_rho' in vgrid_params and 'Cs_r' in vgrid_params:
        _put_var(ds, 's_rho', np.asarray(vgrid_params['s_rho'], 'f8'),
                 ('s_rho',), long_name='S-coordinate at RHO-points')
        _put_var(ds, 'Cs_r', np.asarray(vgrid_params['Cs_r'], 'f8'),
                 ('s_rho',), long_name='S-coordinate stretching at RHO-points')

    # Per-edge coordinate and boundary variables
    for b, active in enumerate(boundaries):
        if not active:
            continue
        edge = EDGE_NAMES[b]
        is_ew = edge in ('west', 'east')

        if is_ew:
            rho_dim_name = f'eta_rho_{edge}'
            u_dim_name = f'eta_u_{edge}'
            v_dim_name = f'eta_v_{edge}'
            rho_len = ny
            u_len = ny
            v_len = ny - 1
        else:
            rho_dim_name = f'xi_rho_{edge}'
            u_dim_name = f'xi_u_{edge}'
            v_dim_name = f'xi_v_{edge}'
            rho_len = nx
            u_len = nx - 1
            v_len = nx

        ds.createDimension(rho_dim_name, rho_len)
        if u_len != rho_len:
            ds.createDimension(u_dim_name, u_len)
        if v_len != rho_len:
            ds.createDimension(v_dim_name, v_len)

    # Write time variables (per-variable convention used by ROMS)
    def _write_time_var(tvar_name):
        tvar = ds.createVariable(tvar_name, 'f8', (tvar_name,))
        tvar[:] = bry_time
        tvar.units = 'seconds since 2000-01-01 00:00:00'
        tvar.long_name = tvar_name
        tvar.field = f'{tvar_name}, scalar, series'
        return tvar_name

    zeta_time_name = _write_time_var('zeta_time')
    v2d_time_name = _write_time_var('v2d_time')
    v3d_time_name = _write_time_var('v3d_time')
    temp_time_name = _write_time_var('temp_time')
    salt_time_name = _write_time_var('salt_time')

    # Write boundary variables with time attribute pointing to proper time variable
    for b, active in enumerate(boundaries):
        if not active:
            continue
        edge = EDGE_NAMES[b]
        is_ew = edge in ('west', 'east')

        rho_dim_name = f'eta_rho_{edge}' if is_ew else f'xi_rho_{edge}'
        u_dim_name = f'eta_u_{edge}' if is_ew else f'xi_u_{edge}'
        v_dim_name = f'eta_v_{edge}' if is_ew else f'xi_v_{edge}'

        if 'zeta' in bry_data and bry_data['zeta'][b] is not None:
            _put_var(ds, f'zeta_{edge}', bry_data['zeta'][b],
                     (zeta_time_name, rho_dim_name),
                     long_name=f'zeta, {edge} boundary', time=zeta_time_name)

        if 'temp' in bry_data and bry_data['temp'][b] is not None:
            _put_var(ds, f'temp_{edge}', bry_data['temp'][b],
                     (temp_time_name, 's_rho', rho_dim_name),
                     long_name=f'temp, {edge} boundary', time=temp_time_name)

        if 'salt' in bry_data and bry_data['salt'][b] is not None:
            _put_var(ds, f'salt_{edge}', bry_data['salt'][b],
                     (salt_time_name, 's_rho', rho_dim_name),
                     long_name=f'salt, {edge} boundary', time=salt_time_name)

        if 'u' in bry_data and bry_data['u'][b] is not None:
            dim_name = u_dim_name if u_dim_name in ds.dimensions else rho_dim_name
            _put_var(ds, f'u_{edge}', bry_data['u'][b],
                     (v3d_time_name, 's_rho', dim_name),
                     long_name=f'u-momentum, {edge} boundary', time=v3d_time_name)

        if 'v' in bry_data and bry_data['v'][b] is not None:
            dim_name = v_dim_name if v_dim_name in ds.dimensions else rho_dim_name
            _put_var(ds, f'v_{edge}', bry_data['v'][b],
                     (v3d_time_name, 's_rho', dim_name),
                     long_name=f'v-momentum, {edge} boundary', time=v3d_time_name)

        if 'ubar' in bry_data and bry_data['ubar'][b] is not None:
            dim_name = u_dim_name if u_dim_name in ds.dimensions else rho_dim_name
            _put_var(ds, f'ubar_{edge}', bry_data['ubar'][b],
                     (v2d_time_name, dim_name),
                     long_name=f'ubar, {edge} boundary', time=v2d_time_name)

        if 'vbar' in bry_data and bry_data['vbar'][b] is not None:
            dim_name = v_dim_name if v_dim_name in ds.dimensions else rho_dim_name
            _put_var(ds, f'vbar_{edge}', bry_data['vbar'][b],
                     (v2d_time_name, dim_name),
                     long_name=f'vbar, {edge} boundary', time=v2d_time_name)

    ds.close()