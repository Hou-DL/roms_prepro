"""
Old core functions kept for backward compatibility (mercator_to_roms_ini).
New CMEMS processing uses _core.py with reference-matching implementations.
"""

import numpy as np
import netCDF4 as nc4
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree


def horizontal_interp(src_lon, src_lat, src_var, dst_lon, dst_lat,
                      mask=None, fill_value=np.nan):
    """Interpolate from source grid to destination grid (for single-file mode)."""
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

    # Scattered data interpolation
    if src_var.ndim == 2:
        return _interp_2d(src_lon, src_lat, src_var, dst_lon, dst_lat, mask)
    else:
        nlev = src_var.shape[0]
        result = np.zeros((nlev, dst_lon.shape[0], dst_lon.shape[1]))
        for k in range(nlev):
            result[k] = _interp_2d(src_lon, src_lat, src_var[k], dst_lon, dst_lat)
        if mask is not None:
            result[:, mask < 0.5] = fill_value
        return result


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


def _read_roms_grid(grid_file):
    """Read ROMS grid into dict."""
    g = nc4.Dataset(grid_file)
    m = {}
    for v in ['h', 'lon_rho', 'lat_rho', 'mask_rho', 'mask_u', 'mask_v',
              'angle', 'lon_u', 'lat_u', 'lon_v', 'lat_v']:
        if v in g.variables:
            m[v] = g.variables[v][:]
    g.close()
    return m


# --- Vertical interpolation ---

def z_to_sigma(var_z, z_levels, sigma_depth, fill_value=np.nan):
    """Interp from z-levels to ROMS sigma coordinates."""
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
    """Interp from ROMS sigma to z-levels."""
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


# --- Velocity rotation ---

def rotate_uv(u, v, src_angle, dst_angle):
    """Rotate u,v between grids."""
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
    """Average velocity to C-grid."""
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
    """Compute barotropic velocity."""
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


# --- IC file writer ---

def write_ic_file(filename, metrics, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar,
                  time_ref='2025-01-01'):
    """Write ROMS IC NetCDF file."""
    spval = 1e37
    ds = nc4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    ds.type = 'ROMS INITIALIZATION file'
    ds.title = 'ROMS initial conditions from roms_prepro'
    ds.data_source = 'CMEMS reanalysis'

    ny, nx = metrics['lon_rho'].shape
    N = temp.shape[0]
    mask = metrics.get('mask_rho', np.ones((ny, nx)))

    for d, name in [((nx), 'xi_rho'), ((nx - 1), 'xi_u'), ((nx), 'xi_v'),
                    ((nx - 1), 'xi_psi'), ((ny), 'eta_rho'), ((ny), 'eta_u'),
                    ((ny - 1), 'eta_v'), ((ny - 1), 'eta_psi'),
                    ((N), 's_rho'), ((N + 1), 's_w'), ((2), 'tracer'),
                    ((None), 'ocean_time')]:
        ds.createDimension(name, d)

    v = ds.createVariable('spherical', 'i4', ())
    v.long_name = 'grid type logical switch'
    v[:] = 1

    for key in ('Vtransform', 'Vstretching'):
        if key in vgrid_params:
            v = ds.createVariable(key, 'i4', ())
            v[:] = int(vgrid_params[key])

    v = ds.createVariable('h', 'f8', ('eta_rho', 'xi_rho'))
    v.long_name = 'bathymetry at RHO-points'
    v.units = 'meter'
    v[:] = metrics['h']

    for vname, long_name, units, dims in [
        ('lon_rho', 'longitude of RHO-points', 'degree_east', ('eta_rho', 'xi_rho')),
        ('lat_rho', 'latitude of RHO-points', 'degree_north', ('eta_rho', 'xi_rho')),
        ('lon_u', 'longitude of U-points', 'degree_east', ('eta_u', 'xi_u')),
        ('lat_u', 'latitude of U-points', 'degree_north', ('eta_u', 'xi_u')),
        ('lon_v', 'longitude of V-points', 'degree_east', ('eta_v', 'xi_v')),
        ('lat_v', 'latitude of V-points', 'degree_north', ('eta_v', 'xi_v')),
    ]:
        if vname in metrics:
            v = ds.createVariable(vname, 'f8', dims)
            v.long_name = long_name
            v.units = units
            v[:] = metrics[vname]

    for key, value, long_name, units in [
        ('theta_s', vgrid_params.get('theta_s', 5.0), 'S-coordinate surface control parameter', ''),
        ('theta_b', vgrid_params.get('theta_b', 0.4), 'S-coordinate bottom control parameter', ''),
        ('Tcline', vgrid_params.get('Tcline', 10.0), 'S-coordinate surface/bottom layer width', 'meter'),
        ('hc', vgrid_params.get('hc', 10.0), 'S-coordinate parameter, critical depth', 'meter'),
    ]:
        v = ds.createVariable(key, 'f8', ())
        v.long_name = long_name
        if units:
            v.units = units
        v[:] = float(value)

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

    v = ds.createVariable('ocean_time', 'f8', ('ocean_time',))
    v.long_name = 'time since initialization'
    v.units = f'seconds since {time_ref} 00:00:00'
    v.calendar = 'gregorian'
    v[:] = ocean_time

    def _def_var(name, data, dims, long_name, units):
        v = ds.createVariable(name, 'f8', dims)
        v.long_name = long_name
        v.units = units
        v.time = 'ocean_time'
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
        v.coordinates = coords
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
    _def_var('u', u[None], ('ocean_time', 's_rho', 'eta_u', 'xi_u'),
             'u-momentum component', 'meter second-1')
    _def_var('v', v[None], ('ocean_time', 's_rho', 'eta_v', 'xi_v'),
             'v-momentum component', 'meter second-1')
    _def_var('ubar', ubar[None], ('ocean_time', 'eta_u', 'xi_u'),
             'vertically integrated u-momentum', 'meter second-1')
    _def_var('vbar', vbar[None], ('ocean_time', 'eta_v', 'xi_v'),
             'vertically integrated v-momentum', 'meter second-1')

    ds.close()


# --- Date/time helpers ---

def _parse_date(date_str):
    """Parse date string to seconds since epoch."""
    from datetime import datetime
    if date_str is None:
        return None
    if isinstance(date_str, (int, float)):
        return float(date_str)
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
    return (dt - datetime(1970, 1, 1)).total_seconds()


def _get_time(file_path, time_var=None):
    """Read time array from file."""
    import netCDF4 as nc4
    try:
        ds = nc4.Dataset(file_path, 'r')
        candidates = [time_var] if time_var else ['time', 'time_counter', 'ocean_time']
        for c in candidates:
            if c and c in ds.variables:
                t = np.atleast_1d(ds.variables[c][:]).astype(float)
                units = getattr(ds.variables[c], 'units', '')
                ds.close()
                import re
                m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', str(units))
                if m:
                    ref_epoch = _parse_date(m.group(1))
                    if 'days' in str(units).lower():
                        t = t * 86400.0 + ref_epoch
                    elif 'hours' in str(units).lower():
                        t = t * 3600.0 + ref_epoch
                    else:
                        t = t + ref_epoch
                return t
        ds.close()
        return None
    except Exception:
        return None


def _match_idx(source_times, target_date):
    """Find closest time index."""
    target = _parse_date(target_date) if target_date else None
    if target is None:
        return 0
    return int(np.argmin(np.abs(source_times - target)))
