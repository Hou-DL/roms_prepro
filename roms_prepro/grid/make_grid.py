"""
Create complete ROMS grid / grid files.

Core workflow:
  1. Define the domain extent (lon/lat corners and resolution).
  2. Generate a curvilinear orthogonal grid covering the domain.
  3. Optionally interpolate bathymetry from a source dataset.
  4. Smooth bathymetry to satisfy the rx0 roughness criterion.
  5. Compute all grid metrics (pm, pn, angle, f, masks).
  6. Write the standard ROMS grid NetCDF file.

Examples
--------
>>> from roms_prepro.grid import create_roms_grid
>>> create_roms_grid(
...     lon_corners=[117, 117, 127, 127],
...     lat_corners=[35, 41, 41, 35],
...     nx=152, ny=122,
...     grid_file='my_grid.nc'
... )
"""

import warnings

import numpy as np
from datetime import datetime
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


# ---------------------------------------------------------------------------
# 1.  Grid generation from corner points
# ---------------------------------------------------------------------------

def make_grid_corner(lon_corners, lat_corners, nx, ny):
    """
    Generate a simple rectilinear lon/lat grid from four corner points.

    The grid is created by bilinearly interpolating the corner coordinates
    onto a uniform (ny, nx) grid in index space.

    Parameters
    ----------
    lon_corners : array-like of length 4
        Longitudes of the four corners [SW, NW, NE, SE].
    lat_corners : array-like of length 4
        Latitudes of the four corners [SW, NW, NE, SE].
    nx : int
        Number of RHO-points in the xi-direction.
    ny : int
        Number of RHO-points in the eta-direction.

    Returns
    -------
    lon_rho : ndarray (ny, nx)
        Longitudes at RHO-points.
    lat_rho : ndarray (ny, nx)
        Latitudes at RHO-points.
    """
    lc = np.asarray(lon_corners)
    lac = np.asarray(lat_corners)

    # Index-space points for corners
    idx_pts = np.array([[1, 1], [1, ny], [nx, ny], [nx, 1]])
    val_lon = np.array([lc[0], lc[1], lc[2], lc[3]])
    val_lat = np.array([lac[0], lac[1], lac[2], lac[3]])

    xi, eta = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1))

    interp_lon = LinearNDInterpolator(idx_pts, val_lon)
    interp_lat = LinearNDInterpolator(idx_pts, val_lat)

    lon_rho = interp_lon(xi, eta)
    lat_rho = interp_lat(xi, eta)

    return lon_rho, lat_rho


# ---------------------------------------------------------------------------
# 2.  Grid metrics
# ---------------------------------------------------------------------------

def _lonlat_to_grid_metrics(lon_rho, lat_rho):
    """
    Compute ROMS grid metrics from RHO-point lon/lat using the algorithm
    from roms_metrics.m (Hernan Arango / ROMS group).

    Uses PSI-point corner cells with Earth-radius spherical distances to
    compute pm, pn, angle, dndx, dmde, and Cartesian coordinates (x_rho,
    y_rho) that are consistent with the ROMS governing equations.

    Parameters
    ----------
    lon_rho, lat_rho : ndarray (eta_rho, xi_rho)
        RHO-point coordinates in degrees.

    Returns
    -------
    metrics : dict
    """
    lon_rho = np.asarray(lon_rho, dtype=float)
    lat_rho = np.asarray(lat_rho, dtype=float)
    ny, nx = lon_rho.shape

    deg2rad = np.pi / 180.0
    Eradius = 6371315.0  # Earth radius (m)

    # -----------------------------------------------------------------------
    # 1.  Interpolate lon/lat to staggered points (as in set_grid.m)
    # -----------------------------------------------------------------------
    # Use griddedInterpolant-style bilinear interpolation to PSI, U, V points.
    # RHO centres at integer+0.5; PSI at integer; U at (int, half); V at (half, int).

    Xr, Yr = np.meshgrid(np.arange(0.5, nx + 0.5, 1.0),   # xi indices for rho
                          np.arange(0.5, ny + 0.5, 1.0))   # eta indices for rho
    # PSI points: integer indices 1..nx-1, 1..ny-1
    Xp, Yp = np.meshgrid(np.arange(1.0, nx, 1.0),
                          np.arange(1.0, ny, 1.0))
    # U points: integer xi, half eta
    Xu, Yu = np.meshgrid(np.arange(1.0, nx, 1.0),
                          np.arange(0.5, ny + 0.5, 1.0))
    # V points: half xi, integer eta
    Xv, Yv = np.meshgrid(np.arange(0.5, nx + 0.5, 1.0),
                          np.arange(1.0, ny, 1.0))

    def _interp(Xsrc, Ysrc, Zsrc, X, Y):
        """Bilinear interpolation using RegularGridInterpolator."""
        x_vals = Xsrc[0, :].astype(float)
        y_vals = Ysrc[:, 0].astype(float)
        interp = RegularGridInterpolator((y_vals, x_vals), Zsrc,
                                         bounds_error=False, fill_value=None)
        pts = np.column_stack((Y.ravel(), X.ravel()))
        return interp(pts).reshape(X.shape)

    lon_psi = _interp(Xr, Yr, lon_rho, Xp, Yp)
    lat_psi = _interp(Xr, Yr, lat_rho, Xp, Yp)
    lon_u = _interp(Xr, Yr, lon_rho, Xu, Yu)
    lat_u = _interp(Xr, Yr, lat_rho, Xu, Yu)
    lon_v = _interp(Xr, Yr, lon_rho, Xv, Yv)
    lat_v = _interp(Xr, Yr, lat_rho, Xv, Yv)

    # -----------------------------------------------------------------------
    # 2.  Build extended PSI-point array (including edges) as in roms_metrics.m
    # -----------------------------------------------------------------------
    # lonp/latp are (ny+1, nx+1) arrays indexed from 0 at corners
    np_psi = nx - 1  # number of PSI points in xi
    mp_psi = ny - 1  # number of PSI points in eta

    lonp = np.full((ny + 1, nx + 1), np.nan)
    latp = np.full((ny + 1, nx + 1), np.nan)

    # Interior: lon_psi / lat_psi (1-indexed in MATLAB)
    lonp[1:ny, 1:nx] = lon_psi
    latp[1:ny, 1:nx] = lat_psi

    # Western edge (j=0 in python): lon_v(1, 1:Jm-1) at j=1 in matlab
    lonp[0, 1:nx] = lon_v[0, :-1]
    latp[0, 1:nx] = lat_v[0, :-1]

    # Eastern edge (j=end): lon_v(end, 1:Jm-1) at j=Mp
    lonp[-1, 1:nx] = lon_v[-1, :-1]
    latp[-1, 1:nx] = lat_v[-1, :-1]

    # Southern edge (i=0): lon_u(1:Im-1, 1) at i=1
    lonp[1:ny, 0] = lon_u[:-1, 0]
    latp[1:ny, 0] = lat_u[:-1, 0]

    # Northern edge (i=end): lon_u(1:Im-1, end) at i=Lp
    lonp[1:ny, -1] = lon_u[:-1, -1]
    latp[1:ny, -1] = lat_u[:-1, -1]

    # Corners
    lonp[0, 0] = lon_rho[0, 0]
    latp[0, 0] = lat_rho[0, 0]
    lonp[-1, 0] = lon_rho[-1, 0]
    latp[-1, 0] = lat_rho[-1, 0]
    lonp[0, -1] = lon_rho[0, -1]
    latp[0, -1] = lat_rho[0, -1]
    lonp[-1, -1] = lon_rho[-1, -1]
    latp[-1, -1] = lat_rho[-1, -1]

    # -----------------------------------------------------------------------
    # 3.  Compute pm, pn, angle using the Shchepetkin algorithm
    # -----------------------------------------------------------------------
    lonr = deg2rad * lon_rho
    latr = deg2rad * lat_rho
    lonp_rad = deg2rad * lonp
    latp_rad = deg2rad * latp

    pm = np.zeros((ny, nx))
    pn = np.zeros((ny, nx))
    ang = np.zeros((ny, nx))

    for j in range(nx):
        for i in range(ny):
            # dLon/dXi
            dLnX1 = lonp_rad[i + 1, j + 1] - lonp_rad[i, j + 1]
            if dLnX1 > np.pi:      dLnX1 -= 2 * np.pi
            elif dLnX1 < -np.pi:   dLnX1 += 2 * np.pi
            dLnX = lonp_rad[i + 1, j] - lonp_rad[i, j]
            if dLnX > np.pi:       dLnX -= 2 * np.pi
            elif dLnX < -np.pi:    dLnX += 2 * np.pi

            # dLon/dEta
            dLnY1 = lonp_rad[i + 1, j + 1] - lonp_rad[i + 1, j]
            if dLnY1 > np.pi:      dLnY1 -= 2 * np.pi
            elif dLnY1 < -np.pi:   dLnY1 += 2 * np.pi
            dLnY = lonp_rad[i, j + 1] - lonp_rad[i, j]
            if dLnY > np.pi:       dLnY -= 2 * np.pi
            elif dLnY < -np.pi:    dLnY += 2 * np.pi

            cff = 0.5 * np.cos(latr[i, j])

            a11 = cff * (dLnX + dLnX1)
            a12 = cff * (dLnY + dLnY1)
            a21 = 0.5 * (latp_rad[i + 1, j + 1] - latp_rad[i, j + 1] +
                         latp_rad[i + 1, j]     - latp_rad[i, j])
            a22 = 0.5 * (latp_rad[i, j + 1] + latp_rad[i + 1, j + 1] -
                         latp_rad[i, j]         - latp_rad[i + 1, j])

            with np.errstate(invalid='ignore'):
                denom_pm = np.sqrt(a11**2 + a21**2)
                denom_pn = np.sqrt(a12**2 + a22**2)
                pm[i, j] = 1.0 / (Eradius * denom_pm) if denom_pm > 0 else 0.0
                pn[i, j] = 1.0 / (Eradius * denom_pn) if denom_pn > 0 else 0.0

            # angle computation (arango roms_metrics.m)
            if a21 < -abs(a11):
                ang1 = -0.5 * np.pi - np.arctan(a11 / a21)
            elif a21 > abs(a11):
                ang1 = 0.5 * np.pi - np.arctan(a11 / a21)
            else:
                ang1 = np.arctan2(a21, a11)

            if a12 < -abs(a22):
                ang2 = 0.5 * np.pi + np.arctan(a22 / a12)
            elif a12 > abs(a22):
                ang2 = -0.5 * np.pi + np.arctan(a22 / a12)
            else:
                ang2 = -np.arctan2(a12, a22)

            ang[i, j] = 0.5 * (ang1 + ang2)

    # -----------------------------------------------------------------------
    # 3b. Check for degenerate edge cells and fix pm/pn
    # -----------------------------------------------------------------------
    for metric_arr in [pm, pn]:
        interior = metric_arr[2:ny-2, 2:nx-2]  # deep interior
        med = np.median(interior[~np.isinf(interior)])
        if med <= 0:
            continue
        bad = ~np.isfinite(metric_arr) | (metric_arr > 100 * med) | (metric_arr < med / 100)
        if bad.any():
            for i, j in zip(*np.where(bad)):
                vals = []
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0),
                               (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx and not bad[ni, nj]:
                        vals.append(metric_arr[ni, nj])
                metric_arr[i, j] = np.median(vals) if vals else med

    # Grid spacing
    dx = 1.0 / pm
    dy = 1.0 / pn

    # -----------------------------------------------------------------------
    # 4.  Compute Cartesian coordinates (x_rho, y_rho)
    # -----------------------------------------------------------------------
    x_rho = np.zeros_like(dx)
    for j in range(nx):
        x_rho[0, j] = -dx[0, j]
        for i in range(ny - 1):
            x_rho[i + 1, j] = x_rho[i, j] + dx[i + 1, j]
        x_rho[0, j]  += 0.5 * dx[0, j]
        x_rho[-1, j] -= 0.5 * dx[-1, j]

    y_rho = np.zeros_like(dy)
    for i in range(ny):
        y_rho[i, 0] = -dy[i, 0]
        for j in range(nx - 1):
            y_rho[i, j + 1] = y_rho[i, j] + dy[i, j + 1]
        y_rho[i, 0]  += 0.5 * dy[i, 0]
        y_rho[i, -1] -= 0.5 * dy[i, -1]

    x_psi = 0.25 * (x_rho[:-1, :-1] + x_rho[1:, :-1] +
                    x_rho[:-1, 1:]  + x_rho[1:, 1:])
    y_psi = 0.25 * (y_rho[:-1, :-1] + y_rho[1:, :-1] +
                    y_rho[:-1, 1:]  + y_rho[1:, 1:])
    x_u = 0.5 * (x_rho[:, :-1] + x_rho[:, 1:])
    y_u = 0.5 * (y_rho[:, :-1] + y_rho[:, 1:])
    x_v = 0.5 * (x_rho[:-1, :] + x_rho[1:, :])
    y_v = 0.5 * (y_rho[:-1, :] + y_rho[1:, :])

    # -----------------------------------------------------------------------
    # 5.  Compute dndx, dmde (inverse metric derivatives)
    # -----------------------------------------------------------------------
    L = ny - 1; Lm = L - 1
    M = nx - 1; Mm = M - 1

    dndx = np.zeros_like(lon_rho)
    dmde = np.zeros_like(lon_rho)

    dndx[1:L, 1:M] = 0.5 * (1.0 / pn[2:ny, 1:M] - 1.0 / pn[0:Lm, 1:M])
    dmde[1:L, 1:M] = 0.5 * (1.0 / pm[1:L, 2:nx] - 1.0 / pm[1:L, 0:Mm])

    # Fill edges
    dndx[0, :]  = dndx[1, :];    dndx[-1, :]  = dndx[-2, :]
    dndx[:, 0]  = dndx[:, 1];    dndx[:, -1]  = dndx[:, -2]
    dndx[0, 0]  = 0.5 * (dndx[0, 1] + dndx[1, 0])
    dndx[0, -1] = 0.5 * (dndx[0, -2] + dndx[1, -1])
    dndx[-1, 0] = 0.5 * (dndx[-1, 1] + dndx[-2, 0])
    dndx[-1, -1] = 0.5 * (dndx[-1, -2] + dndx[-2, -1])

    dmde[0, :]  = dmde[1, :];    dmde[-1, :]  = dmde[-2, :]
    dmde[:, 0]  = dmde[:, 1];    dmde[:, -1]  = dmde[:, -2]
    dmde[0, 0]  = 0.5 * (dmde[0, 1] + dmde[1, 0])
    dmde[0, -1] = 0.5 * (dmde[0, -2] + dmde[1, -1])
    dmde[-1, 0] = 0.5 * (dmde[-1, 1] + dmde[-2, 0])
    dmde[-1, -1] = 0.5 * (dmde[-1, -2] + dmde[-2, -1])

    # -----------------------------------------------------------------------
    # 6.  Coriolis parameter
    # -----------------------------------------------------------------------
    Omega = 7.2921e-5
    f = 2.0 * Omega * np.sin(latr)

    return {
        'lon_rho': lon_rho, 'lat_rho': lat_rho,
        'lon_u': lon_u, 'lat_u': lat_u,
        'lon_v': lon_v, 'lat_v': lat_v,
        'lon_psi': lon_psi, 'lat_psi': lat_psi,
        'x_rho': x_rho, 'y_rho': y_rho,
        'x_u': x_u, 'y_u': y_u,
        'x_v': x_v, 'y_v': y_v,
        'x_psi': x_psi, 'y_psi': y_psi,
        'pm': pm, 'pn': pn,
        'dndx': dndx, 'dmde': dmde,
        'angle_rho': ang,
        'f': f,
        'xl': x_psi.max() - x_psi.min(),
        'el': y_psi.max() - y_psi.min(),
        'spherical': 'T',
    }


def compute_metrics(lon_rho, lat_rho):
    """Alias for _lonlat_to_grid_metrics. Compute all ROMS grid metrics."""
    return _lonlat_to_grid_metrics(lon_rho, lat_rho)


# ---------------------------------------------------------------------------
# 3.  Mask computation
# ---------------------------------------------------------------------------

def compute_mask(mask_rho):
    """
    Compute U, V, and PSI masks from the RHO mask.

    Parameters
    ----------
    mask_rho : ndarray (eta_rho, xi_rho)
        1 = water, 0 = land.

    Returns
    -------
    mask_u, mask_v, mask_psi : ndarray
    """
    mask_rho = np.asarray(mask_rho, dtype=float)
    mask_u = mask_rho[:, :-1] * mask_rho[:, 1:]
    mask_v = mask_rho[:-1, :] * mask_rho[1:, :]
    mask_psi = (mask_rho[:-1, :-1] * mask_rho[:-1, 1:] *
                mask_rho[1:, :-1] * mask_rho[1:, 1:])
    return mask_u, mask_v, mask_psi


# ---------------------------------------------------------------------------
# 4.  Write NetCDF
# ---------------------------------------------------------------------------

def write_grid_file(grid_file, metrics, h=None, mask_rho=None,
                    vgrid_params=None):
    """
    Write a standard ROMS grid NetCDF file.

    Parameters
    ----------
    grid_file : str
        Output file path.
    metrics : dict
        Output from ``compute_metrics()``.
    h : ndarray or None
        Bathymetry at rho-points.
    mask_rho : ndarray or None
        Land/sea mask (1 = water, 0 = land).
    vgrid_params : dict or None
        Vertical grid parameters (optional, for storing in file).
    """
    import netCDF4 as nc4

    ny, nx = metrics['lon_rho'].shape

    ds = nc4.Dataset(grid_file, 'w', format='NETCDF3_64BIT')
    ds.Description = 'ROMS grid file'
    ds.Author = 'roms_prepro.grid.create'
    ds.Created = datetime.now().isoformat()
    ds.type = 'ROMS grid file'

    # Dimensions
    ds.createDimension('xi_rho', nx)
    ds.createDimension('xi_u', nx - 1)
    ds.createDimension('xi_v', nx)
    ds.createDimension('xi_psi', nx - 1)
    ds.createDimension('eta_rho', ny)
    ds.createDimension('eta_u', ny)
    ds.createDimension('eta_v', ny - 1)
    ds.createDimension('eta_psi', ny - 1)
    ds.createDimension('bath', None)

    def put_var(name, data, dims, **attrs):
        var = ds.createVariable(name, 'f8', dims)
        for k, v in attrs.items():
            setattr(var, k, v)
        var[:] = data

    # Spherical switch
    v = ds.createVariable('spherical', 'c')
    v[:] = metrics['spherical']

    # Grid coordinates
    put_var('lon_rho', metrics['lon_rho'], ('eta_rho', 'xi_rho'),
            long_name='longitude of RHO-points', units='degree_east')
    put_var('lat_rho', metrics['lat_rho'], ('eta_rho', 'xi_rho'),
            long_name='latitude of RHO-points', units='degree_north')
    put_var('lon_u', metrics['lon_u'], ('eta_u', 'xi_u'),
            long_name='longitude of U-points', units='degree_east')
    put_var('lat_u', metrics['lat_u'], ('eta_u', 'xi_u'),
            long_name='latitude of U-points', units='degree_north')
    put_var('lon_v', metrics['lon_v'], ('eta_v', 'xi_v'),
            long_name='longitude of V-points', units='degree_east')
    put_var('lat_v', metrics['lat_v'], ('eta_v', 'xi_v'),
            long_name='latitude of V-points', units='degree_north')
    put_var('lon_psi', metrics['lon_psi'], ('eta_psi', 'xi_psi'),
            long_name='longitude of PSI-points', units='degree_east')
    put_var('lat_psi', metrics['lat_psi'], ('eta_psi', 'xi_psi'),
            long_name='latitude of PSI-points', units='degree_north')

    # Metrics
    put_var('pm', metrics['pm'], ('eta_rho', 'xi_rho'),
            long_name='curvilinear coordinate metric in XI', units='meter-1')
    put_var('pn', metrics['pn'], ('eta_rho', 'xi_rho'),
            long_name='curvilinear coordinate metric in ETA', units='meter-1')
    put_var('angle', metrics['angle_rho'], ('eta_rho', 'xi_rho'),
            long_name='angle between XI-axis and EAST', units='radians')
    put_var('f', metrics['f'], ('eta_rho', 'xi_rho'),
            long_name='Coriolis parameter at RHO-points', units='second-1')

    if h is not None:
        put_var('h', h, ('eta_rho', 'xi_rho'),
                long_name='model bathymetry at RHO-points', units='meter')
        # Store raw bathymetry (pre-smoothing) if available
        put_var('hraw', h[None, :, :], ('bath', 'eta_rho', 'xi_rho'),
                long_name='working bathymetry at RHO-points', units='meter')

    if mask_rho is not None:
        mask_u, mask_v, mask_psi = compute_mask(mask_rho)
        put_var('mask_rho', mask_rho, ('eta_rho', 'xi_rho'),
                long_name='mask on RHO-points')
        put_var('mask_u', mask_u, ('eta_u', 'xi_u'),
                long_name='mask on U-points')
        put_var('mask_v', mask_v, ('eta_v', 'xi_v'),
                long_name='mask on V-points')
        put_var('mask_psi', mask_psi, ('eta_psi', 'xi_psi'),
                long_name='mask on PSI-points')

    # Domain extent
    for name in ('xl', 'el'):
        if name in metrics:
            v = ds.createVariable(name, 'f8', ())
            prefix = 'XI' if name == 'xl' else 'ETA'
            setattr(v, 'long_name', f'basin length in the {prefix}-direction')
            setattr(v, 'units', 'meter')
            v[:] = metrics[name]

    # Metric derivatives
    for name in ('dndx', 'dmde'):
        if name in metrics:
            put_var(name, metrics[name], ('eta_rho', 'xi_rho'),
                    long_name=f'{name} derivative of inverse metric',
                    units='meter')

    if vgrid_params is not None:
        for name in ('Vtransform', 'Vstretching'):
            if name in vgrid_params:
                v = ds.createVariable(name, 'i4', ())
                v[:] = vgrid_params[name]

    ds.close()


# ---------------------------------------------------------------------------
# 5.  High-level driver
# ---------------------------------------------------------------------------

def create_roms_grid(lon_corners, lat_corners, nx, ny, grid_file,
                     source_lon=None, source_lat=None, source_depth=None,
                     mask_rho=None, min_depth=5.0,
                     mask_method='depth', coastline_refine=False,
                     coastline_resolution='10m',
                     coastline_smooth=True,
                     smooth_method='rx0', rx0max=0.2,
                     vgrid_params=None):
    """
    Create a complete ROMS grid file from corner coordinates.

    Parameters
    ----------
    lon_corners : list of 4 floats
        [SW, NW, NE, SE] longitudes.
    lat_corners : list of 4 floats
        [SW, NW, NE, SE] latitudes.
    nx, ny : int
        Number of RHO-points.
    grid_file : str
        Output NetCDF file path.
    source_lon, source_lat, source_depth : ndarray or None
        Source bathymetry. If provided, mask is auto-generated from depth.
    mask_rho : ndarray or None
        Pre-made land/sea mask (overrides auto-generation).
    min_depth : float
        Minimum water depth (m).
    mask_method : str
        How to generate mask if auto: 'depth' (from source sign),
        'coastline' (GSHHG), or 'both'.
    coastline_refine : bool
        If True, refine the raw depth-based mask with GSHHG coastline.
    coastline_resolution : str
        GSHHG resolution: '10m', '50m', '110m'.
    smooth_method : str
        'rx0' or 'shapiro'.
    rx0max : float
        Target rx0 roughness.
    vgrid_params : dict or None
        Vertical grid parameters to store.

    Returns
    -------
    metrics : dict
    h : ndarray
    mask_rho : ndarray
    """
    from .bathymetry import (interp_bathymetry, fill_bathymetry,
                              smooth_bathymetry, roughness_report)
    from .mask import (mask_from_depth, mask_from_coastline,
                        mask_statistics, clean_mask, smooth_coastline)

    print("Generating grid coordinates from corners ...")
    lon_rho, lat_rho = make_grid_corner(lon_corners, lat_corners, nx, ny)

    print("Computing grid metrics ...")
    metrics = compute_metrics(lon_rho, lat_rho)

    # ----- Bathymetry -----
    if source_depth is not None and source_lon is not None and source_lat is not None:
        print("Interpolating bathymetry from source ...")
        h_sign = interp_bathymetry(lon_rho, lat_rho, source_lon, source_lat,
                                   source_depth)
        # Convert to ROMS convention: positive downward
        h = np.abs(h_sign)
    else:
        h = np.full(lon_rho.shape, min_depth, dtype=float)
        h_sign = None

    # ----- Mask -----
    if mask_rho is None:
        print(f"Generating land/sea mask (method={mask_method}) ...")
        if mask_method == 'depth' or mask_method == 'both':
            # Use the signed values for mask generation (GEBCO: neg=sea)
            mask_from = h_sign if h_sign is not None else -h
            mask_rho = mask_from_depth(mask_from)
        elif mask_method == 'coastline':
            mask_rho = mask_from_coastline(lon_rho, lat_rho,
                                           resolution=coastline_resolution)
        else:
            raise ValueError(f"Unknown mask_method: {mask_method}")

        if coastline_refine and mask_method != 'coastline':
            print("  Refining mask with GSHHG coastline ...")
            mask_coast = mask_from_coastline(lon_rho, lat_rho,
                                             resolution=coastline_resolution)
            mask_rho = mask_rho * mask_coast

        # ----- Clean mask -----
        print("  Cleaning mask (removing isolated features) ...")
        mask_rho = clean_mask(mask_rho, max_iter=5)

        # ----- Smooth coastline -----
        if coastline_smooth:
            print("  Smoothing coastline (morphological) ...")
            mask_rho = smooth_coastline(mask_rho, iterations=1)

        mask_statistics(mask_rho)

    # ----- Apply minimum depth and land fill -----
    h[(mask_rho == 1) & (h < min_depth)] = min_depth
    h[mask_rho == 0] = min_depth

    # ----- Smoothing -----
    print(f"Smoothing bathymetry (method={smooth_method}, rx0max={rx0max}) ...")
    roughness_report(h, mask=mask_rho, label="before smoothing")
    h = smooth_bathymetry(h, mask=mask_rho, method=smooth_method, rx0max=rx0max)
    roughness_report(h, mask=mask_rho, label="after smoothing")

    # ----- Write -----
    print(f"Writing grid file: {grid_file} ...")
    write_grid_file(grid_file, metrics, h=h, mask_rho=mask_rho,
                    vgrid_params=vgrid_params)
    print("Done.")

    return metrics, h, mask_rho