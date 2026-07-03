"""
Interpolate ROMS sigma-coordinate fields to standard z-levels.

Similar in spirit to pyroms ``roms2z`` / ``sta2z`` but uses the
``sigma_to_z`` routine from ``icbc._core`` and reads ROMS grid + vertical
coordinates via the ``grid`` module (no ``pyroms`` dependency).
"""

import numpy as np
import netCDF4 as nc
from ..ic._core import sigma_to_z
from ..grid.vgrid import set_depth, stretching


def sigma_to_z_levels(var, grid_file, z_levels, Cpos='rho',
                      Vtransform=2, Vstretching=4,
                      theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                      time_index=0, spval=1e37):
    """
    Interpolate a 3-D ROMS variable from sigma to fixed z-levels.

    Parameters
    ----------
    var : ndarray (N, eta, xi) or (eta, xi)
        Variable on ROMS sigma levels (N slices).  If 2-D it is broadcast.
    grid_file : str
        ROMS grid NetCDF file path.
    z_levels : ndarray (nz,)
        Target z-levels (negative downward, e.g. ``np.arange(-5, -5000, -5)``).
    Cpos : str
        Grid position: ``'rho'`` (default), ``'u'``, ``'v'``, or ``'w'``.
    Vtransform, Vstretching, theta_s, theta_b, Tcline : float
        ROMS vertical coordinate parameters.
    N : int
        Number of sigma levels (only used when grid_file has no s_rho).
    time_index : int
        Time index for 4-D variables (used if var has a leading time dim).
    spval : float
        Fill value for land / out-of-range.

    Returns
    -------
    var_z : ndarray (nz, eta, xi)
        Interpolated field on z-levels.
    """
    var = np.asarray(var, dtype=float)

    # Strip leading time dimension if present
    if var.ndim == 4:
        var = var[time_index]

    assert var.ndim == 3, f'var must be 3-D (N, eta, xi), got shape {var.shape}'
    N_in, eta, xi = var.shape

    # Read bathymetry and mask
    ds = nc.Dataset(grid_file)
    h = ds.variables['h'][:]
    mask = ds.variables.get('mask_rho', np.ones_like(h))
    if 'mask_rho' in ds.variables:
        mask = ds.variables['mask_rho'][:]
    else:
        mask = np.ones_like(h)
    lat_rho = ds.variables.get('lat_rho', np.zeros_like(h))
    ds.close()

    # Check if grid_file already has s-coordinate params
    # (we trust the caller's Vtransform etc. since grid may be raw)
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N_in,
                    h, igrid=1)

    # Handle C-position offsets
    if Cpos == 'rho':
        sigma_depth = z_r
    elif Cpos == 'w':
        z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N_in,
                        h, igrid=5)
        sigma_depth = z_w
    elif Cpos == 'u':
        z_r_u = 0.5 * (z_r[:, :, :-1] + z_r[:, :, 1:])
        sigma_depth = z_r_u
        mask = 0.5 * (mask[:, :-1] + mask[:, 1:])
    elif Cpos == 'v':
        z_r_v = 0.5 * (z_r[:, :-1, :] + z_r[:, 1:, :])
        sigma_depth = z_r_v
        mask = 0.5 * (mask[:-1, :] + mask[1:, :])
    else:
        raise ValueError(f"Unknown Cpos='{Cpos}'; use rho/u/v/w")

    var_z = sigma_to_z(var, sigma_depth, z_levels, fill_value=spval)

    # Apply land mask
    land = mask < 0.5
    if land.any():
        var_z[:, land] = spval

    return var_z