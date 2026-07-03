"""
Interpolate ROMS sigma-coordinate station data to standard z-levels.

Analogous to pyroms ``sta2z``: station data has shape (N_sigma, N_sta),
no horizontal dimension — vertical interpolation per station only.
"""

import numpy as np
from ..ic._core import sigma_to_z


def station_to_z_levels(var_sigma, sigma_depth, z_levels, spval=1e37):
    """
    Interpolate station profile data from sigma to fixed z-levels.

    Parameters
    ----------
    var_sigma : ndarray (N_sigma, N_sta)
        Variable on ROMS sigma levels at each station.
    sigma_depth : ndarray (N_sigma, N_sta)
        Sigma-level depths at each station (negative downward).
    z_levels : ndarray (nz,)
        Target z-levels (negative downward).
    spval : float
        Fill value for out-of-range.

    Returns
    -------
    var_z : ndarray (nz, N_sta)
        Interpolated profiles on z-levels.
    """
    var_sigma = np.asarray(var_sigma, dtype=float)
    sigma_depth = np.asarray(sigma_depth, dtype=float)

    assert var_sigma.ndim == 2, 'var_sigma must be 2-D (N_sigma, N_sta)'
    N_sigma, N_sta = var_sigma.shape

    # Reshape to 3-D (ns, eta=1, xi=sta) to reuse sigma_to_z
    var_3d = var_sigma[:, np.newaxis, :]       # (N_sigma, 1, N_sta)
    depth_3d = sigma_depth[:, np.newaxis, :]   # (N_sigma, 1, N_sta)

    result = sigma_to_z(var_3d, depth_3d, z_levels, fill_value=spval)

    # Squeeze back to 2-D (nz, N_sta)
    return result[:, 0, :]