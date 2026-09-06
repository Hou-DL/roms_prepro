"""
Bathymetry interpolation, smoothing, and quality control for ROMS grids.

Provides:
- Interpolation from global datasets (GEBCO, ETOPO) to a ROMS grid
- Shapiro and Laplacian-based smoothing
- Beckmann & Haidvogel (rx0) and Haney (rx1) roughness metrics
- rx0-constrained iterative smoothing (Mellor-Ezer-Oey style)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interp_bathymetry(lon_rho, lat_rho, source_lon, source_lat, source_depth):
    """
    Interpolate bathymetry from a regular-grid dataset to ROMS rho-points.

    Parameters
    ----------
    lon_rho, lat_rho : ndarray (eta_rho, xi_rho)
        Target ROMS grid longitudes / latitudes.
    source_lon : ndarray (nx,)
        Source longitude array (monotonic).
    source_lat : ndarray (ny,)
        Source latitude array (monotonic).
    source_depth : ndarray (nx, ny) or (ny, nx)
        Source bathymetry. Convention is preserved:
        for GEBCO: positive = land, negative = water.

    Returns
    -------
    h_sign : ndarray (eta_rho, xi_rho)
        Interpolated bathymetry, preserving source sign convention.
    """
    lon_rho = np.asarray(lon_rho)
    lat_rho = np.asarray(lat_rho)

    # Determine source array orientation
    if source_depth.shape[0] == len(source_lat):
        # (lat, lon) ordering — common in NetCDF
        src = source_depth.T.astype(float)  # -> (lon, lat)
    else:
        src = source_depth.astype(float)

    # Ensure source coords are sorted ascending
    lon = np.asarray(source_lon, dtype=float)
    lat = np.asarray(source_lat, dtype=float)
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        src = src[::-1, :]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        src = src[:, ::-1]

    interpolator = RegularGridInterpolator((lon, lat), src,
                                           bounds_error=False,
                                           fill_value=np.nan)

    pts = np.column_stack((lon_rho.ravel(), lat_rho.ravel()))
    h_sign = interpolator(pts).reshape(lon_rho.shape)

    # Fill any NaN with nearest neighbour
    bad = np.isnan(h_sign)
    if bad.any():
        good = pts[~bad.ravel()]
        good_vals = h_sign.ravel()[~bad.ravel()]
        if len(good_vals) > 0:
            tree = cKDTree(good)
            _, idx = tree.query(pts[bad.ravel()])
            h_sign[bad] = good_vals[idx]

    return h_sign


def fill_bathymetry(h, mask, min_depth=5.0, fill_value=None):
    """
    Enforce minimum depth and fill land cells with a default value.

    Parameters
    ----------
    h : ndarray
        Bathymetry at rho-points (positive downward).
    mask : ndarray
        Land/sea mask (1 = water, 0 = land).
    min_depth : float
        Minimum water depth (m).
    fill_value : float or None
        Value for land cells. If None, uses min_depth / 2.

    Returns
    -------
    h_filled : ndarray
    """
    h = h.copy()
    if fill_value is None:
        fill_value = min_depth / 2.0
    h[mask == 0] = fill_value
    h[(h < min_depth) & (mask == 1)] = min_depth
    return h


# ---------------------------------------------------------------------------
# Shapiro filter (1D and 2D)
# ---------------------------------------------------------------------------

def _shapiro1(Finp, order, scheme=1):
    """
    1-D Shapiro filter (4th-order by default).

    Parameters
    ----------
    Finp : ndarray (m,)
        Input field.
    order : int
        Filter order (2, 4, 8, ...).
    scheme : int
        Boundary scheme (1 = no change at walls, constant order).

    Returns
    -------
    Fout : ndarray (m,)
    """
    fourk = np.array([
        2.500000e-1, 6.250000e-2, 1.562500e-2, 3.906250e-3,
        9.765625e-4, 2.44140625e-4, 6.103515625e-5, 1.5258789063e-5,
        3.814697e-6, 9.536743e-7, 2.384186e-7, 5.960464e-8,
        1.490116e-8, 3.725290e-9, 9.313226e-10, 2.328306e-10
    ])

    m = Finp.shape[0]
    order2 = order // 2
    F = Finp.copy()

    for _ in range(order2):
        cor = np.zeros(m)
        cor[0] = 2.0 * (F[0] - F[1])
        cor[-1] = 2.0 * (F[-1] - F[-2])
        cor[1:-1] = 2.0 * F[1:-1] - F[:-2] - F[2:]
        F -= cor * fourk[order2 - 1]

    return F


def shapiro_filter(Finp, order=4, scheme=1, napp=1):
    """
    2-D Shapiro filter applied row-wise then column-wise.

    Parameters
    ----------
    Finp : ndarray (m, n)
    order : int
    scheme : int
    napp : int
        Number of applications.

    Returns
    -------
    Fout : ndarray (m, n)
    """
    F = Finp.copy()
    for _ in range(napp):
        F = np.apply_along_axis(lambda x: _shapiro1(x, order, scheme), 0, F)
        F = np.apply_along_axis(lambda x: _shapiro1(x, order, scheme), 1, F)
    return F


# ---------------------------------------------------------------------------
# Roughness metrics (rx0 = Beckmann & Haidvogel, rx1 = Haney)
# ---------------------------------------------------------------------------

def rx0(h, mask=None):
    """
    Beckmann & Haidvogel (1993) stiffness ratio.

    For each water point, computes the maximum of
    ``|h_i - h_j| / (h_i + h_j)`` over all four neighbours.
    Values < 0.2 are generally safe.

    Parameters
    ----------
    h : ndarray (eta_rho, xi_rho)
        Bathymetry (positive downward).
    mask : ndarray or None
        1 = water, 0 = land.

    Returns
    -------
    r : ndarray (eta_rho, xi_rho)
    """
    h = np.asarray(h, dtype=float)
    if mask is None:
        mask = np.ones_like(h)

    r = np.zeros_like(h)
    # xi-direction neighbours
    hx = np.abs(h[:, 1:] - h[:, :-1]) / (h[:, 1:] + h[:, :-1])
    hx *= mask[:, 1:] * mask[:, :-1]   # only where both are water
    r[:, :-1] = np.maximum(r[:, :-1], hx)
    r[:, 1:] = np.maximum(r[:, 1:], hx)

    # eta-direction neighbours
    hy = np.abs(h[1:, :] - h[:-1, :]) / (h[1:, :] + h[:-1, :])
    hy *= mask[1:, :] * mask[:-1, :]
    r[:-1, :] = np.maximum(r[:-1, :], hy)
    r[1:, :] = np.maximum(r[1:, :], hy)

    return r


def rx1(z_w, mask=None):
    """
    Haney (1991) stiffness ratio for sigma-coordinate systems.

    For each water point, computes the maximum over all vertical levels
    of the interfacial slope ratio. Values < 0.3 are generally safe.

    Parameters
    ----------
    z_w : ndarray (N+1, eta_rho, xi_rho)
        Layer interface depths (negative, from set_depth with igrid=5).
    mask : ndarray or None

    Returns
    -------
    r : ndarray (eta_rho, xi_rho)
        Maximum rx1 across all vertical levels at each point.
    """
    z_w = np.asarray(z_w, dtype=float)
    Np, eta, xi = z_w.shape
    if mask is None:
        mask = np.ones((eta, xi))

    r = np.zeros((eta, xi))
    # Masks at edges
    um = mask[:, 1:] * mask[:, :-1]   # (eta, xi-1) at U-points
    vm = mask[1:, :] * mask[:-1, :]   # (eta-1, xi) at V-points

    for k in range(1, Np):
        # xi-direction slope ratio at U-points  (eta, xi-1)
        # r = |dz_i - dz_{i-1}| / |dz_i + dz_{i-1}|
        # dz_i = z_w[k,i] - z_w[k-1,i]
        dz_i = z_w[k, :, 1:] - z_w[k - 1, :, 1:]     # (eta, xi-1)
        dz_im1 = z_w[k, :, :-1] - z_w[k - 1, :, :-1]  # (eta, xi-1)
        num = np.abs(dz_i - dz_im1)
        den = np.abs(dz_i + dz_im1)
        with np.errstate(divide='ignore', invalid='ignore'):
            zx = np.where(den > 1e-10, num / den, 0.0) * um

        # eta-direction slope ratio at V-points  (eta-1, xi)
        dz_j = z_w[k, 1:, :] - z_w[k - 1, 1:, :]     # (eta-1, xi)
        dz_jm1 = z_w[k, :-1, :] - z_w[k - 1, :-1, :]  # (eta-1, xi)
        num = np.abs(dz_j - dz_jm1)
        den = np.abs(dz_j + dz_jm1)
        with np.errstate(divide='ignore', invalid='ignore'):
            zy = np.where(den > 1e-10, num / den, 0.0) * vm

        # Spread U-edge values to the two adjacent rho-points
        for j in range(eta):
            for i in range(xi):
                vals = []
                if i > 0 and um[j, i - 1] > 0:
                    vals.append(zx[j, i - 1])
                if i < xi - 1 and um[j, i] > 0:
                    vals.append(zx[j, i])
                if j > 0 and vm[j - 1, i] > 0:
                    vals.append(zy[j - 1, i])
                if j < eta - 1 and vm[j, i] > 0:
                    vals.append(zy[j, i])
                r[j, i] = max(r[j, i], max(vals) if vals else 0.0)

    return r


def roughness_report(h, z_w=None, mask=None, label=""):
    """
    Print a summary of grid roughness metrics.

    Parameters
    ----------
    h : ndarray
    z_w : ndarray or None
    mask : ndarray or None
    label : str
    """
    r0 = rx0(h, mask)
    print(f"  [{label}] rx0: min={r0.min():.4f}, max={r0.max():.4f}, "
          f"mean={r0.mean():.4f}, median={np.median(r0):.4f}")

    if z_w is not None:
        r1 = rx1(z_w, mask)
        print(f"  [{label}] rx1: min={r1.min():.4f}, max={r1.max():.4f}, "
              f"mean={r1.mean():.4f}, median={np.median(r1):.4f}")


# ---------------------------------------------------------------------------
# rfactor (PSI-point based, matching ROMS rfactor.m)
# ---------------------------------------------------------------------------

def rfactor(h, mask=None):
    """
    Compute the r-factor at PSI-points exactly matching ROMS rfactor.m.

    Parameters
    ----------
    h : ndarray (eta_rho, xi_rho)
    mask : ndarray or None

    Returns
    -------
    r : ndarray (eta_psi, xi_psi)
    """
    h = np.asarray(h, dtype=float)
    Lp, Mp = h.shape
    L = Lp - 1
    M = Mp - 1

    if mask is None:
        mask = np.ones_like(h)

    # U-point mask
    umask = np.zeros((L, Mp))
    for j in range(Mp):
        for i in range(1, Lp):
            umask[i - 1, j] = mask[i, j] * mask[i - 1, j]

    # V-point mask
    vmask = np.zeros((Lp, M))
    for j in range(1, Mp):
        for i in range(Lp):
            vmask[i, j - 1] = mask[i, j] * mask[i, j - 1]

    # Differences at U and V points
    hx = np.abs(h[1:Lp, :Mp] - h[:L, :Mp]) / (h[1:Lp, :Mp] + h[:L, :Mp])
    hy = np.abs(h[:Lp, 1:Mp] - h[:Lp, :M]) / (h[:Lp, 1:Mp] + h[:Lp, :M])

    hx *= umask
    hy *= vmask

    # Combine at PSI points
    r = np.maximum(
        np.maximum(hx[:L, :M], hx[:L, 1:Mp]),
        np.maximum(hy[:L, :M], hy[1:Lp, :M])
    )

    return r


# ---------------------------------------------------------------------------
# rx0-constrained smoothing (RHO-point granularity, Shapiro + r-factor)
# ---------------------------------------------------------------------------

def smooth_bathymetry_rx0(h, mask, rx0max=0.2, max_iter=50, order=4,
                          npass=5):
    """
    Smooth bathymetry using iterative Shapiro + local r-factor patching.

    Inspired by the ROMS standard approach (smooth_bath.m) but adapted
    for direct use without pre-smoothing. Uses RHO-point rx0 for finer
    granularity and stronger Shapiro passes.

    Parameters
    ----------
    h : ndarray (eta_rho, xi_rho)
    mask : ndarray
    rx0max : float
    max_iter : int
    order : int
        Shapiro filter order (default 4 for stronger smoothing).
    npass : int
        Shapiro passes per iteration (default 5).

    Returns
    -------
    hout : ndarray
    """
    hout = h.copy()

    for n in range(max_iter):
        # Current roughness
        r = rx0(hout, mask)
        rmax = r.max()

        if rmax <= rx0max:
            print(f"  rx0 smoothing converged in {n} iterations "
                  f"(rx0max={rx0max}, r={rmax:.4f})")
            break

        # Apply Shapiro to get candidate smoothed field
        hsmth = shapiro_filter(hout, order=order, napp=npass)

        # Patch cells where current roughness exceeds the limit
        modified = 0
        eta, xi = hout.shape
        for i in range(eta):
            for j in range(xi):
                if r[i, j] > rx0max:
                    hout[i, j] = hsmth[i, j]
                    modified += 1
                    # Also damp adjacent cells along the steepest direction
                    for di, dj in [(1, 0), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < eta and 0 <= nj < xi and r[ni, nj] > rx0max:
                            hout[ni, nj] = hsmth[ni, nj]

        if modified == 0:
            print(f"  rx0 smoothing: no more cells to modify, "
                  f"rmax={rmax:.4f}")
            break
        elif n % 5 == 0:
            print(f"  iteration {n}: rmax={rmax:.4f}, "
                  f"modified={modified}")
    else:
        r = rx0(hout, mask)
        print(f"  rx0 smoothing: max_iter={max_iter} reached, "
              f"final rmax={r.max():.4f}")

    return hout


def smooth_bathymetry_shapiro(h, mask, order=4, napp=5, apply_over_land=False):
    """
    Simple Shapiro filter applied only to water points,
    with land points held fixed.

    Parameters
    ----------
    h : ndarray
    mask : ndarray
    order : int
    napp : int
    apply_over_land : bool
        If True, also filter land values (for filling purposes).

    Returns
    -------
    h_smooth : ndarray
    """
    if apply_over_land:
        return shapiro_filter(h, order=order, napp=napp)
    # Only smooth where water, keep land unchanged
    hw = h * mask
    hs = shapiro_filter(hw, order=order, napp=napp)
    # Put back land values
    hs[mask < 0.5] = h[mask < 0.5]
    return hs


def smooth_bathymetry(h, mask=None, method='rx0', rx0max=0.2, **kwargs):
    """
    Unified entry point for bathymetry smoothing.

    Parameters
    ----------
    h : ndarray
    mask : ndarray or None
    method : str
        'rx0' for Mellor-Ezer-Oey iterative constraint,
        'shapiro' for Shapiro filter.
    rx0max : float
        Target rx0 (only used when method='rx0').
    **kwargs : passed to the underlying smoother.

    Returns
    -------
    h_smooth : ndarray
    """
    if mask is None:
        mask = np.ones_like(h)
    if method == 'rx0':
        return smooth_bathymetry_rx0(h, mask, rx0max=rx0max, **kwargs)
    elif method == 'shapiro':
        return smooth_bathymetry_shapiro(h, mask, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'rx0' or 'shapiro'.")