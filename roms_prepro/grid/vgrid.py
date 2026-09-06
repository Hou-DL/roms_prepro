"""
Vertical (S-coordinate) grid for ROMS.

Provides stretching functions for all standard Vstretching types (1-5)
and depth calculation for Vtransform 1, 2, and 4.

References
----------
- Song and Haidvogel (1994) — Vtransform=1, Vstretching=1
- Shchepetkin (2005 / UCLA-ROMS) — Vstretching=2
- Geyer (2009) — Vstretching=3 (BBL refinement)
- Shchepetkin (2010 / UCLA-ROMS) — Vstretching=4
- Powell (2008) — Vstretching=5 (surface-focused)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def s_rho(N):
    """S-coordinate values at RHO-points (cell centres). (-1 < s < 0)"""
    return -1.0 + (np.arange(1, N + 1) - 0.5) / N


def s_w(N):
    """S-coordinate values at W-points (cell interfaces). (-1 <= s <= 0)"""
    return -1.0 + np.arange(N + 1) / N


# ---------------------------------------------------------------------------
# Stretching functions
# ---------------------------------------------------------------------------

def _Cs_1(s, theta_s, theta_b):
    """Vstretching=1 – Song & Haidvogel (1994)."""
    if theta_s > 0:
        Ptheta = np.sinh(theta_s * s) / np.sinh(theta_s)
        Rtheta = (np.tanh(theta_s * (s + 0.5)) /
                  (2.0 * np.tanh(0.5 * theta_s)) - 0.5)
        return (1.0 - theta_b) * Ptheta + theta_b * Rtheta
    return s.copy()


def _Cs_2(s, theta_s, theta_b):
    """Vstretching=2 – Shchepetkin (2005)."""
    if theta_s > 0:
        Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
        if theta_b > 0:
            Cbot = np.sinh(theta_b * (s + 1.0)) / np.sinh(theta_b) - 1.0
            weight = (s + 1.0) * (1.0 + (1.0 - (s + 1.0)))  # alfa=beta=1
            return weight * Csur + (1.0 - weight) * Cbot
        return Csur
    return s.copy()


def _Cs_3(s, theta_s, theta_b):
    """Vstretching=3 – Geyer BBL (2009)."""
    if theta_s > 0:
        exp_s = theta_s      # surface stretching exponent
        exp_b = theta_b      # bottom stretching exponent
        alpha = 3.0          # scale factor for all hyperbolic functions
        Cbot = np.log(np.cosh(alpha * (s + 1.0) ** exp_b)) / np.log(np.cosh(alpha)) - 1.0
        Csur = -np.log(np.cosh(alpha * np.abs(s) ** exp_s)) / np.log(np.cosh(alpha))
        weight = (1.0 - np.tanh(alpha * (s + 0.5))) / 2.0
        return weight * Cbot + (1.0 - weight) * Csur
    return s.copy()


def _Cs_4(s, theta_s, theta_b):
    """Vstretching=4 – Shchepetkin (2010)."""
    if theta_s > 0:
        Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
    else:
        Csur = -(s**2)
    if theta_b > 0:
        return (np.exp(theta_b * Csur) - 1.0) / (1.0 - np.exp(-theta_b))
    return Csur


def _Cs_5(s, theta_s, theta_b):
    """Vstretching=5 – Powell / surface-focused."""
    if theta_s > 0:
        Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
    else:
        Csur = -(s * s)
    if theta_b > 0:
        return (np.exp(theta_b * (Csur + 1.0)) - 1.0) / \
               (np.exp(theta_b) - 1.0) - 1.0
    return Csur


_STRETCHING = {1: _Cs_1, 2: _Cs_2, 3: _Cs_3, 4: _Cs_4, 5: _Cs_5}


def stretching(Vstretching, theta_s, theta_b, N, kgrid=0):
    """
    Compute S-coordinate independent variable s and stretching curve C(s).

    Parameters
    ----------
    Vstretching : int
        Vertical stretching function (1, 2, 3, 4, or 5).
        - 1: Song & Haidvogel (1994)
        - 2: Shchepetkin (2005)
        - 3: Geyer BBL (2009)
        - 4: Shchepetkin (2010)
        - 5: Powell / surface-focused
    theta_s : float
        Surface control parameter.
    theta_b : float
        Bottom control parameter.
    N : int
        Number of vertical levels.
    kgrid : int, optional
        0 for RHO-points (cell centres), 1 for W-points (interfaces).

    Returns
    -------
    s : ndarray, shape (N,) or (N+1,)
        S-coordinate values in [-1, 0].
    Cs : ndarray, shape (N,) or (N+1,)
        Stretching curve values.
    """
    s = s_w(N) if kgrid else s_rho(N)
    fn = _STRETCHING.get(Vstretching)
    if fn is None:
        raise ValueError(f"Vstretching={Vstretching} is not supported. "
                         f"Choose from {list(_STRETCHING)}")
    return s, fn(s, theta_s, theta_b)


# ---------------------------------------------------------------------------
# Depth computation (Vtransform = 1, 2, 4)
# ---------------------------------------------------------------------------

def _transform_1(h, hc, s, Cs, zeta):
    """Vtransform=1: original Song & Haidvogel (1994)."""
    z0 = hc * s + (h - hc) * Cs          # broadcast: (N,...) + (...)
    return z0 + zeta * (1.0 + z0 / h)


def _transform_2(h, hc, s, Cs, zeta):
    """Vtransform=2 (also 4): UCLA-ROMS, Shchepetkin (2005)."""
    z0 = (hc * s + h * Cs) / (hc + h)
    return zeta + (zeta + h) * z0


_TRANSFORM = {1: _transform_1, 2: _transform_2, 4: _transform_2}


def set_depth(Vtransform, Vstretching, theta_s, theta_b, hc, N,
              h, zeta=None, igrid=1):
    """
    Compute 3-D depths on a ROMS staggered grid.

    Parameters
    ----------
    Vtransform : int
        Vertical transformation equation (1, 2, or 4).
    Vstretching : int
        Vertical stretching function (1, 2, 4, or 5).
    theta_s, theta_b : float
        S-coordinate control parameters.
    hc : float
        Critical / stretching depth (m).
    N : int
        Number of vertical levels.
    h : ndarray (eta_rho, xi_rho)
        Bathymetry (positive downward, m).
    zeta : ndarray (eta_rho, xi_rho) or None
        Free-surface elevation (m). If None, assumed zero.
    igrid : int
        Grid point type:
        1 = RHO, 2 = PSI, 3 = U, 4 = V, 5 = W.

    Returns
    -------
    z : ndarray
        Depths (m, negative below mean sea level).
        Shape: (N, eta, xi) for igrid=1-4; (N+1, eta, xi) for igrid=5.
    """
    h = np.asarray(h, dtype=float)
    if zeta is None:
        zeta = np.zeros_like(h)
    else:
        zeta = np.asarray(zeta, dtype=float)

    # Determine averaging for staggered positions
    if igrid == 1:      # RHO
        h_g, zeta_g = h, zeta
    elif igrid == 2:    # PSI
        h_g = 0.25 * (h[:-1, :-1] + h[1:, :-1] + h[:-1, 1:] + h[1:, 1:])
        zeta_g = 0.25 * (zeta[:-1, :-1] + zeta[1:, :-1] +
                         zeta[:-1, 1:] + zeta[1:, 1:])
    elif igrid == 3:    # U (between rho in xi-direction)
        h_g = 0.5 * (h[:, :-1] + h[:, 1:])
        zeta_g = 0.5 * (zeta[:, :-1] + zeta[:, 1:])
    elif igrid == 4:    # V (between rho in eta-direction)
        h_g = 0.5 * (h[:-1, :] + h[1:, :])
        zeta_g = 0.5 * (zeta[:-1, :] + zeta[1:, :])
    elif igrid == 5:    # W (interfaces, same location as RHO)
        h_g, zeta_g = h, zeta
    else:
        raise ValueError(f"igrid={igrid} not supported (1-5).")

    # S-coordinate and stretching
    kgrid = 1 if igrid == 5 else 0
    s, Cs = stretching(Vstretching, theta_s, theta_b, N, kgrid=kgrid)

    # Reshape for broadcasting: (N_or_Np, 1, 1) over (eta, xi)
    s = s.reshape(-1, 1, 1)
    Cs = Cs.reshape(-1, 1, 1)

    fn = _TRANSFORM.get(Vtransform)
    if fn is None:
        raise ValueError(f"Vtransform={Vtransform} not supported "
                         f"(must be 1, 2, or 4).")

    z = fn(h_g, hc, s, Cs, zeta_g)

    # Ensure bottom of W-grid exactly matches -h
    if igrid == 5:
        z[0, ...] = -h_g

    return z