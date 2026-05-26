"""
Land/sea mask generation, cleaning, and coastline smoothing for ROMS grids.

Provides:
- mask_from_depth / mask_from_coastline
- clean_mask (remove isolated features, single-cell channels)
- smooth_coastline (morphological coastline smoothing)
- get_littoral (identify coastal boundary cells)
- compute_uvp_masks (standard ROMS staggered masks)
"""

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Mask from depth / source
# ---------------------------------------------------------------------------

def mask_from_depth(depth_val):
    """
    Create a raw land/sea mask from signed depth values.

    Parameters
    ----------
    depth_val : ndarray (eta_rho, xi_rho)
        Signed depth: typically negative = water, positive = land
        (GEBCO convention).

    Returns
    -------
    mask_rho : ndarray
        1 = water, 0 = land.
    """
    d = np.asarray(depth_val, dtype=float)
    if np.nanmin(d) >= 0:
        # ROMS convention: positive = water depth
        mask = (d > 0).astype(np.uint8)
    else:
        # GEBCO convention: negative = water
        mask = (d < 0).astype(np.uint8)
    return mask


def mask_from_coastline(lon_rho, lat_rho, resolution='10m'):
    """
    Create a mask using GSHHG coastline vectors via cartopy.

    Parameters
    ----------
    lon_rho, lat_rho : ndarray
    resolution : str
        '10m', '50m', '110m'.

    Returns
    -------
    mask_rho : ndarray
    """
    try:
        import cartopy.feature as cfeature
        import matplotlib.path as mpath
    except ImportError:
        raise ImportError("cartopy required for coastline masking")

    lon = np.asarray(lon_rho).ravel()
    lat = np.asarray(lat_rho).ravel()
    eta, xi = lon_rho.shape

    land = cfeature.GSHHGFeature(scale=resolution, levels=[1],
                                 facecolor='none')
    mask = np.ones(len(lon), dtype=np.uint8)

    for geom in list(land.geometries()):
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
            path = mpath.Path(coords)
            inside = path.contains_points(np.column_stack((lon, lat)))
            mask[inside] = 0

    return mask.reshape(eta, xi)


# ---------------------------------------------------------------------------
# Mask cleaning: remove isolated features
# ---------------------------------------------------------------------------

def _fill_small_holes(mask, max_hole_cells=16):
    """
    Fill small holes (0 surrounded by 1) in a binary mask.
    A "hole" is a connected component of 0-cells fully surrounded by 1-cells.
    """
    inv = 1 - mask
    labels, n = ndimage.label(inv)
    for lab in range(1, n + 1):
        region = labels == lab
        if region.sum() <= max_hole_cells:
            mask[region] = 1
    return mask


def _remove_small_regions(mask, max_cells=16):
    """
    Remove small connected components of 1-cells (isolated water cells
    surrounded by land).
    """
    labels, n = ndimage.label(mask)
    for lab in range(1, n + 1):
        region = labels == lab
        if region.sum() <= max_cells:
            mask[region] = 0
    return mask


def _keep_largest_water_region(mask):
    """
    Keep only the single largest connected water component. All other
    water regions (isolated lakes) become land.
    """
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    main = np.argmax(sizes) + 1
    mask[labels != main] = 0
    return mask


def _remove_single_row_channels(mask):
    """
    Remove water cells that form single-cell-wide channels (peninsulas)
    by checking 4-neighbor connectivity: if a water cell has <= 1 water
    neighbor in the 4-connected sense, mark it as land.
    Repeat until no changes.
    """
    ny, nx = mask.shape
    changed = True
    while changed:
        changed = False
        # Count 4-connected water neighbors
        neighbours = np.zeros_like(mask, dtype=int)
        neighbours[:-1, :] += mask[1:, :]
        neighbours[1:, :] += mask[:-1, :]
        neighbours[:, :-1] += mask[:, 1:]
        neighbours[:, 1:] += mask[:, :-1]
        # Water cell with 0 or 1 neighbour → remove
        to_remove = (mask == 1) & (neighbours <= 1)
        if to_remove.any():
            mask[to_remove] = 0
            changed = True
    return mask


def clean_mask(mask_rho, max_iter=5):
    """
    Iteratively clean a mask: remove isolated water features,
    fill small land holes, keep only the largest water region,
    and remove single-cell channels.

    Parameters
    ----------
    mask_rho : ndarray (eta_rho, xi_rho)
        Raw 0/1 mask.
    max_iter : int
        Maximum cleaning iterations.

    Returns
    -------
    mask_rho : ndarray
        Cleaned mask.
    """
    mask = mask_rho.copy().astype(np.uint8)

    for it in range(max_iter):
        before = mask.copy()
        # Step 1: fill small land holes in water
        mask = _fill_small_holes(mask, max_hole_cells=9)
        # Step 2: remove very small water regions
        mask = _remove_small_regions(mask, max_cells=4)
        # Step 3: keep only the main water body
        mask = _keep_largest_water_region(mask)
        # Step 4: remove 1-cell-wide channels
        mask = _remove_single_row_channels(mask)
        if np.array_equal(mask, before):
            print(f"  mask cleaning: converged in {it + 1} iterations")
            break
    else:
        print(f"  mask cleaning: max_iter={max_iter} reached")

    return mask


# ---------------------------------------------------------------------------
# U, V, PSI mask computation (from roms uvp_masks.m)
# ---------------------------------------------------------------------------

def compute_uvp_masks(mask_rho):
    """
    Compute U-, V-, and PSI-point masks from the RHO mask using the
    ROMS standard algorithm (uvp_masks.m).

    Parameters
    ----------
    mask_rho : ndarray (eta_rho, xi_rho)
        1 = water, 0 = land.

    Returns
    -------
    mask_u, mask_v, mask_psi : ndarray
    """
    rmask = np.asarray(mask_rho, dtype=float)
    Lp, Mp = rmask.shape
    L = Lp - 1
    M = Mp - 1

    # U mask
    umask = rmask[1:, :Mp] * rmask[:L, :Mp]

    # V mask
    vmask = rmask[:Lp, 1:] * rmask[:Lp, :M]

    # PSI mask: >= 3 of 4 corner water cells → water, else 0
    #   special case: opposite corners land → check (2 in uvp_masks.m)
    pmask = np.zeros((L, M), dtype=float)
    for jr in range(1, Mp):
        for ir in range(1, Lp):
            ip, jp = ir - 1, jr - 1
            a = rmask[ir - 1, jr]
            b = rmask[ir, jr]
            c = rmask[ir - 1, jr - 1]
            d = rmask[ir, jr - 1]
            water_count = (a > 0.5).astype(int) + (b > 0.5).astype(int) + \
                          (c > 0.5).astype(int) + (d > 0.5).astype(int)
            if water_count >= 3:
                pmask[ip, jp] = 1
            elif water_count == 2:
                # Opposite corners both water → water (diagonal channel)
                if ((a > 0.5) and (d > 0.5)) or ((b > 0.5) and (c > 0.5)):
                    pmask[ip, jp] = 1
            # else: 0

    return umask, vmask, pmask


# ---------------------------------------------------------------------------
# Refine mask with depth
# ---------------------------------------------------------------------------

def refine_mask(mask_rho, h, min_depth=5.0):
    """
    Refine mask: force shallow cells to water, fill land depth.

    Parameters
    ----------
    mask_rho : ndarray
    h : ndarray (positive downward).
    min_depth : float

    Returns
    -------
    mask_rho, h : refined arrays
    """
    mask = mask_rho.copy()
    h = h.copy()
    mask[(h > 0) & (h < min_depth)] = 1
    h[(mask == 1) & (h < min_depth)] = min_depth
    h[mask == 0] = min_depth
    return mask, h


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mask_statistics(mask_rho):
    """Print land/sea mask summary."""
    total = mask_rho.size
    water = int(mask_rho.sum())
    land = total - water
    print(f"  mask: total={total}, water={water} ({100.*water/total:.1f}%), "
          f"land={land} ({100.*land/total:.1f}%)")


# ---------------------------------------------------------------------------
# Coastline smoothing (morphological)
# ---------------------------------------------------------------------------

def smooth_coastline(mask_rho, iterations=1, structure=None):
    """
    Smooth jagged coastline edges using binary morphological closing/opening.

    Applies a closing (dilation → erosion) followed by an opening
    (erosion → dilation) to smooth stair-step artifacts along the land/sea
    boundary.

    Parameters
    ----------
    mask_rho : ndarray (eta_rho, xi_rho)
        1 = water, 0 = land.
    iterations : int
        Number of smoothing passes. More = smoother but may lose small
        coastal features.
    structure : ndarray or None
        Structuring element for morphology. Default is 8-connected (3x3).

    Returns
    -------
    mask : ndarray
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)

    mask = mask_rho.astype(bool)
    for _ in range(iterations):
        # Closing: dilate then erode (fills small holes, smooths convex corners)
        mask = ndimage.binary_closing(mask, structure=structure)
        # Opening: erode then dilate (removes small protrusions)
        mask = ndimage.binary_opening(mask, structure=structure)

    return mask.astype(np.uint8)


def get_littoral(mask_rho):
    """
    Identify coastal boundary cells: water cells that have at least one
    land neighbour (8-connected).

    Parameters
    ----------
    mask_rho : ndarray (eta_rho, xi_rho)
        1 = water, 0 = land.

    Returns
    -------
    eta_idx, xi_idx : ndarray
        Indices of coastal water cells.
    """
    mask = mask_rho.astype(bool)
    ny, nx = mask.shape

    # Dilate land by 1 cell in all 8 directions
    land = ~mask
    land_edges = ndimage.binary_dilation(land, iterations=1)
    # Coastal water cells = water cells adjacent to land edges
    littoral = mask & land_edges

    return np.where(littoral)