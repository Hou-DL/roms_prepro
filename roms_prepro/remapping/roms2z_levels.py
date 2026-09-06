"""
Interpolate ROMS sigma-coordinate fields to standard depth levels.

Provides both a Python API and a CLI for converting ROMS output files
from sigma coordinates to fixed z-levels.

Standard depth levels (default):
    0, 5, 10, 15, ..., 100, 125, 150, ..., 500, 550, ..., 5000, 5500 m

Usage (Python) — one-click: temp/salt/u/v interpolated to z, zeta
copied as a 2-D surface field, vertical parameters auto-detected:
    from roms_prepro.remapping import process_file
    process_file('ocean_avg_0003.nc', 'output.nc')

Keep native staggered grids instead of the default all-RHO output:
    process_file('ocean_avg_0003.nc', 'output_z.nc', to_rho=False)

One-click with explicit variables, depths and vertical overrides:
    process_file('ocean_avg_0003.nc', 'output_z.nc',
                 variables=['temp', 'salt'],
                 std_depths=[0, 10, 25, 50, 100, 200, 500, 1000],
                 vgrid_params={'Vtransform': 2, 'Vstretching': 4,
                               'theta_s': 7.0, 'theta_b': 0.1,
                               'Tcline': 20.0, 'N': 30})

Usage (CLI):
    python -m roms_prepro.remapping.roms2z_levels -i ocean_avg_0003.nc -o output.nc
    python -m roms_prepro.remapping.roms2z_levels -i ocean_avg_*.nc -d ./output/ -j 8
"""

import argparse
import glob
import os
import sys
import time
import numpy as np

try:
    from ..grid.vgrid import set_depth, stretching
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from grid.vgrid import set_depth, stretching


# Default standard depth levels (positive, meters) — 30 levels,
# surface-refined: every 10 m to 30 m, then widening spacing with depth
DEFAULT_STD_DEPTHS = [
    0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300,
    400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000,
    2500, 3000, 3500, 4000, 4500, 5000, 5500,
]


def _get_grid_shape(igrid, eta_rho, xi_rho):
    """Get the spatial shape for a given grid position."""
    if igrid == 3:  # U
        return (eta_rho, xi_rho - 1)
    elif igrid == 4:  # V
        return (eta_rho - 1, xi_rho)
    else:  # RHO, PSI, W
        return (eta_rho, xi_rho)


def _parse_vertical_params(ds):
    """
    Parse vertical coordinate parameters from a ROMS dataset.

    Returns Vtransform, Vstretching, theta_s, theta_b, hc, N
    """
    def _get_var(name, default=None):
        if name in ds.variables:
            val = ds.variables[name][:]
            return float(val) if np.ndim(val) == 0 else float(val.flat[0])
        if default is not None:
            return default
        raise KeyError(f"Cannot find '{name}' in dataset and no default given")

    Vtransform = int(_get_var('Vtransform', 2))
    Vstretching = int(_get_var('Vstretching', 4))
    theta_s = _get_var('theta_s', 7.0)
    theta_b = _get_var('theta_b', 0.1)
    hc = _get_var('hc', 25.0)

    # N from s_rho dimension
    if 's_rho' in ds.dimensions:
        N = len(ds.dimensions['s_rho'])
    else:
        N = 30

    return Vtransform, Vstretching, theta_s, theta_b, hc, N



def roms_to_z_levels(var_data, z_sigma, std_depths, valid_mask):
    """
    Interpolate a ROMS variable from sigma to standard z-levels.

    Parameters
    ----------
    var_data : ndarray (..., s_rho) or (..., eta, xi)
        Variable on sigma levels. Can be 3-D (time, s_rho, eta, xi) or
        2-D (s_rho, eta, xi).
    z_sigma : ndarray
        Sigma-level depths. Shape (s_rho, eta, xi) for single time,
        or (time, s_rho, eta, xi) for multiple times.
    std_depths : array-like
        Standard depth levels (positive, meters).
    valid_mask : ndarray (eta, xi)
        True for water points.

    Returns
    -------
    var_z : ndarray
        Interpolated variable on z-levels.
        Shape (..., nz, eta, xi) where nz = len(std_depths).
    """
    std_z = -np.array(std_depths, dtype=float)
    nz = len(std_depths)

    # Determine input dimensions
    has_time = var_data.ndim == 4
    if has_time:
        ntime, _, eta, xi = var_data.shape
        var_out = np.full((ntime, nz, eta, xi), np.nan)
    elif var_data.ndim == 3:
        ntime = 1
        _, eta, xi = var_data.shape
        var_out = np.full((nz, eta, xi), np.nan)
        var_data = var_data[np.newaxis, :]
        if z_sigma.ndim == 3:
            z_sigma = z_sigma[np.newaxis, :]
    else:
        raise ValueError(f"Expected 3-D or 4-D input, got {var_data.ndim}-D")

    z_profiles_is_4d = z_sigma.ndim == 4

    for t in range(ntime):
        zp = z_sigma[t] if z_profiles_is_4d else z_sigma

        for j in range(eta):
            for i in range(xi):
                if not valid_mask[j, i]:
                    continue

                val_prof = var_data[t, :, j, i]
                z_prof = z_sigma[t, :, j, i] if z_profiles_is_4d else z_sigma[:, j, i]

                if np.any(np.isnan(val_prof)):
                    continue

                result = _interp_to_z_1d(val_prof, z_prof, std_z)

                if has_time:
                    var_out[t, :, j, i] = result
                else:
                    var_out[:, j, i] = result

    return var_out


def _staggered_to_rho(var_data, axis, fill):
    """
    Reconstruct a u/v-point field on the RHO grid.

    Interior RHO points (n+1 of them for n staggered points) get the mean
    of their two flanking staggered values; if only one neighbour is valid
    (coast) that value is used, otherwise ``fill``.  The two outer RHO
    points are extrapolated from the nearest staggered value.  ``axis`` is
    the staggered axis (-1 for u/xi, -2 for v/eta).
    """
    n = var_data.shape[axis]

    def _take(idx):
        return np.ma.filled(np.take(var_data, idx, axis=axis), np.nan)

    a = _take(range(n - 1))     # staggered point left of each interior rho point
    b = _take(range(1, n))      # staggered point right of it
    av = np.abs(a) < 1e30
    bv = np.abs(b) < 1e30
    mid = np.full(a.shape, fill, dtype=float)
    both = av & bv
    mid[both] = 0.5 * (a + b)[both]
    mid[av & ~bv] = a[av & ~bv]
    mid[~av & bv] = b[~av & bv]

    def _edge(idx):
        e = _take(idx)
        return np.expand_dims(np.where(np.abs(e) < 1e30, e, fill), axis=axis)

    return np.concatenate([_edge(0), mid, _edge(n - 1)], axis=axis)


def process_file(input_file, output_file=None, std_depths=None, suffix='_z',
                 variables=None, vgrid_params=None, to_rho=True):
    """
    One-click conversion of a ROMS file from sigma to standard z-levels.

    Reads the vertical coordinate parameters from the input file
    (Vtransform/Vstretching/theta_s/theta_b/hc/N); every one of them can be
    overridden via ``vgrid_params``.

    Parameters
    ----------
    input_file : str
        Input ROMS file path (IC/BC/history/average — anything with h,
        optionally zeta and s_rho variables).
    output_file : str, optional
        Output file path. If None, adds suffix to input filename.
    std_depths : array-like, optional
        Target depth levels, positive meters downward (sign is normalized).
        Uses DEFAULT_STD_DEPTHS (30 levels) if None.  Levels deeper than
        the maximum bathymetry are dropped automatically.  The output
        vertical coordinate is named ``depth`` (positive down).
    suffix : str
        Output filename suffix when output_file is not specified.
    variables : list of str, optional
        Variables to process. None = ROMS-standard default set
        ['temp', 'salt', 'u', 'v', 'zeta'].  3-D fields (on s_rho/s_w)
        are interpolated to z; 2-D fields such as zeta are copied
        as-is without vertical interpolation.
    vgrid_params : dict, optional
        Vertical coordinate overrides, e.g.
        {'Vtransform': 2, 'Vstretching': 4, 'theta_s': 7.0, 'theta_b': 0.1,
         'Tcline': 20.0, 'N': 30}.  Keys set to None (or absent) keep the
        values auto-detected from the input file.
    to_rho : bool, optional
        True (default): put EVERYTHING on the RHO grid — u/v are averaged
        onto RHO points, the file carries a single horizontal grid with
        coordinates named ``lon``/``lat``, and no staggered dimensions are
        created (compact, easy to read).
        False: keep each variable on its native staggered grid
        (u on eta_u/xi_u, v on eta_v/xi_v) with lon_rho/lon_u/lon_v
        coordinate names.

    Returns
    -------
    str or None
        Output file path on success, None on failure.
    """
    import netCDF4 as nc4

    if std_depths is not None:
        std_depths = [abs(float(d)) for d in std_depths]
    else:
        std_depths = DEFAULT_STD_DEPTHS

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}{suffix}{ext}"

    print(f"  Processing: {os.path.basename(input_file)}")
    t0 = time.time()

    try:
        ds = nc4.Dataset(input_file, 'r')

        # Parse vertical parameters (file values + caller overrides)
        Vtransform, Vstretching, theta_s, theta_b, hc, N = _parse_vertical_params(ds)
        if vgrid_params:
            overrides = {k: v for k, v in vgrid_params.items() if v is not None}
            Vtransform = int(overrides.get('Vtransform', Vtransform))
            Vstretching = int(overrides.get('Vstretching', Vstretching))
            theta_s = float(overrides.get('theta_s', theta_s))
            theta_b = float(overrides.get('theta_b', theta_b))
            hc = float(overrides.get('Tcline', overrides.get('hc', hc)))
            N = int(overrides.get('N', N))
            print(f"    vgrid overrides: {overrides}")

        # Read bathymetry and SSH
        h = ds.variables['h'][:]
        eta_rho, xi_rho = h.shape

        # Time
        if 'ocean_time' in ds.variables:
            time_var = ds.variables['ocean_time']
            time_vals = time_var[:]
            time_units = getattr(time_var, 'units', '')
            ntime = len(time_vals)
        else:
            time_vals = np.array([0.0])
            time_units = ''
            ntime = 1

        # Filter depth levels by bathymetry
        h_max = np.nanmax(h)
        std_depths_filtered = [d for d in std_depths if d <= h_max]
        if not std_depths_filtered:
            print(f"    ERROR: no standard depths within bathymetry (hmax={h_max:.0f}m)")
            ds.close()
            return None

        # Land mask
        if 'mask_rho' in ds.variables:
            mask_rho = ds.variables['mask_rho'][:]
        else:
            mask_rho = np.ones((eta_rho, xi_rho))

        water = mask_rho > 0.5

        # Read zeta for all times
        if 'zeta' in ds.variables:
            zeta = ds.variables['zeta'][:]
            if zeta.ndim == 2:
                zeta = zeta[np.newaxis, :]
        else:
            zeta = np.zeros((ntime, eta_rho, xi_rho))

        # Helper: compute depths for one grid position at a time
        def _compute_depths(igrid, h_in, zeta_in):
            return set_depth(Vtransform, Vstretching, theta_s, theta_b, hc, N,
                             h_in, zeta=zeta_in, igrid=igrid)

        # Variables to write: caller-selected or ROMS-standard default set.
        # 3-D fields (s_rho/s_w) are interpolated to z; 2-D fields such as
        # zeta (surface elevation) are copied as-is without interpolation.
        if variables is None:
            var_list = ['temp', 'salt', 'u', 'v', 'zeta']
        else:
            var_list = list(dict.fromkeys(variables))  # de-dup, keep order
        missing = [v for v in var_list if v not in ds.variables]
        if missing:
            print(f"    WARNING: variables not in input file, skipped: {missing}")
        var_list = [v for v in var_list if v in ds.variables]
        if not var_list:
            print("    ERROR: none of the requested variables exist in the input file")
            ds.close()
            return None

        # Staggered dimensions are created only when a requested variable
        # stays on its native grid (to_rho=False); in the default to_rho
        # mode everything lives on the single RHO grid
        need_u = (not to_rho) and any('xi_u' in ds.variables[v_].dimensions for v_ in var_list)
        need_v = (not to_rho) and any('eta_v' in ds.variables[v_].dimensions for v_ in var_list)

        # Create output dataset
        ds_out = nc4.Dataset(output_file, 'w')

        # Dimensions
        ds_out.createDimension('depth', len(std_depths_filtered))
        ds_out.createDimension('eta_rho', eta_rho)
        ds_out.createDimension('xi_rho', xi_rho)
        if need_u:
            ds_out.createDimension('eta_u', eta_rho)
            ds_out.createDimension('xi_u', xi_rho - 1)
        if need_v:
            ds_out.createDimension('eta_v', eta_rho - 1)
            ds_out.createDimension('xi_v', xi_rho)
        ds_out.createDimension('ocean_time', None)

        # Coordinate variables
        v = ds_out.createVariable('depth', 'f8', ('depth',))
        v[:] = std_depths_filtered
        v.long_name = 'depth'
        v.units = 'meter'
        v.positive = 'down'

        v = ds_out.createVariable('ocean_time', 'f8', ('ocean_time',))
        v[:] = time_vals
        v.units = time_units
        v.long_name = 'time since initialization'

        # Horizontal coordinates: to_rho mode writes a single lon/lat pair;
        # native mode copies lon_rho/lon_u/lon_v under their ROMS names
        if to_rho:
            coord_names = [('lon_rho', 'lon'), ('lat_rho', 'lat')]
        else:
            coord_names = [('lon_rho', 'lon_rho'), ('lat_rho', 'lat_rho')]
            if need_u:
                coord_names += [('lon_u', 'lon_u'), ('lat_u', 'lat_u')]
            if need_v:
                coord_names += [('lon_v', 'lon_v'), ('lat_v', 'lat_v')]
        for src_name, out_name in coord_names:
            if src_name in ds.variables:
                src_v = ds.variables[src_name]
                dims = src_v.dimensions
                if to_rho:
                    dims = ('eta_rho', 'xi_rho')
                v_out = ds_out.createVariable(out_name, 'f8', dims)
                v_out[:] = src_v[:]
                for attr in src_v.ncattrs():
                    setattr(v_out, attr, getattr(src_v, attr))

        for var_name in var_list:
            src_v = ds.variables[var_name]
            var_data = src_v[:]
            fill_val = getattr(src_v, '_FillValue', np.nan)

            # Determine grid position from dimensions
            dims = src_v.dimensions
            has_vertical = ('s_rho' in dims) or ('s_w' in dims)
            on_u = 'xi_u' in dims
            on_v = 'eta_v' in dims

            # to_rho mode: move u/v fields onto the RHO grid first
            # (average of the two flanking staggered values, coastal
            #  one-sided neighbours taken as-is, fill kept as fill)
            if to_rho and (on_u or on_v):
                axis = -1 if on_u else -2
                var_data = _staggered_to_rho(var_data, axis, fill_val)
                on_u = on_v = False

            if has_vertical:
                if on_u:
                    igrid = 3  # u-points
                elif on_v:
                    igrid = 4  # v-points
                elif 's_w' in dims:
                    igrid = 5  # w-points
                else:
                    igrid = 1  # rho-points

                # Compute z for this grid position
                # set_depth handles h/zeta averaging internally for igrid 1-5
                z_var = np.zeros((ntime, N,) + _get_grid_shape(igrid, eta_rho, xi_rho))
                for t in range(ntime):
                    z_var[t] = _compute_depths(igrid, h, zeta[t])

                if igrid == 1 or igrid == 5 or to_rho:
                    dims_out = ('ocean_time', 'depth', 'eta_rho', 'xi_rho')
                elif igrid == 3:
                    dims_out = ('ocean_time', 'depth', 'eta_u', 'xi_u')
                else:
                    dims_out = ('ocean_time', 'depth', 'eta_v', 'xi_v')

                # Interpolated data
                var_out = roms_to_z_levels(var_data, z_var, std_depths_filtered, water)

                # Write to output
                if var_out.ndim == 3:
                    var_out = var_out[np.newaxis, :]
            else:
                # 2-D surface field (e.g. zeta): copy without vertical interpolation
                var_out = var_data
                if to_rho:
                    dims_out = ('ocean_time', 'eta_rho', 'xi_rho') if 'ocean_time' in dims \
                        else ('eta_rho', 'xi_rho')
                else:
                    dims_out = dims

            v_out = ds_out.createVariable(var_name, 'f8', dims_out, fill_value=fill_val)
            v_out[:] = var_out
            for attr in src_v.ncattrs():
                if attr == '_FillValue':
                    continue
                setattr(v_out, attr, getattr(src_v, attr))

        # Global attributes
        ds_out.title = 'ROMS output interpolated to standard depth levels'
        ds_out.source_file = os.path.basename(input_file)
        ds_out.history = f'Interpolated at {time.ctime()}'

        ds.close()
        ds_out.close()

        elapsed = time.time() - t0
        print(f"    Done: {os.path.basename(output_file)} ({elapsed:.1f}s)")
        return output_file

    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def _interp_to_z_1d(val_profile, z_profile, std_z):
    """
    Interpolate a single vertical profile to standard z-levels.

    Returns the interpolated values at std_z (negative depths).
    NaN for out-of-range points.
    """
    valid = np.isfinite(val_profile) & np.isfinite(z_profile)
    if valid.sum() < 2:
        return np.full(len(std_z), np.nan)

    z_sorted = z_profile[valid]
    v_sorted = val_profile[valid]

    # Sort by depth (most negative first = deepest)
    order = np.argsort(z_sorted)
    z_sorted = z_sorted[order]
    v_sorted = v_sorted[order]

    from scipy.interpolate import interp1d
    # z_sorted: deepest(-) → shallowest(+); v_sorted: deepest → shallowest
    # Use top sigma level value for depths shallower than it (surface gap)
    f = interp1d(z_sorted, v_sorted, bounds_error=False,
                 fill_value=(np.nan, v_sorted[-1]))
    return f(std_z)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Interpolate ROMS sigma-coordinate fields to standard depth levels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input ROMS file(s), supports glob patterns')
    parser.add_argument('-o', '--output',
                        help='Output file path (single file only)')
    parser.add_argument('-d', '--output-dir', default='.',
                        help='Output directory for batch processing')
    parser.add_argument('--suffix', default='_z',
                        help='Output filename suffix (default: _z)')
    parser.add_argument('--depths', nargs='+', type=float,
                        help='Custom standard depth levels (meters, positive)')
    parser.add_argument('--vars', nargs='+',
                        help='Variables to process (default: temp salt u v zeta; '
                             'zeta is copied as a 2-D field)')
    parser.add_argument('--native-grid', action='store_true',
                        help='Keep variables on their native staggered grids '
                             '(default: everything on the RHO grid, coords lon/lat)')
    parser.add_argument('-j', '--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')

    args = parser.parse_args()

    std_depths = sorted(args.depths) if args.depths else DEFAULT_STD_DEPTHS

    # Collect input files
    input_files = []
    for f in args.input:
        if os.path.isfile(f):
            input_files.append(f)
        else:
            matched = sorted(glob.glob(f))
            input_files.extend(matched)

    if not input_files:
        print("ERROR: No input files found")
        return 1

    input_files = sorted(set(input_files))

    print(f"ROMS sigma -> z-levels interpolation")
    print(f"  Files: {len(input_files)}")
    print(f"  Depth levels: {len(std_depths)} ({std_depths[0]}-{std_depths[-1]}m)")

    to_rho = not args.native_grid
    if args.output and len(input_files) == 1:
        process_file(input_files[0], args.output, std_depths,
                     variables=args.vars, to_rho=to_rho)
    elif len(input_files) == 1:
        process_file(input_files[0], std_depths=std_depths, suffix=args.suffix,
                     variables=args.vars, to_rho=to_rho)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for f in input_files:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(args.output_dir, f"{base}{args.suffix}.nc")
            process_file(f, out, std_depths, variables=args.vars, to_rho=to_rho)
        print(f"\nDone! {len(input_files)} files processed.")

    return 0


if __name__ == '__main__':
    sys.exit(main())