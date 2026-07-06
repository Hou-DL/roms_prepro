"""
Interpolate ROMS sigma-coordinate fields to standard depth levels.

Provides both a Python API and a CLI for converting ROMS output files
from sigma coordinates to fixed z-levels.

Standard depth levels (default):
    0, 5, 10, 15, ..., 100, 125, 150, ..., 500, 550, ..., 5000, 5500 m

Usage (Python):
    from roms_prepro.remapping import roms_to_std_levels
    roms_to_std_levels('ocean_avg_0003.nc', 'output.nc')

Usage (CLI):
    python -m roms_prepro.remapping.roms2z_levels -i ocean_avg_0003.nc -o output.nc
    python -m roms_prepro.remapping.roms2z_levels -i ocean_avg_*.nc -d ./output/ -j 8
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    from ..grid.vgrid import set_depth, stretching
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from grid.vgrid import set_depth, stretching


# Default standard depth levels (positive, meters)
DEFAULT_STD_DEPTHS = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 900, 1000, 1250, 1500, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500,
    5000, 5500,
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


def process_file(input_file, output_file=None, std_depths=None, suffix='_z'):
    """
    Process a single ROMS NetCDF file: interpolate to standard z-levels.

    Parameters
    ----------
    input_file : str
        Input ROMS file path.
    output_file : str, optional
        Output file path. If None, adds suffix to input filename.
    std_depths : array-like, optional
        Custom standard depth levels. Uses DEFAULT_STD_DEPTHS if None.
    suffix : str
        Output filename suffix when output_file is not specified.

    Returns
    -------
    str or None
        Output file path on success, None on failure.
    """
    import netCDF4 as nc4

    if std_depths is None:
        std_depths = DEFAULT_STD_DEPTHS

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}{suffix}{ext}"

    print(f"  Processing: {os.path.basename(input_file)}")
    t0 = time.time()

    try:
        ds = nc4.Dataset(input_file, 'r')

        # Parse vertical parameters
        Vtransform, Vstretching, theta_s, theta_b, hc, N = _parse_vertical_params(ds)

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

        # Create output dataset
        ds_out = nc4.Dataset(output_file, 'w')

        # Dimensions
        ds_out.createDimension('z', len(std_depths_filtered))
        ds_out.createDimension('eta_rho', eta_rho)
        ds_out.createDimension('xi_rho', xi_rho)
        ds_out.createDimension('eta_u', eta_rho)
        ds_out.createDimension('xi_u', xi_rho - 1)
        ds_out.createDimension('eta_v', eta_rho - 1)
        ds_out.createDimension('xi_v', xi_rho)
        ds_out.createDimension('ocean_time', None)

        # Coordinate variables
        v = ds_out.createVariable('z', 'f8', ('z',))
        v[:] = std_depths_filtered
        v.long_name = 'depth'
        v.units = 'meter'
        v.positive = 'down'

        v = ds_out.createVariable('ocean_time', 'f8', ('ocean_time',))
        v[:] = time_vals
        v.units = time_units
        v.long_name = 'time since initialization'

        # Copy lon/lat
        for vname in ['lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v']:
            if vname in ds.variables:
                src_v = ds.variables[vname]
                dims = src_v.dimensions
                v_out = ds_out.createVariable(vname, 'f8', dims)
                v_out[:] = src_v[:]
                for attr in src_v.ncattrs():
                    setattr(v_out, attr, getattr(src_v, attr))

        # Variables to interpolate
        var_list = ['temp', 'salt', 'u', 'v', 'AKv', 'AKs', 'AKt']

        for var_name in var_list:
            if var_name not in ds.variables:
                continue

            src_v = ds.variables[var_name]
            var_data = src_v[:]

            # Determine grid position from dimensions
            dims = src_v.dimensions
            if 'xi_u' in dims:
                igrid = 3  # u-points
            elif 'eta_v' in dims:
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

            if igrid == 1:
                dims_out = ('ocean_time', 'z', 'eta_rho', 'xi_rho')
            elif igrid == 3:
                dims_out = ('ocean_time', 'z', 'eta_u', 'xi_u')
            elif igrid == 4:
                dims_out = ('ocean_time', 'z', 'eta_v', 'xi_v')
            else:
                dims_out = ('ocean_time', 'z', 'eta_rho', 'xi_rho')

            # Interpolated data
            var_out = roms_to_z_levels(var_data, z_var, std_depths_filtered, water)

            # Write to output
            if var_out.ndim == 3:
                var_out = var_out[np.newaxis, :]

            fill_val = getattr(src_v, '_FillValue', np.nan)
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

    if args.output and len(input_files) == 1:
        process_file(input_files[0], args.output, std_depths)
    elif len(input_files) == 1:
        process_file(input_files[0], std_depths=std_depths, suffix=args.suffix)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for f in input_files:
            base = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(args.output_dir, f"{base}{args.suffix}.nc")
            process_file(f, out, std_depths)
        print(f"\nDone! {len(input_files)} files processed.")

    return 0


if __name__ == '__main__':
    sys.exit(main())