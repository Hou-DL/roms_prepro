# -*- coding: utf-8 -*-
"""
roms_prepro — ROMS Pre-processing Toolkit
==========================================

Master workflow script.  Each feature is a self-contained function with
parameters clearly documented at the top.  Edit the parameters in each
function, then call them from ``main()``.

Usage
-----
::

    python main.py          # run all enabled steps
    python main.py grid     # run only grid creation
    python main.py ic       # run only IC creation
    python main.py bc       # run only BC creation
    python main.py forc     # run only ERA5 forcing
    python main.py tide     # run only TPXO tidal forcing
    python main.py river    # run only river forcing
    python main.py remap    # run only sigma→z remapping

Environment
-----------
Requires the ``roms_prepro`` conda environment::

    conda activate roms_prepro

Module structure
----------------
::

    roms_prepro/
      grid/          — Grid creation (make_grid.py, bathymetry.py, vgrid.py, mask.py)
      icbc/          — IC/BC from CMEMS/Mercator (cmems_icbc.py, roms2roms.py, _core.py)
      forcing/       — ERA5 atmospheric forcing (era5.py)
      tide/          — TPXO8 tidal forcing (make_tide.py)
      river/         — River forcing (make_river.py)
      remapping/     — sigma↔z coordinate interpolation (roms2z.py, sta2z.py)
      utils/         — Utility functions
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))


# ===========================================================================
# 1.  Grid creation
# ===========================================================================

def make_grid():
    """
    Create a ROMS grid file from corner coordinates + GEBCO bathymetry.

    Steps (all automatic):
      1.  Generate lon/lat from four corner points
      2.  Compute grid metrics (pm, pn, angle, f, dndx, dmde, x/el)
      3.  Interpolate GEBCO bathymetry to the grid
      4.  Generate and clean land/sea mask
      5.  Smooth coastline edges
      6.  Smooth bathymetry to satisfy rx0 roughness criterion
      7.  Write standard ROMS grid NetCDF
    """
    from roms_prepro.grid import create_roms_grid, set_depth, roughness_report
    import numpy as np, netCDF4 as nc4

    # ---------- edit these parameters ----------
    grid_file = 'my_grid.nc'              # output grid file

    # GEBCO data will be auto-discovered under D:\ROMS
    # If you need a specific file, set gebco_path explicitly below

    # Domain corners [SW, NW, NE, SE]
    lon_corners = [119.0, 119.0, 125.7, 125.7]
    lat_corners = [33.5, 37.1, 37.1, 33.5]
    nx, ny = 300, 200                     # number of RHO-points
    min_depth = 5.0                       # minimum water depth (m)

    # Smoothing
    rx0max = 0.2                          # target roughness
    # -------------------------------------------

    # Read GEBCO subset (handle encoding gracefully)
    gebco_path = None
    for root, dirs, files in os.walk(r'D:\ROMS'):
        for f in files:
            if 'gebco' in f.lower() and f.endswith('.nc'):
                gebco_path = os.path.join(root, f)
                break
        if gebco_path:
            break

    if gebco_path is None or not os.path.exists(gebco_path):
        print("WARNING: GEBCO file not found — generating flat bottom.")
        slon = slat = sdep = None
    else:
        # Copy to temp if path has encoding issues on Windows
        import shutil, tempfile
        gebco_temp = os.path.join(tempfile.gettempdir(),
                                  '_roms_prepro_gebco.nc')
        if not os.path.exists(gebco_temp):
            shutil.copy2(gebco_path, gebco_temp)
        ds = nc4.Dataset(gebco_temp)
        glon = ds.variables['lon'][:]
        glat = ds.variables['lat'][:]

        margin = 1.0
        i = np.where((glon >= lon_corners[0] - margin) &
                     (glon <= lon_corners[2] + margin))[0]
        j = np.where((glat >= lat_corners[0] - margin) &
                     (glat <= lat_corners[1] + margin))[0]
        slon = glon[i]; slat = glat[j]
        sdep = ds.variables['elevation'][j, :][:, i].T  # → (lon, lat)
        ds.close()
        print(f"GEBCO subset: {len(slon)}×{len(slat)} points")

    create_roms_grid(
        lon_corners=lon_corners, lat_corners=lat_corners,
        nx=nx, ny=ny, grid_file=grid_file,
        source_lon=slon, source_lat=slat, source_depth=sdep,
        min_depth=min_depth, rx0max=rx0max,
        vgrid_params={'Vtransform': 2, 'Vstretching': 4}
    )

    # Quick roughness check with a typical vertical grid
    g = nc4.Dataset(grid_file)
    h = g.variables['h'][:]
    mask = g.variables['mask_rho'][:]
    g.close()
    z_w = set_depth(2, 4, 7.0, 0.1, 20.0, 30, h, igrid=5)
    roughness_report(h, z_w, mask=mask, label='final')

    print(f"\nGrid written to: {os.path.abspath(grid_file)}")
    return grid_file


# ===========================================================================
# 2.  Initial conditions (Mercator/HYCOM/CMEMS → ROMS)
# ===========================================================================

def make_ic_mercator(grid_file=None):
    """
    Create ROMS initial conditions from a Mercator/HYCOM/CMEMS source file.

    Auto-detects variable names in the source.  Override with keyword args
    like ``temp_var='thetao'`` if needed.
    """
    from roms_prepro.icbc import mercator_to_roms_ini

    # ---------- edit these parameters ----------
    if grid_file is None:
        grid_file = 'my_grid.nc'          # ROMS grid file
    source_file = 'hycom_2020_001.nc'     # Mercator/HYCOM file
    ini_file = 'my_ini.nc'                # output IC file

    Vtransform, Vstretching = 2, 4
    theta_s, theta_b, Tcline = 7.0, 0.1, 20.0
    N = 30                                # vertical levels

    init_date = '2020-01-15'              # IC date (set to None for first step)
    time_ref = 'seconds since 2000-01-01 00:00:00'
    # -------------------------------------------

    mercator_to_roms_ini(
        roms_grid_file=grid_file,
        source_file=source_file,
        ini_file=ini_file,
        Vtransform=Vtransform, Vstretching=Vstretching,
        theta_s=theta_s, theta_b=theta_b, Tcline=Tcline, N=N,
        init_date=init_date, time_ref=time_ref,
        # Override auto-detected names if needed:
        # temp_var='thetao', salt_var='so', u_var='uo', v_var='vo',
    )


# ===========================================================================
# 3.  Boundary conditions (Mercator/HYCOM/CMEMS → ROMS)
# ===========================================================================

def make_bry_mercator(grid_file=None):
    """
    Create ROMS time-dependent boundary conditions from Mercator/HYCOM files.

    The source_files list will be filtered by start_date/end_date if given.
    """
    from roms_prepro.icbc import mercator_to_roms_bry
    from glob import glob

    # ---------- edit these parameters ----------
    if grid_file is None:
        grid_file = 'my_grid.nc'
    source_dir = './hycom_data/'          # directory with source files
    source_pattern = '*.nc'               # glob pattern for source files
    bry_file = 'my_bry.nc'                # output BC file

    Vtransform, Vstretching = 2, 4
    theta_s, theta_b, Tcline = 7.0, 0.1, 20.0
    N = 30
    boundaries = (True, True, True, True)  # [W, E, S, N]

    start_date = '2020-01-01'             # filter time range
    end_date = '2020-01-31'
    time_ref = 'seconds since 2000-01-01 00:00:00'
    # -------------------------------------------

    import glob as g
    source_files = sorted(g.glob(os.path.join(source_dir, source_pattern)))
    if not source_files:
        print(f"WARNING: no source files found in {source_dir}")
        return
    print(f"Found {len(source_files)} source files")

    mercator_to_roms_bry(
        roms_grid_file=grid_file,
        source_files=source_files,
        bry_file=bry_file,
        Vtransform=Vtransform, Vstretching=Vstretching,
        theta_s=theta_s, theta_b=theta_b, Tcline=Tcline, N=N,
        boundaries=boundaries,
        start_date=start_date, end_date=end_date, time_ref=time_ref,
    )


# ===========================================================================
# 4.  Initial conditions (ROMS → ROMS)
# ===========================================================================

def make_ic_roms2roms(dst_grid_file=None):
    """
    Create ROMS IC by remapping from another ROMS run.

    Uses intermediate standard-z levels for conservative vertical remapping.
    Automatically handles grid rotation angles.
    """
    from roms_prepro.icbc import roms_to_roms_ini

    # ---------- edit these parameters ----------
    if dst_grid_file is None:
        dst_grid_file = 'my_grid.nc'
    src_grid_file = 'parent_grid.nc'      # source ROMS grid
    src_hist_file = 'parent_hist.nc'      # source ROMS history file

    ini_file = 'my_ini_from_parent.nc'
    Vtransform, Vstretching = 2, 4
    theta_s, theta_b, Tcline = 7.0, 0.1, 20.0
    N = 30
    src_time = 0                          # time index in source file

    init_date = None                      # or '2020-01-15' to auto-match
    time_ref = None                       # or 'seconds since 2000-01-01 00:00:00'
    # -------------------------------------------

    roms_to_roms_ini(
        src_grid_file=src_grid_file,
        src_hist_file=src_hist_file,
        dst_grid_file=dst_grid_file,
        ini_file=ini_file,
        Vtransform=Vtransform, Vstretching=Vstretching,
        theta_s=theta_s, theta_b=theta_b, Tcline=Tcline, N=N,
        src_time=src_time, init_date=init_date, time_ref=time_ref,
        # Override variable names in source if needed:
        # var_mapping={'temp': 'temp_sur', 'u': 'u_eastward'},
    )


# ===========================================================================
# 5.  Boundary conditions (ROMS → ROMS)
# ===========================================================================

def make_bry_roms2roms(dst_grid_file=None):
    """
    Create ROMS time-dependent BC by remapping from another ROMS run.
    """
    from roms_prepro.icbc import roms_to_roms_bry
    from glob import glob

    # ---------- edit these parameters ----------
    if dst_grid_file is None:
        dst_grid_file = 'my_grid.nc'
    src_grid_file = 'parent_grid.nc'
    src_hist_dir = './parent_output/'     # directory with source history files
    src_hist_pattern = 'ocean_avg_*.nc'   # glob pattern

    bry_file = 'my_bry_from_parent.nc'
    Vtransform, Vstretching = 2, 4
    theta_s, theta_b, Tcline = 7.0, 0.1, 20.0
    N = 30
    boundaries = (True, True, True, True)

    start_date = '2020-01-01'
    end_date = '2020-01-31'
    time_ref = 'seconds since 2000-01-01 00:00:00'
    # -------------------------------------------

    import glob as g
    src_hist_files = sorted(g.glob(os.path.join(src_hist_dir,
                                                src_hist_pattern)))
    if not src_hist_files:
        print(f"WARNING: no history files found in {src_hist_dir}")
        return
    print(f"Found {len(src_hist_files)} source history files")

    roms_to_roms_bry(
        src_grid_file=src_grid_file,
        src_hist_files=src_hist_files,
        dst_grid_file=dst_grid_file,
        bry_file=bry_file,
        Vtransform=Vtransform, Vstretching=Vstretching,
        theta_s=theta_s, theta_b=theta_b, Tcline=Tcline, N=N,
        boundaries=boundaries,
        start_date=start_date, end_date=end_date, time_ref=time_ref,
    )


# ===========================================================================
# 6.  Atmospheric forcing (ERA5 → ROMS)
# ===========================================================================

def make_forcing_era5(grid_file=None):
    """
    Create ROMS bulk-flux forcing file from ERA5 data.

    Supports ERA5 single-level variables:
      u10, v10, t2m, msl, msdwlwrf, msnswrf, tp, d2m
    Relative humidity can come from a separate RH_*.nc file or be
    computed from d2m and t2m.
    """
    from roms_prepro.forcing import era5_to_roms_forcing
    from glob import glob

    # ---------- edit these parameters ----------
    if grid_file is None:
        grid_file = 'my_grid.nc'

    era5_dir = '/data/hdl/oceanfiles/era5/in/'
    era5_files = sorted(glob(os.path.join(era5_dir, 'ERA5_*.nc')))
    rh_files = sorted(glob(os.path.join(era5_dir, 'RH_*.nc')))

    forc_file = 'my_forc_era5.nc'

    start_date = '2024-12-27'
    end_date = '2025-02-20 23:00:00'
    time_ref = 'seconds since 2000-01-01 00:00:00'
    # -------------------------------------------

    if not era5_files:
        print(f"WARNING: no ERA5 files found in {era5_dir}")
        return
    print(f"Found {len(era5_files)} ERA5 files")

    era5_to_roms_forcing(
        roms_grid_file=grid_file,
        era5_files=era5_files,
        rh_files=rh_files,
        out_file=forc_file,
        start_date=start_date, end_date=end_date, time_ref=time_ref,
        get_lwrad=True,
        get_swrad=True,
        get_rain=True,
        get_Tair=True,
        get_Pair=True,
        get_Qair=True,
        get_Wind=True,
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    """
    Run enabled workflow steps.  Edit the list below to control which
    steps execute.
    """
    # ----- enable / disable steps -----
    steps = [
        # 'grid',              # 1. create ROMS grid
        # 'ic_mercator',       # 2. IC from mercator/HYCOM
        # 'bry_mercator',      # 3. BC from mercator/HYCOM
        # 'ic_roms2roms',      # 4. IC via ROMS → ROMS remapping
        # 'bry_roms2roms',     # 5. BC via ROMS → ROMS remapping
        # 'forcing_era5',      # 6. ERA5 atmospheric forcing
    ]
    # -----------------------------------

    if len(sys.argv) > 1:
        steps = [sys.argv[1]]

    grid_file = None

    for step in steps:
        print(f"\n{'=' * 60}")
        print(f"  STEP: {step}")
        print(f"{'=' * 60}")

        if step == 'grid':
            grid_file = make_grid()

        elif step == 'ic_mercator':
            make_ic_mercator(grid_file)

        elif step == 'bry_mercator':
            make_bry_mercator(grid_file)

        elif step == 'ic_roms2roms':
            make_ic_roms2roms(grid_file)

        elif step == 'bry_roms2roms':
            make_bry_roms2roms(grid_file)

        elif step == 'forcing_era5':
            make_forcing_era5(grid_file)

        else:
            print(f"Unknown step: {step}")
            print("Available: grid, ic_mercator, bry_mercator, "
                  "ic_roms2roms, bry_roms2roms, forcing_era5")


if __name__ == '__main__':
    main()