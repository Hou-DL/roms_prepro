"""
Example: Create ROMS initial conditions from CMEMS data (auto-detect mode).

This script demonstrates the simplified CMEMS IC workflow:
  - Auto-detect CMEMS files in a directory
  - Auto-detect time reference from file metadata
  - Specify IC date directly (no manual time_index needed)

Usage:
    python example_ic_cmems.py
"""

from roms_prepro.ic import cmems_to_roms_ini

# ===================================================================
# Configuration - MODIFY FOR YOUR APPLICATION
# ===================================================================

# ROMS grid file (must contain lon_rho, lat_rho, h, mask_rho, etc.)
GRID_FILE = 'my_grid.nc'

# Output IC file
INI_FILE = 'my_ini.nc'

# CMEMS data directory (auto-detects files)
CMEMS_DIR = '/data/hdl/oceanfiles/CMEMS/'

# ROMS vertical grid parameters
VTRANSFORM = 2
VSTRETCHING = 3
THETA_S = 2.5
THETA_B = 1.0
TCLINE = 25.0
N = 30

# Which date's IC to make
IC_DATE = '2025-05-01'

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    result = cmems_to_roms_ini(
        roms_grid_file=GRID_FILE,
        ini_file=INI_FILE,
        data_dir=CMEMS_DIR,
        Vtransform=VTRANSFORM,
        Vstretching=VSTRETCHING,
        theta_s=THETA_S,
        theta_b=THETA_B,
        Tcline=TCLINE,
        N=N,
        init_date=IC_DATE,
    )

    print('\nDone!')
    print(f'Output: {INI_FILE}')
    print(f'Fields: {list(result.keys())}')
