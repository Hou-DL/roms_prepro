"""
Example: Create ROMS IC from a single CMEMS/Mercator/HYCOM file.

Use when all variables (SSH, temp, salt, u, v) are in one file.

Usage:
    python example_ic_single_file.py
"""

from roms_prepro.ic import mercator_to_roms_ini

# ===================================================================
# Configuration
# ===================================================================

GRID_FILE = 'my_grid.nc'
SOURCE_FILE = 'cmems_glo_phy_20250101.nc'  # single file with all vars
INI_FILE = 'my_ini.nc'

IC_DATE = '2025-01-15'

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    result = mercator_to_roms_ini(
        roms_grid_file=GRID_FILE,
        source_file=SOURCE_FILE,
        ini_file=INI_FILE,
        Vtransform=2,
        Vstretching=4,
        theta_s=7.0,
        theta_b=0.1,
        Tcline=20.0,
        N=30,
        init_date=IC_DATE,
    )

    print('\nDone!')
