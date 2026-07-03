"""
Example: Create ROMS IC from another ROMS run (nesting).

Uses intermediate standard-z levels for conservative vertical remapping.
Auto-detects source files in a directory.

Usage:
    python example_ic_roms2roms.py
"""

from roms_prepro.ic import roms_to_roms_ini

# ===================================================================
# Configuration
# ===================================================================

# Source (parent) and target (child) grid files
SRC_GRID = 'parent_grid.nc'
DST_GRID = 'my_grid.nc'

# Source history file directory (auto-detects *.nc files)
SRC_HIST_DIR = './parent_output/'

# Output
INI_FILE = 'my_ini_nested.nc'

# Target vertical grid
VTRANSFORM = 2
VSTRETCHING = 4
THETA_S = 7.0
THETA_B = 0.1
TCLINE = 20.0
N = 30

# Which date's IC to make
IC_DATE = '2025-01-15'

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    roms_to_roms_ini(
        src_grid_file=SRC_GRID,
        dst_grid_file=DST_GRID,
        ini_file=INI_FILE,
        src_hist_dir=SRC_HIST_DIR,
        Vtransform=VTRANSFORM,
        Vstretching=VSTRETCHING,
        theta_s=THETA_S,
        theta_b=THETA_B,
        Tcline=TCLINE,
        N=N,
        init_date=IC_DATE,
    )

    print('\nDone!')
