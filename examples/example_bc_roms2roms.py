"""
Example: Create ROMS boundary conditions from another ROMS run.

Uses intermediate standard-z levels for conservative vertical remapping.
Auto-detects source files in a directory.

Usage:
    python example_bc_roms2roms.py
"""

from roms_prepro.bc import roms_to_roms_bry

# ===================================================================
# Configuration
# ===================================================================

SRC_GRID = 'parent_grid.nc'
DST_GRID = 'my_grid.nc'
SRC_HIST_DIR = './parent_output/'

BRY_FILE = 'my_bry_nested.nc'

VTRANSFORM = 2
VSTRETCHING = 4
THETA_S = 7.0
THETA_B = 0.1
TCLINE = 20.0
N = 30

BOUNDARIES = [True, True, True, True]  # [W, E, S, N]

START_DATE = '2025-01-01'
END_DATE = '2025-01-31'

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    roms_to_roms_bry(
        src_grid_file=SRC_GRID,
        dst_grid_file=DST_GRID,
        bry_file=BRY_FILE,
        src_hist_dir=SRC_HIST_DIR,
        Vtransform=VTRANSFORM,
        Vstretching=VSTRETCHING,
        theta_s=THETA_S,
        theta_b=THETA_B,
        Tcline=TCLINE,
        N=N,
        boundaries=BOUNDARIES,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print('\nDone!')
