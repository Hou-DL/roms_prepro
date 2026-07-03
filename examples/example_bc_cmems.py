"""
Example: Create ROMS boundary conditions from CMEMS data.

Reads monthly CMEMS files and creates time-dependent BC.
Supports single-file or directory auto-detect modes.

Usage:
    python example_bc_cmems.py
"""

import roms_prepro.bc.d_obc_cmems as cfg
from roms_prepro.bc.d_obc_cmems import main as cmems_bry_main

# ===================================================================
# Configuration
# ===================================================================

# Input/output files
GRID_FILE = 'my_grid.nc'
BRY_FILE = 'my_bry.nc'

# CMEMS data (single file or directory)
CMEMS_DIR = '/data/hdl/oceanfiles/CMEMS/2025/'

# Boundary switches [west, east, south, north]
BOUNDARY = [0, 1, 1, 0]

# Vertical grid
N_LEVELS = 30
VTRANSFORM = 2
VSTRETCHING = 3
THETA_S = 2.5
THETA_B = 1.0
TCLINE = 25.0

# Time range
TIME_START = '2025-05-01 00:00:00'
TIME_END = '2025-09-30 23:00:00'
ROMS_TIME_REF = '1990-01-01 00:00:00'

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    cfg.DATA_DIR = CMEMS_DIR
    cfg.GRD_NAME = GRID_FILE
    cfg.BRY_NAME = BRY_FILE
    cfg.BOUNDARY = BOUNDARY
    cfg.N_LEVELS = N_LEVELS
    cfg.VTRANSFORM = VTRANSFORM
    cfg.VSTRETCHING = VSTRETCHING
    cfg.THETA_S = THETA_S
    cfg.THETA_B = THETA_B
    cfg.TCLINE = TCLINE
    cfg.TIME_START = TIME_START
    cfg.TIME_END = TIME_END
    cfg.ROMS_TIME_REF = ROMS_TIME_REF

    cmems_bry_main()
