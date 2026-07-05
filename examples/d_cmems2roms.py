"""
Example: Create ROMS initial conditions from CMEMS data.

Reads from 5 separate CMEMS NetCDF files (zos, thetao, so, uo, vo).
Time step auto-detected from init_date (no manual time_index needed).

Usage:
    python d_cmems2roms.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roms_prepro.ic import cmems_to_roms_ini

# ===================================================================
# Configuration - MODIFY FOR YOUR APPLICATION
# ===================================================================

# ROMS grid file
GRDNAME = r"D:\testFile\windcompare\tropical_5km_grid_629.nc"

# Output IC file
ININAME = r"D:\testFile\windcompare\tropical_5km_ini20160903.nc"

# CMEMS data directory
DATA_DIR = r"E:\Ocean_data\mercator\CMEMS_TRO_2016.nc"

# CMEMS file paths
ZFILE = f'{DATA_DIR}/cmems_zos_202505.nc'
TFILE = f'{DATA_DIR}/cmems_thetao_202505.nc'
SFILE = f'{DATA_DIR}/cmems_so_202505.nc'
UFILE = f'{DATA_DIR}/cmems_uo_202505.nc'
VFILE = f'{DATA_DIR}/cmems_vo_202505.nc'

# CMEMS variable names
ZVAR, TVAR, SVAR, UVAR, VVAR = 'zos', 'thetao', 'so', 'uo', 'vo'

# ROMS parameters
VTRANSFORM = 2
VSTRETCHING = 3
THETA_S = 2.5
THETA_B = 1.0
TCLINE = 25.0
N = 30

# Which date's IC to make (time_index auto-detected)
IC_DATE = '2016-09-03'

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    result = cmems_to_roms_ini(
        roms_grid_file=GRDNAME,
        zeta_file=ZFILE,
        temp_file=TFILE,
        salt_file=SFILE,
        u_file=UFILE,
        v_file=VFILE,
        ini_file=ININAME,
        zeta_var=ZVAR,
        temp_var=TVAR,
        salt_var=SVAR,
        u_var=UVAR,
        v_var=VVAR,
        Vtransform=VTRANSFORM,
        Vstretching=VSTRETCHING,
        theta_s=THETA_S,
        theta_b=THETA_B,
        Tcline=TCLINE,
        N=N,
        init_date=IC_DATE,
    )

    print('\nDone!')
    print(f'Output: {ININAME}')
