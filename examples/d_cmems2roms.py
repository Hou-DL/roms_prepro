"""
Example: Create ROMS initial conditions from CMEMS data.

This script replicates the MATLAB d_cmems2roms.m functionality,
reading from 5 separate CMEMS NetCDF files.
"""

import sys
sys.path.insert(0, r'Z:\hdl\roms_prepro')

from roms_prepro.ic import cmems_to_roms_ini

# ===================================================================
# Paths and file names - MODIFY FOR YOUR APPLICATION
# ===================================================================

# Data directory
Data_Dir = r'\\DS1825\q1\public_ocean_data\Mercator\2025'

# ROMS grid file
GRDname = r'E:\Ocean_data\ERA5\romsinput\output\NSCS_grd_operational_adjust.nc'

# Output initial condition file
INIname = 'roms_ini_20250501.nc'

# CMEMS file names (5 separate files)
Zfile = f'{Data_Dir}/cmems_zos_202505.nc'      # sea surface height
Tfile = f'{Data_Dir}/cmems_thetao_202505.nc'   # temperature
Sfile = f'{Data_Dir}/cmems_so_202505.nc'       # salinity
Ufile = f'{Data_Dir}/cmems_uo_202505.nc'       # u velocity
Vfile = f'{Data_Dir}/cmems_vo_202505.nc'       # v velocity

# CMEMS variable names
Zvar = 'zos'       # sea surface height
Tvar = 'thetao'    # temperature
Svar = 'so'        # salinity
Uvar = 'uo'        # u velocity
Vvar = 'vo'        # v velocity

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    result = cmems_to_roms_ini(
        roms_grid_file=GRDname,
        zeta_file=Zfile,
        temp_file=Tfile,
        salt_file=Sfile,
        u_file=Ufile,
        v_file=Vfile,
        ini_file=INIname,
        zeta_var=Zvar,
        temp_var=Tvar,
        salt_var=Svar,
        u_var=Uvar,
        v_var=Vvar,
        Vtransform=2,
        Vstretching=3,      # Geyer BBL
        theta_s=2.5,
        theta_b=1.0,
        Tcline=25.0,
        N=30,
        time_ref='1990-01-01',   # ROMS time reference
        init_date='2025-05-01',  # Initial condition date
        time_index=0             # First time step in CMEMS files
    )
    
    print('\nDone!')
