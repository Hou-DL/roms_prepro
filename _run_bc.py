import sys,time;sys.path.insert(0,'.');t0=time.time()
from roms_prepro.icbc import mercator_to_roms_bry
mercator_to_roms_bry(
    roms_grid_file='yellow_sea_grid2.nc',
    source_files=['/data/hdl/oceanfiles/CMEMS/CMEMS_SCS_uvts2025.nc'],
    bry_file='yellow_sea_bry_2025_jan-feb.nc',
    Vtransform=2,Vstretching=4,theta_s=7.0,theta_b=0.1,Tcline=20.0,N=30,
    boundaries=(False,True,True,True),
    start_date='2025-01-01',end_date='2025-02-15',
    time_ref='seconds since 2000-01-01 00:00:00',
    # CMEMS 变量名
    u_var='uo', v_var='vo', temp_var='thetao', salt_var='so', zeta_var='zos',
)
print(f'BC done in {(time.time()-t0)/60:.1f} min')