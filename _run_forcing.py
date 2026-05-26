import sys, time
sys.path.insert(0, '.')
from roms_prepro.forcing import era5_to_roms_forcing
from glob import glob

t0 = time.time()

ERA5_DIR = '/data/hdl/oceanfiles/era5/in'

era5_files = sorted(glob(f'{ERA5_DIR}/ERA5_201801*.nc'))
rh_files = sorted(glob(f'{ERA5_DIR}/RH_201801*.nc'))

era5_to_roms_forcing(
    roms_grid_file='yellow_sea_grid2.nc',
    era5_files=era5_files,
    rh_files=rh_files if rh_files else None,
    out_file='yellow_sea_forc_era5.nc',
    start_date='2018-01-01',
    end_date='2018-01-31 23:00:00',
    time_ref='seconds since 2000-01-01 00:00:00',
    get_lwrad=True,
    get_swrad=True,
    get_rain=True,
    get_Tair=True,
    get_Pair=True,
    get_Qair=True,
    get_Wind=True,
)
print(f'Done in {(time.time()-t0)/60:.1f} min')