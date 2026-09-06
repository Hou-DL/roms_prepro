import cdsapi, os, sys
import netCDF4 as nc
import numpy as np

OUTPUT_DIR = '/data/hdl/oceanfiles/era5/in'
client = cdsapi.Client()

ALL_VARS = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'mean_sea_level_pressure',
    'mean_surface_downward_long_wave_radiation_flux',
    'mean_surface_net_long_wave_radiation_flux',
    'mean_surface_net_short_wave_radiation_flux',
    'total_precipitation',
]

DOWNLOAD_MONTHS = ['202412', '202501', '202502']
AREA = [38, 118, 32, 127]
ALL_DAYS = [f'{d:02d}' for d in range(1, 32)]
ALL_HOURS = [f'{h:02d}:00' for h in range(24)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for ym in DOWNLOAD_MONTHS:
    year, month = ym[:4], ym[4:]
    out_file = os.path.join(OUTPUT_DIR, f'ERA5_{ym}.nc')
    if os.path.exists(out_file):
        sz = os.path.getsize(out_file) / 1e6
        print(f'Skip existing: {out_file} ({sz:.0f}MB)')
        continue

    monthly_parts = []
    for vi, var in enumerate(ALL_VARS):
        part_file = os.path.join(OUTPUT_DIR, f'ERA5_{ym}_v{vi}.nc')
        if not os.path.exists(part_file):
            print(f'  [{vi+1}/{len(ALL_VARS)}] {var} {ym}...', flush=True)
            try:
                client.retrieve('reanalysis-era5-single-levels', {
                    'product_type': 'reanalysis',
                    'variable': var,
                    'year': year,
                    'month': month,
                    'day': ALL_DAYS,
                    'time': ALL_HOURS,
                    'area': AREA,
                    'format': 'netcdf',
                }, part_file)
                sz = os.path.getsize(part_file) / 1e6
                print(f'    OK ({sz:.0f}MB)', flush=True)
            except Exception as e:
                print(f'    Failed: {e}', flush=True)
                if os.path.exists(part_file):
                    os.remove(part_file)
                sys.exit(1)
        else:
            print(f'  [{vi+1}/{len(ALL_VARS)}] {var} {ym} (skip)', flush=True)
        monthly_parts.append(part_file)

    print(f'  Merging {len(monthly_parts)} variables...', flush=True)
    _merge_and_cleanup(monthly_parts, out_file)

print('All downloads done.')


def _merge_and_cleanup(part_files, out_file):
    if os.path.exists(out_file):
        os.remove(out_file)

    src = nc.Dataset(part_files[0])
    nc_out = nc.Dataset(out_file, 'w', format='NETCDF3_64BIT')
    for dim in src.dimensions:
        nc_out.createDimension(dim,
            len(src.dimensions[dim]) if not src.dimensions[dim].isunlimited() else None)
    for var in src.variables:
        v = src.variables[var]
        nc_out.createVariable(var, v.dtype, tuple(v.dimensions))
        for attr in v.ncattrs():
            nc_out.variables[var].setncattr(attr, v.getncattr(attr))
        nc_out.variables[var][:] = v[:]
    src.close()

    for pf in part_files[1:]:
        src = nc.Dataset(pf)
        for var in src.variables:
            nc_out.variables[var][:] = src.variables[var][:]
        src.close()
        os.remove(pf)

    nc_out.close()
    print(f'  Merged: {out_file} ({os.path.getsize(out_file)/1e6:.0f}MB)')
