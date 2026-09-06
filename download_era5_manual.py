import cdsapi, os, sys

OUTPUT_DIR = '/data/hdl/oceanfiles/era5/in'
client = cdsapi.Client()
# API 客户端初始化
try:
    client = cdsapi.Client()
except Exception:
    # 如果自动读取 .cdsapirc 失败，请在这里填入您的 Key
    CDS_URL = "https://cds.climate.copernicus.eu/api"
    CDS_KEY = "7d647511-a727-4322-a7a8-1689ca68e005"
    client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
    
VARIABLES = [
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

DOWNLOAD_MONTHS = [
    ('2024', '12'),
    ('2025', '01'),
    ('2025', '02'),
]

AREA = [38, 118, 32, 127]
ALL_DAYS = [f'{d:02d}' for d in range(1, 32)]
ALL_HOURS = [f'{h:02d}:00' for h in range(24)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for year, month in DOWNLOAD_MONTHS:
    out_file = os.path.join(OUTPUT_DIR, f'ERA5_{year}{month}.nc')
    if os.path.exists(out_file):
        print(f'Skip existing: {out_file} ({os.path.getsize(out_file)/1e6:.0f}MB)')
        continue

    print(f'Downloading {year}-{month}...')
    req = {
        'product_type': 'reanalysis',
        'variable': VARIABLES,
        'year': year,
        'month': month,
        'day': ALL_DAYS,
        'time': ALL_HOURS,
        'area': AREA,
        'format': 'netcdf',
    }
    try:
        client.retrieve('reanalysis-era5-single-levels', req, out_file)
        print(f'Saved: {out_file} ({os.path.getsize(out_file)/1e6:.0f}MB)')
    except Exception as e:
        print(f'Failed: {e}')
        sys.exit(1)

print('All downloads done.')