from era5_to_roms import ERA5toROMS

# # 方式1：处理全部数据（自动检测时间范围和基准日期）
converter1 = ERA5toROMS(
    in_dir=r'E:\Ocean_data\ERA5\romsinput',
    out_dir=r'E:\Ocean_data\ERA5\romsinput\output',
    base_date='1990-01-01 00:00:00',
    variables=['t2m', 'd2m', 'msl', 'tp', 'tcc', 'msdrswrf', 'msnlwrf', 'msdwlwrf', 'r'],
    forcing_file='era5_forc2025day.nc',  
)
converter1.process()

# # 方式2：只处理一个月
# converter = ERA5toROMS(
#     in_dir=r'E:\Ocean_data\ERA5\romsinput',
#     out_dir=r'E:\Ocean_data\ERA5\romsinput\output',
#     time_start='2025-05-01 00:00:00',
#     time_end='2025-05-31 23:00:00',
# )

# # 方式3：自定义基准日期（如ROMS需要）
# converter = ERA5toROMS(
#     in_dir=r'E:\Ocean_data\ERA5\romsinput',
#     out_dir=r'E:\Ocean_data\ERA5\romsinput\output',
#     base_date='1990-01-01 00:00:00',
# )
# 方式4：只处理某些变量
# converter3 = ERA5toROMS(
#     in_dir=r'D:\工作文件\6.10不同风场对比\windfarm\normal',
#     out_dir=r'D:\工作文件\6.10不同风场对比\windfarm\output',
#     base_date='1990-01-01 00:00:00',
#     wind_file='era5_normalwind_2025day.nc',   
#     variables=['u10', 'v10'],
# )
# converter3.process()

# converter2 = ERA5toROMS(
#     in_dir=r'D:\工作文件\6.10不同风场对比\windfarm\wake',
#     out_dir=r'D:\工作文件\6.10不同风场对比\windfarm\output',
#     base_date='1990-01-01 00:00:00',
#     wind_file='era5_wakewind_2025day.nc',   
#     variables=['u10', 'v10'],
# )
# converter2.process()

