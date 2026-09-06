from roms_prepro.remapping import process_file

# 最简：全自动（变量自动检测 temp/salt/u/v/AK*，深度用默认标准层）
process_file(''/data/hdl/test/ocean_avg_0003.nc'')

# # 完整：指定变量 + 指定深度 + 覆盖垂直参数（只需给出要覆盖的键）
# process_file('/data/hdl/test/ocean_avg_0003.nc', 'out_z.nc',
#              variables=['temp', 'salt'],
#              std_depths=[0, 10, 25, 50, 100, 200, 500, 1000],
#              vgrid_params={'Vtransform': 2, 'Vstretching': 4,
#                            'theta_s': 7.0, 'theta_b': 0.1, 'Tcline': 20.0, 'N': 30})
