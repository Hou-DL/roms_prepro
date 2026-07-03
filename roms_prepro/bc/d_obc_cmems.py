"""
CMEMS -> ROMS 边界条件制作
对标MATLAB d_obc_cmems.m + obc_mercator.m

用法:
    python d_obc_cmems.py

依赖:
    numpy, scipy, xarray, netCDF4, h5netcdf
    roms_tools.py (同目录)

输入文件要求:
-------------
CMEMS 数据支持以下几种组织方式：

1. 单文件模式：设置 CMEMS_FILE 为包含所有变量的文件路径
2. 目录模式：设置 DATA_DIR 为 CMEMS 数据目录，自动搜索文件

文件命名模式（自动识别）:
  - zos:     cmems_zos_*.nc, cmems_merged_*.nc
  - thetao:  cmems_thetao_*.nc, cmems_merged_*.nc
  - so:      cmems_so_*.nc, cmems_merged_*.nc
  - uo:      cmems_uo_*.nc, cmems_merged_*.nc
  - vo:      cmems_vo_*.nc, cmems_merged_*.nc

变量名（自动识别）:
  - SSH:     zos, sla, adt
  - 温度:    thetao, t, temperature, votemper
  - 盐度:    so, s, salinity, vosaline
  - U流速:   uo, u, water_u, vozocrtx
  - V流速:   vo, v, water_v, vomecrty

维度名（自动识别）:
  - 经度: longitude, lon, nav_lon
  - 纬度: latitude, lat, nav_lat
  - 深度: depth, lev, deptht
  - 时间: time, time_counter
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import griddata, interp1d, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from datetime import datetime
from roms_tools import stretching, set_depth, roms_vectors, uv_barotropic

# ============================================================
#  配置参数 - 修改以下参数
# ============================================================

# CMEMS数据来源:
# 方式1: 指定单个文件路径（该文件需包含所有变量: zos, thetao, so, uo, vo）
# CMEMS_FILE = 'path/to/cmems_all_variables.nc'
# 方式2: 指定目录，程序会自动搜索该目录下的各变量文件
CMEMS_FILE = None  # None表示使用DATA_DIR方式

DATA_DIR = '//DS1825/q1/public_ocean_data/Mercator/2025'
GRD_NAME = r'E:\Ocean_data\ERA5\romsinput\output\NSCS_grd_operational_adjust.nc'
BRY_NAME = 'roms_bry_2025.nc'

# 边界开关 [west, east, south, north]
BOUNDARY = [0, 1, 1, 0]

# ROMS垂直参数
N_LEVELS = 30
VTRANSFORM = 2
VSTRETCHING = 3
THETA_S = 2.5
THETA_B = 1.0
TCLINE = 25.0

# 时间范围
TIME_START = '2025-05-01 00:00:00'
TIME_END = '2025-09-30 23:00:00'

# ROMS时间基准
ROMS_TIME_REF = '1990-01-01 00:00:00'

# 填充值
FILL_VAL = 1.0e+37


# ============================================================
#  辅助函数
# ============================================================

def fill_nan_single(layer):
    """单层最近邻NaN填充（对标MATLAB bwdist）"""
    nan_mask = np.isnan(layer)
    if not np.any(nan_mask) or np.all(nan_mask):
        return layer
    _, idx = distance_transform_edt(~nan_mask, return_indices=True)
    result = layer.copy()
    result[nan_mask] = layer[idx[0][nan_mask], idx[1][nan_mask]]
    return result


def fill_nan_2d(data):
    """2D/3D最近邻NaN填充"""
    if data.ndim == 2:
        return fill_nan_single(data)
    result = data.copy()
    for k in range(data.shape[0]):
        result[k] = fill_nan_single(data[k])
    return result


def interp_boundary_2d(Finp, lat_1d, lon_1d, target_lat, target_lon):
    """2D边界线性插值 + nearest填NaN"""
    interp = RegularGridInterpolator((lat_1d, lon_1d), Finp,
                                      bounds_error=False, fill_value=np.nan)
    target_pts = np.column_stack([target_lat, target_lon])
    r = interp(target_pts)
    nan_mask = np.isnan(r)
    if np.any(nan_mask):
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        valid = ~np.isnan(Finp) & (Finp != 0)
        src_pts = np.column_stack([lat2d[valid], lon2d[valid]])
        src_vals = Finp[valid]
        r[nan_mask] = griddata(src_pts, src_vals, target_pts[nan_mask], method='nearest')
    return r


def interp_boundary_3d(Finp, lat_1d, lon_1d, target_lat, target_lon):
    """3D边界逐层线性插值 + nearest填NaN"""
    target_pts = np.column_stack([target_lat, target_lon])
    nlev = Finp.shape[0]
    result = np.zeros((len(target_lat), nlev))
    for k in range(nlev):
        layer = Finp[k]
        valid = ~np.isnan(layer) & (layer != 0)
        if not np.any(valid):
            continue
        interp = RegularGridInterpolator((lat_1d, lon_1d), layer,
                                          bounds_error=False, fill_value=np.nan)
        r = interp(target_pts)
        nan_mask = np.isnan(r)
        if np.any(nan_mask):
            lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
            src_pts = np.column_stack([lat2d[valid], lon2d[valid]])
            src_vals = layer[valid]
            r[nan_mask] = griddata(src_pts, src_vals, target_pts[nan_mask], method='nearest')
        result[:, k] = r
    return result


# ============================================================
#  主程序
# ============================================================

def main():
    print('=' * 50)
    print('CMEMS -> ROMS Boundary Conditions')
    print('=' * 50)
    t0 = time.time()

    time_start_dt = datetime.strptime(TIME_START, '%Y-%m-%d %H:%M:%S')
    time_end_dt = datetime.strptime(TIME_END, '%Y-%m-%d %H:%M:%S')
    roms_ref = datetime.strptime(ROMS_TIME_REF, '%Y-%m-%d %H:%M:%S')

    # ---- 1. 读取ROMS网格 ----
    print('\n[1/5] 读取ROMS网格...')
    try:
        ds = xr.open_dataset(GRD_NAME, engine='h5netcdf')
    except:
        ds = xr.open_dataset(GRD_NAME)
    h = ds['h'].values
    lon_rho, lat_rho = ds['lon_rho'].values, ds['lat_rho'].values
    lon_u, lat_u = ds['lon_u'].values, ds['lat_u'].values
    lon_v, lat_v = ds['lon_v'].values, ds['lat_v'].values
    mask_rho = ds['mask_rho'].values
    mask_u, mask_v = ds['mask_u'].values, ds['mask_v'].values
    angle = ds['angle'].values
    ds.close()
    Lp, Mp = h.shape
    print(f'  Grid: {Lp} x {Mp} (eta, xi)')

    # 垂直坐标
    hc = TCLINE; ssh = np.zeros_like(h)
    s_rho, Cs_r = stretching(VSTRETCHING, THETA_S, THETA_B, hc, N_LEVELS, 0)
    s_w, Cs_w = stretching(VSTRETCHING, THETA_S, THETA_B, hc, N_LEVELS, 1)
    z_r = set_depth(VTRANSFORM, VSTRETCHING, THETA_S, THETA_B, hc, N_LEVELS, 1, h, ssh)
    z_u = set_depth(VTRANSFORM, VSTRETCHING, THETA_S, THETA_B, hc, N_LEVELS, 3, h, ssh)
    z_v = set_depth(VTRANSFORM, VSTRETCHING, THETA_S, THETA_B, hc, N_LEVELS, 4, h, ssh)
    z_w = set_depth(VTRANSFORM, VSTRETCHING, THETA_S, THETA_B, hc, N_LEVELS, 5, h, ssh)
    Hz = z_w[:,:,1:N_LEVELS+1] - z_w[:,:,0:N_LEVELS]

    # 边界网格点 (eta=dim0, xi=dim1)
    # 边界开关 [west, east, south, north]
    w_bound, e_bound, s_bound, n_bound = BOUNDARY
    
    # East: xi=end = last column
    lon_east, lat_east = lon_rho[:, -1], lat_rho[:, -1]
    z_east = z_r[:, -1, :]
    bm_east = mask_rho[:, -1]
    
    # West: xi=0 = first column
    lon_west, lat_west = lon_rho[:, 0], lat_rho[:, 0]
    z_west = z_r[:, 0, :]
    bm_west = mask_rho[:, 0]
    
    # South: eta=0 = first row
    lon_south, lat_south = lon_rho[0, :], lat_rho[0, :]
    z_south = z_r[0, :, :]
    bm_south = mask_rho[0, :]
    
    # North: eta=end = last row
    lon_north, lat_north = lon_rho[-1, :], lat_rho[-1, :]
    z_north = z_r[-1, :, :]
    bm_north = mask_rho[-1, :]

    # 打印各边界信息
    if w_bound:
        print(f'  西边界海洋点: {int(np.sum(bm_west > 0))}/{Lp}')
    if e_bound:
        print(f'  东边界海洋点: {int(np.sum(bm_east > 0))}/{Lp}')
    if s_bound:
        print(f'  南边界海洋点: {int(np.sum(bm_south > 0))}/{Mp}')
    if n_bound:
        print(f'  北边界海洋点: {int(np.sum(bm_north > 0))}/{Mp}')

    # ---- 2. 发现CMEMS文件 ----
    print('\n[2/5] 发现CMEMS文件...')
    
    # 检查配置模式
    if CMEMS_FILE is not None and os.path.isfile(CMEMS_FILE):
        # 模式1: 使用指定的单个文件（需包含所有变量）
        print(f'  使用指定文件: {os.path.basename(CMEMS_FILE)}')
        nfiles = 1
        zfiles = tfiles = sfiles = ufiles = vfiles = [CMEMS_FILE]
    else:
        # 模式2: 在目录中搜索文件
        
        # 先尝试查找合并文件（包含所有变量的单个文件）
        merged_files = sorted(glob.glob(os.path.join(DATA_DIR, 'cmems_merged_*.nc')))
        
        if merged_files:
            # 使用合并文件模式
            nfiles = len(merged_files)
            print(f'  发现 {nfiles} 个合并文件')
            print(f'  文件列表: {[os.path.basename(f) for f in merged_files]}')
            zfiles = tfiles = sfiles = ufiles = vfiles = merged_files
        else:
            # 使用分离文件模式（各变量单独的文件）
            zfiles = sorted(glob.glob(os.path.join(DATA_DIR, 'cmems_zos_*.nc')))
            tfiles = sorted(glob.glob(os.path.join(DATA_DIR, 'cmems_thetao_*.nc')))
            sfiles = sorted(glob.glob(os.path.join(DATA_DIR, 'cmems_so_*.nc')))
            ufiles = sorted(glob.glob(os.path.join(DATA_DIR, 'cmems_uo_*.nc')))
            vfiles = sorted(glob.glob(os.path.join(DATA_DIR, 'cmems_vo_*.nc')))
            
            # 检查文件数量一致性
            file_counts = {
                'zos': len(zfiles),
                'thetao': len(tfiles),
                'so': len(sfiles),
                'uo': len(ufiles),
                'vo': len(vfiles)
            }
            
            nfiles = len(zfiles)
            
            # 输出各变量文件数量
            print(f'  各变量文件数量:')
            for var, cnt in file_counts.items():
                status = 'OK' if cnt == nfiles else 'MISS'
                print(f'    [{status}] {var}: {cnt} 个')
            
            # 检查是否有缺失
            missing_vars = [var for var, cnt in file_counts.items() if cnt == 0]
            if missing_vars:
                print(f'  Warning: 未找到 {missing_vars} 的文件')
            
            # 检查数量不一致
            inconsistent = [var for var, cnt in file_counts.items() if cnt != nfiles and cnt > 0]
            if inconsistent:
                print(f'  Warning: {inconsistent} 文件数量与zos不一致')
            
            print(f'\n  共 {nfiles} 个月文件')
    
    # 检查是否有数据
    if nfiles == 0:
        if CMEMS_FILE is not None:
            print(f'  Error: 指定的文件不存在!')
            print(f'  文件路径: {CMEMS_FILE}')
        else:
            print('  Error: 未找到任何CMEMS文件!')
            print(f'  搜索路径: {DATA_DIR}')
        sys.exit(1)

    # ---- 3. 创建NetCDF ----
    print(f'\n[3/5] 创建 {BRY_NAME}...')
    os.makedirs(os.path.dirname(os.path.abspath(BRY_NAME)), exist_ok=True)
    ncf = nc.Dataset(BRY_NAME, 'w', format='NETCDF4')
    ncf.createDimension('eta_rho', Lp)
    ncf.createDimension('xi_rho', Mp)
    ncf.createDimension('s_rho', N_LEVELS)
    ncf.createDimension('bry_time', None)
    tv = ncf.createVariable('bry_time', 'f8', ('bry_time',))
    tv.units = f'seconds since {ROMS_TIME_REF}'
    tv.calendar = 'gregorian'
    
    # 根据边界配置创建变量
    for vn in ['zeta', 'ubar', 'vbar']:
        if e_bound:
            ncf.createVariable(f'{vn}_east', 'f8', ('bry_time', 'eta_rho'), fill_value=FILL_VAL)
        if w_bound:
            ncf.createVariable(f'{vn}_west', 'f8', ('bry_time', 'eta_rho'), fill_value=FILL_VAL)
        if s_bound:
            ncf.createVariable(f'{vn}_south', 'f8', ('bry_time', 'xi_rho'), fill_value=FILL_VAL)
        if n_bound:
            ncf.createVariable(f'{vn}_north', 'f8', ('bry_time', 'xi_rho'), fill_value=FILL_VAL)
    for vn in ['temp', 'salt', 'u', 'v']:
        if e_bound:
            ncf.createVariable(f'{vn}_east', 'f8', ('bry_time', 'eta_rho', 's_rho'), fill_value=FILL_VAL)
        if w_bound:
            ncf.createVariable(f'{vn}_west', 'f8', ('bry_time', 'eta_rho', 's_rho'), fill_value=FILL_VAL)
        if s_bound:
            ncf.createVariable(f'{vn}_south', 'f8', ('bry_time', 'xi_rho', 's_rho'), fill_value=FILL_VAL)
        if n_bound:
            ncf.createVariable(f'{vn}_north', 'f8', ('bry_time', 'xi_rho', 's_rho'), fill_value=FILL_VAL)

    # ---- 4. 循环处理 ----
    print('\n[4/5] 处理数据...')
    BryRec = 0

    for n in range(nfiles):
        # 读取CMEMS网格
        ds = xr.open_dataset(zfiles[n], engine='h5netcdf')
        lon_1d, lat_1d = ds['longitude'].values, ds['latitude'].values
        time_dt = ds['time'].values
        ntime = len(time_dt)
        Ze_all = ds['zos'].values.astype(float)
        ds.close()

        # 读取深度
        ds = xr.open_dataset(tfiles[n], engine='h5netcdf')
        Tdepth = ds['depth'].values
        Te_all = ds['thetao'].values.astype(float)
        ds.close()

        ds = xr.open_dataset(sfiles[n], engine='h5netcdf')
        Sa_all = ds['so'].values.astype(float)
        ds.close()

        ds = xr.open_dataset(ufiles[n], engine='h5netcdf')
        Uv_all = ds['uo'].values.astype(float)
        ds.close()

        ds = xr.open_dataset(vfiles[n], engine='h5netcdf')
        Vv_all = ds['vo'].values.astype(float)
        ds.close()

        # 裁剪空深度层
        Tc = Te_all[0]
        valid_k = np.array([np.any(~np.isnan(Tc[k]) & (Tc[k] != 0)) for k in range(len(Tdepth))])
        Tdepth = Tdepth[valid_k]
        Te_all = Te_all[:, valid_k]
        Sa_all = Sa_all[:, valid_k]
        Uv_all = Uv_all[:, valid_k]
        Vv_all = Vv_all[:, valid_k]
        Nlev = len(Tdepth)
        depth_neg = -np.abs(Tdepth); depth_neg[0] = 0.0

        print(f'\n  [{n+1}/{nfiles}] {os.path.basename(zfiles[n])}, {Nlev} depth levels')

        for tstep in range(ntime):
            t = time_dt[tstep]
            if hasattr(t, 'astype'):
                t = t.astype('datetime64[ms]').astype(datetime)
            if t < time_start_dt or t > time_end_dt:
                continue

            bry_time = (t - roms_ref).total_seconds()
            ts = t.strftime('%Y-%m-%d %H:%M')
            t1 = time.time()
            print(f'    {ts}', end='')

            # ---- 读取+QC+填NaN ----
            Ze = fill_nan_2d(Ze_all[tstep].copy())
            Te = fill_nan_2d(Te_all[tstep].copy())
            Sa = fill_nan_2d(Sa_all[tstep].copy())
            Uv = fill_nan_2d(Uv_all[tstep].copy())
            Vv = fill_nan_2d(Vv_all[tstep].copy())

            # zeta均值修复
            zeta_mean = np.nanmean(Ze)

            # ---- 2D：zeta ----
            if e_bound:
                zeta_east = interp_boundary_2d(Ze, lat_1d, lon_1d, lat_east, lon_east) + zeta_mean
            if w_bound:
                zeta_west = interp_boundary_2d(Ze, lat_1d, lon_1d, lat_west, lon_west) + zeta_mean
            if s_bound:
                zeta_south = interp_boundary_2d(Ze, lat_1d, lon_1d, lat_south, lon_south) + zeta_mean
            if n_bound:
                zeta_north = interp_boundary_2d(Ze, lat_1d, lon_1d, lat_north, lon_north) + zeta_mean

            # ---- 3D：temp/salt/u/v ----
            if e_bound:
                temp_east_lev = interp_boundary_3d(Te, lat_1d, lon_1d, lat_east, lon_east)
                salt_east_lev = interp_boundary_3d(Sa, lat_1d, lon_1d, lat_east, lon_east)
                ueast_lev = interp_boundary_3d(Uv, lat_1d, lon_1d, lat_east, lon_east)
                veast_lev = interp_boundary_3d(Vv, lat_1d, lon_1d, lat_east, lon_east)
            if w_bound:
                temp_west_lev = interp_boundary_3d(Te, lat_1d, lon_1d, lat_west, lon_west)
                salt_west_lev = interp_boundary_3d(Sa, lat_1d, lon_1d, lat_west, lon_west)
                uwest_lev = interp_boundary_3d(Uv, lat_1d, lon_1d, lat_west, lon_west)
                vwest_lev = interp_boundary_3d(Vv, lat_1d, lon_1d, lat_west, lon_west)
            if s_bound:
                temp_south_lev = interp_boundary_3d(Te, lat_1d, lon_1d, lat_south, lon_south)
                salt_south_lev = interp_boundary_3d(Sa, lat_1d, lon_1d, lat_south, lon_south)
                usouth_lev = interp_boundary_3d(Uv, lat_1d, lon_1d, lat_south, lon_south)
                vsouth_lev = interp_boundary_3d(Vv, lat_1d, lon_1d, lat_south, lon_south)
            if n_bound:
                temp_north_lev = interp_boundary_3d(Te, lat_1d, lon_1d, lat_north, lon_north)
                salt_north_lev = interp_boundary_3d(Sa, lat_1d, lon_1d, lat_north, lon_north)
                unorth_lev = interp_boundary_3d(Uv, lat_1d, lon_1d, lat_north, lon_north)
                vnorth_lev = interp_boundary_3d(Vv, lat_1d, lon_1d, lat_north, lon_north)

            # ---- 垂直插值 ----
            if e_bound:
                temp_east = np.zeros((Lp, N_LEVELS)); salt_east = np.zeros((Lp, N_LEVELS))
                u_east = np.zeros((Lp, N_LEVELS)); v_east = np.zeros((Lp, N_LEVELS))
                for i in range(Lp):
                    if bm_east[i] > 0:
                        temp_east[i,:] = interp1d(depth_neg, temp_east_lev[i,:], bounds_error=False, fill_value=0)(z_east[i,:])
                        salt_east[i,:] = interp1d(depth_neg, salt_east_lev[i,:], bounds_error=False, fill_value=0)(z_east[i,:])
                        u_east[i,:] = interp1d(depth_neg, ueast_lev[i,:], bounds_error=False, fill_value=0)(z_east[i,:])
                        v_east[i,:] = interp1d(depth_neg, veast_lev[i,:], bounds_error=False, fill_value=0)(z_east[i,:])

            if w_bound:
                temp_west = np.zeros((Lp, N_LEVELS)); salt_west = np.zeros((Lp, N_LEVELS))
                u_west = np.zeros((Lp, N_LEVELS)); v_west = np.zeros((Lp, N_LEVELS))
                for i in range(Lp):
                    if bm_west[i] > 0:
                        temp_west[i,:] = interp1d(depth_neg, temp_west_lev[i,:], bounds_error=False, fill_value=0)(z_west[i,:])
                        salt_west[i,:] = interp1d(depth_neg, salt_west_lev[i,:], bounds_error=False, fill_value=0)(z_west[i,:])
                        u_west[i,:] = interp1d(depth_neg, uwest_lev[i,:], bounds_error=False, fill_value=0)(z_west[i,:])
                        v_west[i,:] = interp1d(depth_neg, vwest_lev[i,:], bounds_error=False, fill_value=0)(z_west[i,:])

            if s_bound:
                temp_south = np.zeros((Mp, N_LEVELS)); salt_south = np.zeros((Mp, N_LEVELS))
                u_south = np.zeros((Mp, N_LEVELS)); v_south = np.zeros((Mp, N_LEVELS))
                for j in range(Mp):
                    if bm_south[j] > 0:
                        temp_south[j,:] = interp1d(depth_neg, temp_south_lev[j,:], bounds_error=False, fill_value=0)(z_south[j,:])
                        salt_south[j,:] = interp1d(depth_neg, salt_south_lev[j,:], bounds_error=False, fill_value=0)(z_south[j,:])
                        u_south[j,:] = interp1d(depth_neg, usouth_lev[j,:], bounds_error=False, fill_value=0)(z_south[j,:])
                        v_south[j,:] = interp1d(depth_neg, vsouth_lev[j,:], bounds_error=False, fill_value=0)(z_south[j,:])

            if n_bound:
                temp_north = np.zeros((Mp, N_LEVELS)); salt_north = np.zeros((Mp, N_LEVELS))
                u_north = np.zeros((Mp, N_LEVELS)); v_north = np.zeros((Mp, N_LEVELS))
                for j in range(Mp):
                    if bm_north[j] > 0:
                        temp_north[j,:] = interp1d(depth_neg, temp_north_lev[j,:], bounds_error=False, fill_value=0)(z_north[j,:])
                        salt_north[j,:] = interp1d(depth_neg, salt_north_lev[j,:], bounds_error=False, fill_value=0)(z_north[j,:])
                        u_north[j,:] = interp1d(depth_neg, unorth_lev[j,:], bounds_error=False, fill_value=0)(z_north[j,:])
                        v_north[j,:] = interp1d(depth_neg, vnorth_lev[j,:], bounds_error=False, fill_value=0)(z_north[j,:])

            # ---- 速度旋转 ----
            if e_bound:
                a_east = angle[:, -1]
                u_east_rot = u_east * np.cos(a_east[:, None]) + v_east * np.sin(a_east[:, None])
                v_east_rot = v_east * np.cos(a_east[:, None]) - u_east * np.sin(a_east[:, None])
            if w_bound:
                a_west = angle[:, 0]
                u_west_rot = u_west * np.cos(a_west[:, None]) + v_west * np.sin(a_west[:, None])
                v_west_rot = v_west * np.cos(a_west[:, None]) - u_west * np.sin(a_west[:, None])
            if s_bound:
                a_south = angle[0, :]
                u_south_rot = u_south * np.cos(a_south[:, None]) + v_south * np.sin(a_south[:, None])
                v_south_rot = v_south * np.cos(a_south[:, None]) - u_south * np.sin(a_south[:, None])
            if n_bound:
                a_north = angle[-1, :]
                u_north_rot = u_north * np.cos(a_north[:, None]) + v_north * np.sin(a_north[:, None])
                v_north_rot = v_north * np.cos(a_north[:, None]) - u_north * np.sin(a_north[:, None])

            # ---- 正压速度 ----
            if e_bound:
                Hz_east = Hz[:, -1, :]
                ubar_east = np.sum(u_east_rot * Hz_east, axis=1) / np.sum(Hz_east, axis=1)
                vbar_east = np.sum(v_east_rot * Hz_east, axis=1) / np.sum(Hz_east, axis=1)
            if w_bound:
                Hz_west = Hz[:, 0, :]
                ubar_west = np.sum(u_west_rot * Hz_west, axis=1) / np.sum(Hz_west, axis=1)
                vbar_west = np.sum(v_west_rot * Hz_west, axis=1) / np.sum(Hz_west, axis=1)
            if s_bound:
                Hz_south = Hz[0, :, :]
                ubar_south = np.sum(u_south_rot * Hz_south, axis=1) / np.sum(Hz_south, axis=1)
                vbar_south = np.sum(v_south_rot * Hz_south, axis=1) / np.sum(Hz_south, axis=1)
            if n_bound:
                Hz_north = Hz[-1, :, :]
                ubar_north = np.sum(u_north_rot * Hz_north, axis=1) / np.sum(Hz_north, axis=1)
                vbar_north = np.sum(v_north_rot * Hz_north, axis=1) / np.sum(Hz_north, axis=1)

            # ---- 陆地点设为fill_val ----
            if e_bound:
                zeta_east[bm_east == 0] = FILL_VAL
                temp_east[bm_east == 0] = FILL_VAL
                salt_east[bm_east == 0] = FILL_VAL
                u_east_rot[bm_east == 0] = FILL_VAL
                v_east_rot[bm_east == 0] = FILL_VAL
            if w_bound:
                zeta_west[bm_west == 0] = FILL_VAL
                temp_west[bm_west == 0] = FILL_VAL
                salt_west[bm_west == 0] = FILL_VAL
                u_west_rot[bm_west == 0] = FILL_VAL
                v_west_rot[bm_west == 0] = FILL_VAL
            if s_bound:
                zeta_south[bm_south == 0] = FILL_VAL
                temp_south[bm_south == 0] = FILL_VAL
                salt_south[bm_south == 0] = FILL_VAL
                u_south_rot[bm_south == 0] = FILL_VAL
                v_south_rot[bm_south == 0] = FILL_VAL
            if n_bound:
                zeta_north[bm_north == 0] = FILL_VAL
                temp_north[bm_north == 0] = FILL_VAL
                salt_north[bm_north == 0] = FILL_VAL
                u_north_rot[bm_north == 0] = FILL_VAL
                v_north_rot[bm_north == 0] = FILL_VAL

            # ---- 写入 ----
            BryRec += 1
            ncf.variables['bry_time'][BryRec-1] = bry_time
            if e_bound:
                ncf.variables['zeta_east'][BryRec-1] = zeta_east
                ncf.variables['temp_east'][BryRec-1] = temp_east
                ncf.variables['salt_east'][BryRec-1] = salt_east
                ncf.variables['u_east'][BryRec-1] = u_east_rot
                ncf.variables['v_east'][BryRec-1] = v_east_rot
                ncf.variables['ubar_east'][BryRec-1] = ubar_east
                ncf.variables['vbar_east'][BryRec-1] = vbar_east
            if w_bound:
                ncf.variables['zeta_west'][BryRec-1] = zeta_west
                ncf.variables['temp_west'][BryRec-1] = temp_west
                ncf.variables['salt_west'][BryRec-1] = salt_west
                ncf.variables['u_west'][BryRec-1] = u_west_rot
                ncf.variables['v_west'][BryRec-1] = v_west_rot
                ncf.variables['ubar_west'][BryRec-1] = ubar_west
                ncf.variables['vbar_west'][BryRec-1] = vbar_west
            if s_bound:
                ncf.variables['zeta_south'][BryRec-1] = zeta_south
                ncf.variables['temp_south'][BryRec-1] = temp_south
                ncf.variables['salt_south'][BryRec-1] = salt_south
                ncf.variables['u_south'][BryRec-1] = u_south_rot
                ncf.variables['v_south'][BryRec-1] = v_south_rot
                ncf.variables['ubar_south'][BryRec-1] = ubar_south
                ncf.variables['vbar_south'][BryRec-1] = vbar_south
            if n_bound:
                ncf.variables['zeta_north'][BryRec-1] = zeta_north
                ncf.variables['temp_north'][BryRec-1] = temp_north
                ncf.variables['salt_north'][BryRec-1] = salt_north
                ncf.variables['u_north'][BryRec-1] = u_north_rot
                ncf.variables['v_north'][BryRec-1] = v_north_rot
                ncf.variables['ubar_north'][BryRec-1] = ubar_north
                ncf.variables['vbar_north'][BryRec-1] = vbar_north

            print(f' {time.time()-t1:.1f}s')

    ncf.close()

    # ---- 5. 完成 ----
    print(f'\n[5/5] 完成! {BryRec} 条记录, 总耗时 {time.time()-t0:.0f}s')
    print(f'输出: {os.path.abspath(BRY_NAME)}')


# 需要导入glob
import glob

if __name__ == '__main__':
    main()