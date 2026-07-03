"""
ROMS → ROMS initial conditions via intermediate standard-z grid.

输入文件要求:
-------------
源 ROMS 历史文件应包含: temp, salt, zeta, u, v, angle
以及维度: ocean_time, s_rho, eta_rho, xi_rho 等

支持指定单个文件路径或目录（自动搜索 *.nc 文件）。
时间参考 (time_ref) 自动从源文件的 ocean_time 单位属性读取。
"""

import os
import re
import glob
import numpy as np
import netCDF4 as nc4

from ._core import (horizontal_interp, sigma_to_z, z_to_sigma,
                    rotate_uv, uv_to_cgrid, compute_ubar_vbar,
                    write_ic_file, _parse_date, _get_time, _match_idx,
                    _read_roms_grid)

DEFAULT_Z = np.array([
    -7500, -7000, -6500, -6000, -5500, -5000, -4500, -4000, -3500,
    -3000, -2500, -2000, -1750, -1500, -1250, -1000, -900, -800, -700,
    -600, -500, -400, -300, -250, -200, -175, -150, -125, -100, -90,
    -80, -70, -60, -50, -45, -40, -35, -30, -25, -20, -17.5,
    -15, -12.5, -10, -7.5, -5, -2.5, 0
])


def _get_var(filename, varname, time_idx=0):
    """Read one variable from a ROMS file."""
    import netCDF4 as nc4
    ds = nc4.Dataset(filename)
    arr = ds.variables[varname][:]
    if arr.ndim == 4:
        arr = arr[time_idx]
    ds.close()
    return arr


def _remap_var(var_name, src_file, src_grd, src_z_r, src_z_w,
               dst_grd, dst_z_r, dst_z_w, z_levels,
               src_time=0, spval=1e37):
    """
    Remap one variable from source ROMS to target ROMS via intermediate Z.
    """
    src_var = _get_var(src_file, var_name, src_time)
    ndim = 3 if src_var.ndim == 3 else 2

    src_lon = src_grd['lon_rho']
    src_lat = src_grd['lat_rho']
    dst_lon = dst_grd['lon_rho']
    dst_lat = dst_grd['lat_rho']
    mask = dst_grd['mask_rho']

    if ndim == 2:
        return horizontal_interp(src_lon, src_lat, src_var,
                                 dst_lon, dst_lat, mask=mask,
                                 fill_value=spval)

    var_z = sigma_to_z(src_var, src_z_r, z_levels, fill_value=spval)
    var_z_dst = np.zeros((len(z_levels), dst_lon.shape[0], dst_lon.shape[1]))
    for k in range(len(z_levels)):
        var_z_dst[k] = horizontal_interp(src_lon, src_lat, var_z[k],
                                          dst_lon, dst_lat, mask=mask,
                                          fill_value=spval)
    return z_to_sigma(var_z_dst, z_levels, dst_z_r, fill_value=spval)


def _detect_roms_time_ref(src_file):
    """
    自动从 ROMS 文件的 ocean_time 变量读取时间参考。

    返回格式: 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD'
    """
    try:
        ds = nc4.Dataset(src_file, 'r')
        # 查找 ocean_time 变量
        time_var = None
        for candidate in ['ocean_time', 'time', 'time_counter']:
            if candidate in ds.variables:
                time_var = ds.variables[candidate]
                break
        if time_var is None:
            ds.close()
            return None

        units = getattr(time_var, 'units', '')
        ds.close()

        # 解析单位字符串
        m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', units)
        if m:
            ref_str = m.group(1).strip()
            # 标准化格式
            try:
                dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    dt = datetime.strptime(ref_str, '%Y-%m-%d')
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    return ref_str
        return None
    except Exception:
        return None


def _detect_roms_files(src_hist_file=None, src_hist_dir=None):
    """
    自动检测 ROMS 历史文件。

    返回排序后的文件路径列表。
    """
    if src_hist_file and os.path.isfile(src_hist_file):
        return [src_hist_file]

    if src_hist_dir and os.path.isdir(src_hist_dir):
        patterns = ['*.nc', 'ocean_his_*.nc', 'ocean_avg_*.nc', 'roms_his_*.nc']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(src_hist_dir, pattern)))
        if files:
            return sorted(set(files))

    return []


def roms_to_roms_ini(src_grid_file, src_hist_file=None, dst_grid_file=None,
                     ini_file=None, src_time=0,
                     Vtransform=2, Vstretching=4,
                     theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                     z_levels=None, var_mapping=None,
                     init_date=None, time_ref=None,
                     src_hist_dir=None):
    """
    Create ROMS IC from another ROMS run.

    Parameters
    ----------
    src_hist_file : str, optional
        源 ROMS 历史文件路径
    src_hist_dir : str, optional
        源 ROMS 历史文件目录（自动搜索）
    init_date : str
        指定制作哪一天的 IC，如 '2025-05-01'
    time_ref : str, optional
        时间参考，自动从源文件 ocean_time 单位读取
    """
    from ..grid import set_depth

    if z_levels is None:
        z_levels = DEFAULT_Z

    # 自动检测文件
    src_files = _detect_roms_files(src_hist_file, src_hist_dir)
    if not src_files:
        raise FileNotFoundError('未找到 ROMS 历史文件')

    # 使用第一个文件获取时间和网格信息
    first_file = src_files[0]

    # 自动检测时间参考
    if time_ref is None:
        time_ref = _detect_roms_time_ref(first_file)
        if time_ref:
            print(f'  自动检测时间参考: {time_ref}')
        else:
            time_ref = '1970-01-01'
            print(f'  警告: 无法检测时间参考，使用默认值: {time_ref}')

    src_grd = _read_roms_grid(src_grid_file)
    dst_grd = _read_roms_grid(dst_grid_file)

    # Resolve time
    src_times = _get_time(first_file)
    if init_date is not None and src_times is not None:
        src_time = _match_idx(src_times, init_date)
        print(f"  init_date={init_date} → time index {src_time}")
    ocean_time_val = src_times[src_time] if (init_date and src_times is not None) else 0.0

    # 计算 ocean_time
    m = re.search(r'since\s+(.+)', time_ref)
    if m:
        ref_epoch = _parse_date(m.group(1))
    else:
        ref_epoch = _parse_date(time_ref)
    ocean_time_val = ocean_time_val - ref_epoch
    ocean_time = np.array([ocean_time_val])

    # Source depths
    src_Vt = int(src_grd.get('Vtransform', 1))
    src_Vs = int(src_grd.get('Vstretching', 1))
    src_N = 30
    if src_hist_file:
        import netCDF4 as nc4
        h = nc4.Dataset(src_hist_file)
        if 's_rho' in h.dimensions:
            src_N = len(h.dimensions['s_rho'])
        h.close()

    src_z_w = set_depth(src_Vt, src_Vs,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N,
                        src_grd['h'], igrid=5)
    src_z_r = set_depth(src_Vt, src_Vs,
                        src_grd.get('theta_s', 5.0),
                        src_grd.get('theta_b', 0.4),
                        src_grd.get('Tcline', 10.0), src_N,
                        src_grd['h'], igrid=1)

    # Target depths
    dst_z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=5)
    dst_z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=1)

    spval = 1e37

    # Resolve variable names
    vmap = var_mapping or {}
    v = lambda k: vmap.get(k, k)

    # Remap source rotation angle
    parent_angle = _remap_var(v('angle'), src_grid_file, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    target_angle = dst_grd.get('angle', np.zeros_like(dst_grd['lat_rho']))

    print("Remapping 3D ...")
    temp = _remap_var(v('temp'), src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    salt = _remap_var(v('salt'), src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    zeta = _remap_var(v('zeta'), src_hist_file, src_grd, src_z_r, src_z_w,
                      dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)

    print("Remapping velocity ...")
    u_rho = _remap_var(v('u'), src_hist_file, src_grd, src_z_r, src_z_w,
                       dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)
    v_rho = _remap_var(v('v'), src_hist_file, src_grd, src_z_r, src_z_w,
                       dst_grd, dst_z_r, dst_z_w, z_levels, src_time, spval)

    # Rotate: source XI/ETA → true N/E → target XI/ETA
    u_rho, v_rho = rotate_uv(u_rho, v_rho, parent_angle, target_angle)
    u, v = uv_to_cgrid(u_rho, v_rho, dst_grd.get('mask_u'),
                       dst_grd.get('mask_v'), spval)
    ubar, vbar = compute_ubar_vbar(u, v, dst_z_w, dst_grd.get('mask_u'),
                                   dst_grd.get('mask_v'))

    # 陆地点设为 fill_value（与 cmems_ic 一致）
    spval = 1e37
    wm3d_rho = np.tile(dst_grd['mask_rho'], (N, 1, 1))
    temp[wm3d_rho == 0] = spval
    salt[wm3d_rho == 0] = spval
    zeta[dst_grd['mask_rho'] == 0] = spval

    wm3d_u = np.tile(dst_grd.get('mask_u', np.ones_like(dst_grd['mask_rho'])), (N, 1, 1))
    u[wm3d_u == 0] = spval
    ubar[dst_grd.get('mask_u', np.ones_like(dst_grd['mask_rho'])) == 0] = spval

    wm3d_v = np.tile(dst_grd.get('mask_v', np.ones_like(dst_grd['mask_rho'])), (N, 1, 1))
    v[wm3d_v == 0] = spval
    vbar[dst_grd.get('mask_v', np.ones_like(dst_grd['mask_rho'])) == 0] = spval

    vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                    'theta_s': theta_s, 'theta_b': theta_b,
                    'Tcline': Tcline, 'hc': Tcline}
    write_ic_file(ini_file, dst_grd, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar,
                  time_ref=time_ref or '1990-01-01')
    print(f"Written: {ini_file}")
