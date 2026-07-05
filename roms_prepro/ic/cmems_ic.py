"""
CMEMS/Mercator/HYCOM → ROMS initial conditions.

输入文件要求:
-------------
CMEMS 数据支持以下几种组织方式：

1. 单文件模式：一个文件包含所有变量（zos, thetao, so, uo, vo）
2. 多文件模式：每个变量一个文件（CMEMS 标准下载格式）
3. 目录模式：指定目录，自动搜索匹配的文件

文件命名模式（自动识别）:
  - zos:     cmems_zos_*.nc, cmems_mod_glo_phy_my_0.083deg_P1D-m_*.nc
  - thetao:  cmems_thetao_*.nc, cmems_mod_glo_phy_my_0.083deg_P1D-m_*.nc
  - so:      cmems_so_*.nc, cmems_mod_glo_phy_my_0.083deg_P1D-m_*.nc
  - uo:      cmems_uo_*.nc, cmems_mod_glo_phy_my_0.083deg_P1D-m_*.nc
  - vo:      cmems_vo_*.nc, cmems_mod_glo_phy_my_0.083deg_P1D-m_*.nc

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
  - 时间: time, time_counter, t
"""

import os
import re
import glob
import numpy as np
import netCDF4 as nc4
from datetime import datetime, timedelta

try:
    from ._core import (horizontal_interp, z_to_sigma,
                        rotate_uv, uv_to_cgrid, compute_ubar_vbar,
                        write_ic_file, _fill_nan, _parse_date, _get_time,
                        _match_idx, _read_source, _read_roms_grid)
except ImportError:
    from _core import (horizontal_interp, z_to_sigma,
                       rotate_uv, uv_to_cgrid, compute_ubar_vbar,
                       write_ic_file, _fill_nan, _parse_date, _get_time,
                       _match_idx, _read_source, _read_roms_grid)


def _find_var_name(ds, candidates):
    """在数据集中查找匹配的变量名"""
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _find_dim_name(ds, candidates):
    """在数据集中查找匹配的维度名"""
    for name in candidates:
        if name in ds.dimensions:
            return name
    return None


def _read_cmems_file(file_path, var_name=None, time_index=0):
    """
    从CMEMS NetCDF文件读取单个变量。

    自动识别变量名、维度名和坐标名。
    如果未指定var_name，则尝试从常用CMEMS变量名中自动识别。
    """
    ds = nc4.Dataset(file_path, 'r')

    # 自动识别坐标维度名
    lon_name = _find_dim_name(ds, ['longitude', 'lon', 'nav_lon'])
    lat_name = _find_dim_name(ds, ['latitude', 'lat', 'nav_lat'])
    depth_name = _find_dim_name(ds, ['depth', 'lev', 'deptht'])
    time_name = _find_dim_name(ds, ['time', 'time_counter', 't'])

    # 读取坐标
    lon_1d = np.asarray(ds.variables[lon_name][:], dtype=float) if lon_name else None
    lat_1d = np.asarray(ds.variables[lat_name][:], dtype=float) if lat_name else None

    # 如果未指定变量名，尝试自动识别
    if var_name is None:
        var_candidates = {
            'zos': ['zos', 'sla', 'adt'],
            'thetao': ['thetao', 't', 'temperature', 'votemper'],
            'so': ['so', 's', 'salinity', 'vosaline'],
            'uo': ['uo', 'u', 'water_u', 'vozocrtx'],
            'vo': ['vo', 'v', 'water_v', 'vomecrty'],
        }
        for std_name, candidates in var_candidates.items():
            found = _find_var_name(ds, candidates)
            if found:
                var_name = found
                break
        if var_name is None:
            ds.close()
            raise ValueError(f"无法在 {file_path} 中识别CMEMS变量")
    elif var_name not in ds.variables:
        # 尝试别名
        var_aliases = {
            'zos': ['zos', 'sla', 'adt'],
            'thetao': ['thetao', 't', 'temperature', 'votemper'],
            'so': ['so', 's', 'salinity', 'vosaline'],
            'uo': ['uo', 'u', 'water_u', 'vozocrtx'],
            'vo': ['vo', 'v', 'water_v', 'vomecrty'],
        }
        found = None
        for aliases in var_aliases.values():
            if var_name in aliases:
                found = _find_var_name(ds, aliases)
                break
        if found:
            var_name = found
        else:
            ds.close()
            raise ValueError(f"变量 {var_name} 不在 {file_path} 中")

    v = ds.variables[var_name]
    data = v[:]

    # 处理时间维度
    var_dims = v.dimensions
    has_time = (time_name and time_name in var_dims and len(var_dims) >= 3)
    if has_time and data.shape[0] > 1:
        data = data[min(time_index, data.shape[0] - 1)]

    # 转换masked array
    if hasattr(data, 'mask'):
        data = data.filled(np.nan)
    data = np.asarray(data, dtype=float)

    # 读取深度
    depth = None
    if depth_name and depth_name in ds.variables:
        depth = np.asarray(ds.variables[depth_name][:], dtype=float)
        depth = -np.abs(depth)

    # 读取时间单位
    time_units = None
    if time_name and time_name in ds.variables:
        time_units = getattr(ds.variables[time_name], 'units', '')

    ds.close()

    return {
        'lon_1d': lon_1d,
        'lat_1d': lat_1d,
        'data': data,
        'depth': depth,
        'time_units': time_units
    }


def _detect_cmems_files(data_dir):
    """
    自动检测CMEMS文件。

    返回字典 {标准变量名: 文件路径}。
    支持单文件（所有变量在一个文件中）和多文件模式。
    """
    var_patterns = {
        'zos': ['zos', 'ssh', 'sla', 'adt', 'surf_el'],
        'thetao': ['thetao', 't', 'temperature', 'votemper', 'water_temp'],
        'so': ['so', 's', 'salinity', 'vosaline'],
        'uo': ['uo', 'u', 'water_u', 'vozocrtx'],
        'vo': ['vo', 'v', 'water_v', 'vomecrty'],
    }

    result = {}
    files_checked = set()

    # 首先检查是否有合并文件（包含所有变量）
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.nc')))
    for fpath in all_files:
        try:
            ds = nc4.Dataset(fpath, 'r')
            found_vars = {}
            for std_name, patterns in var_patterns.items():
                found = _find_var_name(ds, patterns)
                if found:
                    found_vars[std_name] = found
            ds.close()
            if len(found_vars) >= 3:  # 至少找到3个变量
                for std_name in found_vars:
                    result[std_name] = fpath
                return result
        except Exception:
            continue

    # 多文件模式：每个变量一个文件
    file_patterns = {
        'zos': ['cmems_zos_*.nc', 'cmems_*_zos_*.nc', '*zos*.nc', '*ssh*.nc'],
        'thetao': ['cmems_thetao_*.nc', 'cmems_*_thetao_*.nc', '*thetao*.nc', '*temp*.nc'],
        'so': ['cmems_so_*.nc', 'cmems_*_so_*.nc', '*so_*.nc', '*salt*.nc'],
        'uo': ['cmems_uo_*.nc', 'cmems_*_uo_*.nc', '*uo_*.nc', '*_u_*.nc'],
        'vo': ['cmems_vo_*.nc', 'cmems_*_vo_*.nc', '*vo_*.nc', '*_v_*.nc'],
    }

    for std_name, patterns in file_patterns.items():
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(data_dir, pattern)))
            if matches:
                result[std_name] = matches[0]
                break

    return result


def _fill_nan_2d_range_check(data, var_type='temp'):
    """Apply range check and fill NaN using 2D nearest neighbor."""
    data = data.copy()

    if var_type == 'temp':
        data[np.abs(data) > 50] = np.nan
        data[data < -100] = np.nan
    elif var_type == 'salt':
        data[data < 0] = np.nan
        data[data > 50] = np.nan
        data[np.abs(data) > 100] = np.nan
    elif var_type == 'vel':
        data[np.abs(data) > 10] = np.nan
        data[np.abs(data) > 100] = np.nan
    elif var_type == 'zeta':
        data[np.abs(data) > 10] = np.nan
        data[np.abs(data) > 100] = np.nan

    if data.ndim == 2:
        try:
            from ._core import _fill_nan_2d
        except ImportError:
            from _core import _fill_nan_2d
        return _fill_nan_2d(data)
    else:
        try:
            from ._core import _fill_nan_2d
        except ImportError:
            from _core import _fill_nan_2d
        for k in range(data.shape[0]):
            data[k] = _fill_nan_2d(data[k])
        return data


def _fill_nans_vertically(data):
    """Fill NaN values vertically using nearest valid neighbor (vectorized)."""
    data = data.copy()
    nz, ny, nx = data.shape

    valid = np.isfinite(data) & (data != 0)
    if valid.all():
        return data

    need_fill = np.any(valid, axis=0) & np.any(~valid, axis=0)
    if not np.any(need_fill):
        return data

    k_idx = np.arange(nz)
    j_idx, i_idx = np.where(need_fill)

    for j, i in zip(j_idx, i_idx):
        valid_k = k_idx[valid[:, j, i]]
        nan_k = k_idx[~valid[:, j, i]]
        nearest = valid_k[np.argmin(np.abs(nan_k[:, None] - valid_k[None, :]), axis=1)]
        data[nan_k, j, i] = data[nearest, j, i]

    return data


def _extend_depth(depth, data, hmax, var_type='scalar'):
    """Extend CMEMS depth levels if they don't cover ROMS max depth."""
    depth_pos = np.abs(depth)

    if depth_pos[-1] >= hmax:
        return depth, data

    print(f'  Extending depth to {hmax + 200:.0f} m for deep water extrapolation')

    new_depths = np.array([depth_pos[-1] + 200, hmax + 200])
    depth_extended = np.concatenate([depth_pos, new_depths])

    if var_type == 'scalar':
        last_layer = data[-1:, :, :]
        data_extended = np.concatenate([data, np.tile(last_layer, (2, 1, 1))], axis=0)
    else:
        zeros = np.zeros((2, data.shape[1], data.shape[2]))
        data_extended = np.concatenate([data, zeros], axis=0)

    depth_extended = -depth_extended

    return depth_extended, data_extended


def _detect_cmems_time_ref(src_file):
    """
    自动从 CMEMS 文件读取时间参考。

    返回格式: 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
    """
    try:
        ds = nc4.Dataset(src_file, 'r')
        time_var = None
        for candidate in ['time', 'time_counter', 't']:
            if candidate in ds.variables:
                time_var = ds.variables[candidate]
                break
        if time_var is None:
            ds.close()
            return None

        units = getattr(time_var, 'units', '')
        ds.close()

        m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', units)
        if m:
            ref_str = m.group(1).strip()
            try:
                dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                try:
                    dt = datetime.strptime(ref_str, '%Y-%m-%d')
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    return ref_str
        return None
    except Exception:
        return None


def mercator_to_roms_ini(roms_grid_file, source_file, ini_file,
                         Vtransform=2, Vstretching=4,
                         theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                         init_date=None, time_ref=None,
                         **source_kwargs):
    """
    Create ROMS IC from a Mercator/HYCOM/CMEMS file.

    Parameters
    ----------
    init_date : str
        指定制作哪一天的 IC，如 '2025-05-01'
    time_ref : str, optional
        时间参考，自动从源文件时间单位读取
    """
    try:
        from ..grid import set_depth
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from grid import set_depth
    metrics = _read_roms_grid(roms_grid_file)

    # 自动检测时间参考
    if time_ref is None:
        time_ref = _detect_cmems_time_ref(source_file)
        if time_ref:
            print(f'  自动检测时间参考: {time_ref}')
        else:
            time_ref = '1970-01-01'
            print(f'  警告: 无法检测时间参考，使用默认值: {time_ref}')

    src_time = _get_time(source_file, source_kwargs.get('time_var'))
    if init_date is not None:
        idx = _match_idx(src_time, init_date)
        print(f"  init_date={init_date} → time index {idx} "
              f"(source time = {src_time[idx]:.0f} s since 1970)")
    else:
        idx = 0
        print(f"  using first time step (index 0)")

    src = _read_source(source_file, time_index=idx, **source_kwargs)
    h = metrics['h']; mask = metrics['mask_rho']
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, igrid=1)
    spval = 1e37

    zeta = horizontal_interp(src['lon_1d'], src['lat_1d'], src['zeta'],
                              metrics['lon_rho'], metrics['lat_rho'],
                              mask=mask, fill_value=0.0)
    zeta = np.nan_to_num(zeta, nan=0.0)
    temp_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['temp'],
                                metrics['lon_rho'], metrics['lat_rho'])
    salt_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['salt'],
                                metrics['lon_rho'], metrics['lat_rho'])
    u_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['u'],
                             metrics['lon_rho'], metrics['lat_rho'])
    v_h = horizontal_interp(src['lon_1d'], src['lat_1d'], src['v'],
                             metrics['lon_rho'], metrics['lat_rho'])

    temp_h = _fill_nan(temp_h)
    salt_h = _fill_nan(salt_h)
    u_h = _fill_nan(u_h)
    v_h = _fill_nan(v_h)

    temp = z_to_sigma(temp_h, src['depth'], z_r, fill_value=spval)
    salt = z_to_sigma(salt_h, src['depth'], z_r, fill_value=spval)
    u_rho = z_to_sigma(u_h, src['depth'], z_r, fill_value=spval)
    v_rho = z_to_sigma(v_h, src['depth'], z_r, fill_value=spval)

    angle = metrics.get('angle', np.zeros_like(metrics['lat_rho']))
    u_rho, v_rho = rotate_uv(u_rho, v_rho, 0.0, angle)
    u, v = uv_to_cgrid(u_rho, v_rho, metrics.get('mask_u'),
                       metrics.get('mask_v'), spval)
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, igrid=5)
    ubar, vbar = compute_ubar_vbar(u, v, z_w, metrics.get('mask_u'),
                                   metrics.get('mask_v'))

    # 计算 ocean_time
    if src_time is not None:
        ref = _parse_date(time_ref)
        ocean_time = np.array([src_time[idx] - ref])
    else:
        ocean_time = np.array([0.0])

    try:
        from ..grid import stretching
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from grid import stretching
    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, kgrid=0)
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, kgrid=1)
    vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                    'theta_s': theta_s, 'theta_b': theta_b,
                    'Tcline': Tcline, 'hc': Tcline,
                    's_rho': s_rho, 'Cs_r': Cs_r,
                    's_w': s_w, 'Cs_w': Cs_w}
    write_ic_file(ini_file, metrics, vgrid_params, ocean_time,
                  zeta, temp, salt, u, v, ubar, vbar,
                  time_ref=time_ref)
    print(f"Written: {ini_file}")
    return {'zeta': zeta, 'temp': temp, 'salt': salt, 'u': u, 'v': v}


def cmems_to_roms_ini(roms_grid_file=None, zeta_file=None, temp_file=None,
                      salt_file=None, u_file=None, v_file=None,
                      data_dir=None, ini_file=None,
                      zeta_var='zos', temp_var='thetao', salt_var='so',
                      u_var='uo', v_var='vo',
                      Vtransform=2, Vstretching=3,
                      theta_s=2.5, theta_b=1.0, Tcline=25.0, N=30,
                      time_ref='1990-01-01', init_date='2025-05-01',
                      time_index=0):
    """
    Create ROMS initial conditions from CMEMS files.

    Parameters
    ----------
    roms_grid_file : str
        ROMS 网格文件路径
    ini_file : str
        输出初始场文件路径
    data_dir : str, optional
        CMEMS 数据目录（自动搜索文件）
    zeta_file, temp_file, salt_file, u_file, v_file : str, optional
        各变量的文件路径（如果不指定，则在 data_dir 中搜索）
    zeta_var, temp_var, salt_var, u_var, v_var : str
        文件中的变量名
    Vtransform, Vstretching : int
        垂直变换参数
    theta_s, theta_b, Tcline : float
        S 坐标参数
    N : int
        垂直层数
    time_ref : str, optional
        时间参考日期，自动从 CMEMS 文件时间单位读取
    init_date : str
        指定制作哪一天的 IC，如 '2025-05-01'
    time_index : int
        CMEMS 文件中的时间步索引（如果指定了 init_date 则自动计算）
    """
    try:
        from ..grid import set_depth, stretching
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from grid import set_depth, stretching

    print('\nReading CMEMS data ...\n')

    # 自动检测文件
    if data_dir and not all([zeta_file, temp_file, salt_file, u_file, v_file]):
        print(f'  搜索 CMEMS 文件于: {data_dir}')
        detected = _detect_cmems_files(data_dir)
        if not detected:
            raise FileNotFoundError(f'未在 {data_dir} 找到 CMEMS 文件')

        zeta_file = zeta_file or detected.get('zos')
        temp_file = temp_file or detected.get('thetao')
        salt_file = salt_file or detected.get('so')
        u_file = u_file or detected.get('uo')
        v_file = v_file or detected.get('vo')

        print(f'  找到文件:')
        for var, fpath in detected.items():
            print(f'    {var}: {os.path.basename(fpath)}')

    # 检查文件存在
    for name, path in [('zeta', zeta_file), ('temp', temp_file), ('salt', salt_file),
                       ('u', u_file), ('v', v_file)]:
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f'{name} 文件不存在: {path}')

    metrics = _read_roms_grid(roms_grid_file)
    h = metrics['h']
    mask_rho = metrics['mask_rho']
    mask_u = metrics.get('mask_u', np.ones_like(mask_rho))
    mask_v = metrics.get('mask_v', np.ones_like(mask_rho))
    angle = metrics.get('angle', np.zeros_like(mask_rho))
    hmax = np.max(h)

    def _read_var(file, var, is_zeta=False):
        """读取变量，zeta 文件可能无深度"""
        src = _read_cmems_file(file, var, time_index)
        return src['data'], src.get('depth'), src.get('lon_1d'), src.get('lat_1d')

    Zeta, _, Tlon_1d, Tlat_1d = _read_var(zeta_file, zeta_var, is_zeta=True)
    Temp, Tdepth, _, _ = _read_var(temp_file, temp_var)
    Salt, _, _, _ = _read_var(salt_file, salt_var)
    Uvel, Udepth, Ulon_1d, Ulat_1d = _read_var(u_file, u_var)
    Vvel, Vdepth, _, _ = _read_var(v_file, v_var)

    print('\n  Filling NaN in CMEMS data with 2D nearest neighbor...')

    Temp = _fill_nan_2d_range_check(Temp, 'temp')
    Salt = _fill_nan_2d_range_check(Salt, 'salt')
    Uvel = _fill_nan_2d_range_check(Uvel, 'vel')
    Vvel = _fill_nan_2d_range_check(Vvel, 'vel')
    Zeta = _fill_nan_2d_range_check(Zeta, 'zeta')

    print(f'  After fill: Temp NaN={np.sum(np.isnan(Temp))}, '
          f'Zeta NaN={np.sum(np.isnan(Zeta))}')

    Tdepth = np.unique(Tdepth[Tdepth < 0])
    Udepth = np.unique(Udepth[Udepth < 0])
    Vdepth = np.unique(Vdepth[Vdepth < 0])

    Tdepth = np.sort(Tdepth)[::-1]
    Udepth = np.sort(Udepth)[::-1]
    Vdepth = np.sort(Vdepth)[::-1]

    print(f'  CMEMS max depth: {np.abs(Tdepth[-1]):.0f} m, '
          f'ROMS max depth: {hmax:.0f} m')

    Temp = Temp[:len(Tdepth), :, :]
    Salt = Salt[:len(Tdepth), :, :]
    Uvel = Uvel[:len(Udepth), :, :]
    Vvel = Vvel[:len(Vdepth), :, :]

    Tdepth, Temp = _extend_depth(Tdepth, Temp, hmax, 'scalar')
    _, Salt = _extend_depth(Tdepth, Salt, hmax, 'scalar')
    Udepth, Uvel = _extend_depth(Udepth, Uvel, hmax, 'velocity')
    Vdepth, Vvel = _extend_depth(Vdepth, Vvel, hmax, 'velocity')

    print('  Filling NaN vertically...')
    Temp = _fill_nans_vertically(Temp)
    Salt = _fill_nans_vertically(Salt)
    Uvel = _fill_nans_vertically(Uvel)
    Vvel = _fill_nans_vertically(Vvel)

    print(f'  Depth levels: T/S={len(Tdepth)}, U={len(Udepth)}, '
          f'V={len(Vdepth)}')

    Rmask3d = np.ones_like(Temp)
    Rmask3d[np.isnan(Temp) | (Temp == 0)] = 0

    print(f'\n  SSH range: {np.nanmin(Zeta):.4f} to {np.nanmax(Zeta):.4f}')
    print(f'  Temp range: {np.nanmin(Temp):.4f} to {np.nanmax(Temp):.4f}')
    print(f'  Salt range: {np.nanmin(Salt):.4f} to {np.nanmax(Salt):.4f}')

    # 自动检测时间参考
    if time_ref is None:
        time_ref = _detect_cmems_time_ref(temp_file)
        if time_ref:
            print(f'  自动检测时间参考: {time_ref}')
        else:
            time_ref = '1970-01-01'
            print(f'  警告: 无法检测时间参考，使用默认值: {time_ref}')

    # 使用 init_date 自动选择时间步
    if init_date is not None and temp_src.get('time_units'):
        # 从 CMEMS 文件读取时间信息
        try:
            ds = nc4.Dataset(temp_file, 'r')
            time_name = _find_dim_name(ds, ['time', 'time_counter', 't'])
            if time_name:
                time_var = ds.variables[time_name]
                time_units = getattr(time_var, 'units', '')
                time_values = time_var[:]
                ds.close()

                # 解析时间
                m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', time_units)
                if m:
                    ref_str = m.group(1).strip()
                    try:
                        ref_dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            ref_dt = datetime.strptime(ref_str, '%Y-%m-%d')
                        except ValueError:
                            ref_dt = datetime(1950, 1, 1)

                    # 转换时间值
                    time_seconds = time_values * 86400.0  # assuming days
                    time_dates = [ref_dt + timedelta(seconds=float(s)) for s in time_seconds]

                    # 查找匹配的日期
                    init_dt = datetime.strptime(init_date, '%Y-%m-%d')
                    best_idx = 0
                    best_diff = abs((time_dates[0] - init_dt).total_seconds())
                    for i, td in enumerate(time_dates):
                        diff = abs((td - init_dt).total_seconds())
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = i

                    time_index = best_idx
                    print(f'  init_date={init_date} → time index {time_index} ({time_dates[time_index]})')
            else:
                ds.close()
        except Exception as e:
            print(f'  警告: 自动选择时间步失败: {e}')

    ref_dt = datetime.strptime(time_ref, '%Y-%m-%d')
    init_dt = datetime.strptime(init_date, '%Y-%m-%d')
    delta = init_dt - ref_dt
    ocean_time = delta.total_seconds()

    print(f'\n  Time reference: {time_ref}')
    print(f'  Initial date: {init_date}')
    print(f'  ocean_time: {ocean_time:.0f} seconds')

    print('\nInterpolating to ROMS grid ...\n')

    if Temp.ndim == 3:
        if Temp.shape[1] == len(Tlat_1d) and Temp.shape[2] == len(Tlon_1d):
            Tlon_2d, Tlat_2d = np.meshgrid(Tlon_1d, Tlat_1d)
            Ulon_2d, Ulat_2d = np.meshgrid(Ulon_1d, Ulat_1d)
        else:
            Tlat_2d, Tlon_2d = np.meshgrid(Tlat_1d, Tlon_1d)
            Ulat_2d, Ulon_2d = np.meshgrid(Ulat_1d, Ulon_1d)
            Temp = np.transpose(Temp, (0, 2, 1))
            Salt = np.transpose(Salt, (0, 2, 1))
    else:
        Tlon_2d, Tlat_2d = np.meshgrid(Tlon_1d, Tlat_1d)
        Ulon_2d, Ulat_2d = np.meshgrid(Ulon_1d, Ulat_1d)

    if Zeta.ndim == 2:
        if Zeta.shape[0] == len(Tlat_1d) and Zeta.shape[1] == len(Tlon_1d):
            pass
        else:
            Zeta = Zeta.T

    zeta = horizontal_interp(Tlon_1d, Tlat_1d, Zeta,
                             metrics['lon_rho'], metrics['lat_rho'],
                             mask=mask_rho, fill_value=0.0)
    zeta = np.nan_to_num(zeta, nan=0.0)

    print('  Interpolating temperature ...')
    temp = horizontal_interp(Tlon_1d, Tlat_1d, Temp,
                             metrics['lon_rho'], metrics['lat_rho'],
                             mask=mask_rho)

    print('  Interpolating salinity ...')
    salt = horizontal_interp(Tlon_1d, Tlat_1d, Salt,
                             metrics['lon_rho'], metrics['lat_rho'],
                             mask=mask_rho)

    print('  Interpolating u-velocity ...')
    Urho = horizontal_interp(Ulon_1d, Ulat_1d, Uvel,
                             metrics['lon_rho'], metrics['lat_rho'],
                             mask=mask_rho)

    print('  Interpolating v-velocity ...')
    Vrho = horizontal_interp(Ulon_1d, Ulat_1d, Vvel,
                             metrics['lon_rho'], metrics['lat_rho'],
                             mask=mask_rho)

    print('\n  Performing vertical interpolation ...')

    ssh = np.zeros_like(zeta)
    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, zeta=ssh, igrid=1)

    spval = 1e37

    temp = z_to_sigma(temp, Tdepth, z_r, fill_value=spval)
    salt = z_to_sigma(salt, Tdepth, z_r, fill_value=spval)
    Urho = z_to_sigma(Urho, Udepth, z_r, fill_value=spval)
    Vrho = z_to_sigma(Vrho, Vdepth, z_r, fill_value=spval)

    print('  Rotating velocity vectors ...')

    Urho, Vrho = rotate_uv(Urho, Vrho, 0.0, angle)

    u, v = uv_to_cgrid(Urho, Vrho, mask_u, mask_v, spval)

    print('  Computing barotropic velocities ...')

    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, zeta=zeta, igrid=5)
    ubar, vbar = compute_ubar_vbar(u, v, z_w, mask_u, mask_v)

    print('\nApplying masks: land -> fill value ...')

    wm3d_rho = np.tile(mask_rho, (N, 1, 1))
    temp[wm3d_rho == 0] = spval
    salt[wm3d_rho == 0] = spval
    zeta[mask_rho == 0] = spval

    wm3d_u = np.tile(mask_u, (N, 1, 1))
    u[wm3d_u == 0] = spval
    ubar[mask_u == 0] = spval

    wm3d_v = np.tile(mask_v, (N, 1, 1))
    v[wm3d_v == 0] = spval
    vbar[mask_v == 0] = spval

    water_mask = wm3d_rho == 1
    if np.any(water_mask):
        print(f'  Water points: temp min={np.min(temp[water_mask]):.4f} '
              f'max={np.max(temp[water_mask]):.4f}')

    print('\nWriting initial conditions ...\n')

    s_rho, Cs_r = stretching(Vstretching, theta_s, theta_b, N, kgrid=0)
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, kgrid=1)

    vgrid_params = {
        'Vtransform': Vtransform,
        'Vstretching': Vstretching,
        'theta_s': theta_s,
        'theta_b': theta_b,
        'Tcline': Tcline,
        'hc': Tcline,
        's_rho': s_rho,
        'Cs_r': Cs_r,
        's_w': s_w,
        'Cs_w': Cs_w
    }

    ocean_time_arr = np.array([ocean_time])

    write_ic_file(ini_file, metrics, vgrid_params, ocean_time_arr,
                  zeta, temp, salt, u, v, ubar, vbar,
                  time_ref=time_ref)

    ds = nc4.Dataset(ini_file, 'a')
    ds.title = 'CMEMS Global Ocean Physics Analysis and Forecast, initial condition'
    ds.source = 'CMEMS reanalysis'
    ds.grd_file = roms_grid_file
    ds.close()

    print('=' * 60)
    print('Initial condition file created successfully!')
    print(f'Output: {ini_file}')
    print('=' * 60)

    return {
        'zeta': zeta,
        'temp': temp,
        'salt': salt,
        'u': u,
        'v': v,
        'ubar': ubar,
        'vbar': vbar
    }
