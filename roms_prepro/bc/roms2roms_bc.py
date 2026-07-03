"""
ROMS → ROMS boundary conditions via intermediate standard-z grid.

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
                    write_bry_file, EDGE_NAMES,
                    _fill_nan, _parse_date, _get_time, _match_idx,
                    _filter_files, _read_roms_grid)

DEFAULT_Z = np.array([
    -7500, -7000, -6500, -6000, -5500, -5000, -4500, -4000, -3500,
    -3000, -2500, -2000, -1750, -1500, -1250, -1000, -900, -800, -700,
    -600, -500, -400, -300, -250, -200, -175, -150, -125, -100, -90,
    -80, -70, -60, -50, -45, -40, -35, -30, -25, -20, -17.5,
    -15, -12.5, -10, -7.5, -5, -2.5, 0
])


def _get_var(filename, varname, time_idx=0):
    """Read one variable from a ROMS file."""
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

        m = re.search(r'since\s+([\d\-]+\s*[\d:]*)', units)
        if m:
            ref_str = m.group(1).strip()
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


def _detect_roms_files(src_hist_files=None, src_hist_dir=None):
    """
    自动检测 ROMS 历史文件。

    返回排序后的文件路径列表。
    """
    if src_hist_files:
        if isinstance(src_hist_files, str):
            if os.path.isfile(src_hist_files):
                return [src_hist_files]
            elif os.path.isdir(src_hist_files):
                src_hist_dir = src_hist_files
            else:
                return []
        elif isinstance(src_hist_files, (list, tuple)):
            return [f for f in src_hist_files if os.path.isfile(f)]

    if src_hist_dir and os.path.isdir(src_hist_dir):
        patterns = ['*.nc', 'ocean_his_*.nc', 'ocean_avg_*.nc', 'roms_his_*.nc']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(src_hist_dir, pattern)))
        if files:
            return sorted(set(files))

    return []


def roms_to_roms_bry(src_grid_file, src_hist_files=None, dst_grid_file=None,
                     bry_file=None,
                     Vtransform=2, Vstretching=4,
                     theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
                     boundaries=(False, True, True, True),
                     z_levels=None, var_mapping=None,
                     start_date=None, end_date=None,
                     time_ref=None,
                     src_hist_dir=None):
    """
    Create ROMS time-dependent BC by remapping from another ROMS run.

    Parameters
    ----------
    src_hist_files : str or list, optional
        源 ROMS 历史文件路径或路径列表
    src_hist_dir : str, optional
        源 ROMS 历史文件目录（自动搜索）
    start_date, end_date : str, optional
        时间范围过滤，自动从文件时间读取
    time_ref : str, optional
        时间参考，自动从源文件 ocean_time 单位读取
    """
    from ..grid import set_depth
    from datetime import datetime

    if z_levels is None:
        z_levels = DEFAULT_Z

    # 自动检测文件
    detected_files = _detect_roms_files(src_hist_files, src_hist_dir)
    if not detected_files:
        raise FileNotFoundError('未找到 ROMS 历史文件')

    src_hist_files = detected_files

    # 自动检测时间参考
    if time_ref is None:
        time_ref = _detect_roms_time_ref(src_hist_files[0])
        if time_ref:
            print(f'  自动检测时间参考: {time_ref}')
        else:
            time_ref = '1970-01-01'
            print(f'  警告: 无法检测时间参考，使用默认值: {time_ref}')

    filtered = _filter_files(src_hist_files, start_date, end_date,
                             'ocean_time')
    if not filtered:
        raise ValueError("No source files in specified time range.")

    src_grd = _read_roms_grid(src_grid_file)
    dst_grd = _read_roms_grid(dst_grid_file)

    src_Vt = int(src_grd.get('Vtransform', 1))
    src_Vs = int(src_grd.get('Vstretching', 1))
    src_N = 30
    if filtered:
        h = nc4.Dataset(filtered[0][0])
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

    dst_z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=5)
    dst_z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline,
                        N, dst_grd['h'], igrid=1)

    spval = 1e37

    vmap = var_mapping or {}
    v = lambda k: vmap.get(k, k)

    parent_angle = _remap_var(v('angle'), src_grid_file, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, 0, spval)
    target_angle = dst_grd.get('angle', np.zeros_like(dst_grd['lat_rho']))

    m = re.search(r'since\s+(.+)', time_ref) if time_ref else None
    ref_epoch = _parse_date(m.group(1)) if m else 0.0

    total_steps = sum(len(idxs) for _, idxs in filtered)
    bry_time = np.zeros(total_steps)
    t = 0

    for fp, idxs in filtered:
        src_t = _get_time(fp, 'ocean_time')
        for idx in idxs:
            ti = src_t[idx] if src_t is not None else float(t)
            bry_sec = ti - ref_epoch if ref_epoch != 0 else ti
            print(f"\n[{t + 1}/{total_steps}] {os.path.basename(fp)} "
                  f"idx={idx}, time={ti:.0f}s since 1970")

            temp = _remap_var(v('temp'), fp, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, idx, spval)
            salt = _remap_var(v('salt'), fp, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, idx, spval)
            zeta = _remap_var(v('zeta'), fp, src_grd, src_z_r, src_z_w,
                              dst_grd, dst_z_r, dst_z_w, z_levels, idx, spval)
            u_rho = _remap_var(v('u'), fp, src_grd, src_z_r, src_z_w,
                               dst_grd, dst_z_r, dst_z_w, z_levels, idx, spval)
            v_rho = _remap_var(v('v'), fp, src_grd, src_z_r, src_z_w,
                               dst_grd, dst_z_r, dst_z_w, z_levels, idx, spval)

            u_rho, v_rho = rotate_uv(u_rho, v_rho, parent_angle, target_angle)
            u, v = uv_to_cgrid(u_rho, v_rho, dst_grd.get('mask_u'),
                               dst_grd.get('mask_v'), spval)
            ubar, vbar = compute_ubar_vbar(u, v, dst_z_w,
                                           dst_grd.get('mask_u'),
                                           dst_grd.get('mask_v'))

            edge_data = {}
            for name in ['temp', 'salt']:
                f3d = locals()[name]
                edge_data[name] = [None] * 4
                for b, active in enumerate(boundaries):
                    if not active:
                        continue
                    edge = EDGE_NAMES[b]
                    if edge in ('west', 'east'):
                        ix = 0 if edge == 'west' else -1
                        edge_data[name][b] = f3d[:, :, ix]
                    else:
                        ix = 0 if edge == 'south' else -1
                        edge_data[name][b] = f3d[:, ix, :]

            for name, f3d, vname in [('u', u, 'u'), ('v', v, 'v')]:
                edge_data[vname] = [None] * 4
                for b, active in enumerate(boundaries):
                    if not active:
                        continue
                    edge = EDGE_NAMES[b]
                    if edge in ('west', 'east'):
                        ix = 0 if edge == 'west' else -1
                        edge_data[vname][b] = f3d[:, :, ix]
                    else:
                        ix = 0 if edge == 'south' else -1
                        edge_data[vname][b] = f3d[:, ix, :]

            for name, f2d in [('zeta', zeta), ('ubar', ubar), ('vbar', vbar)]:
                edge_data[name] = [None] * 4
                for b, active in enumerate(boundaries):
                    if not active:
                        continue
                    edge = EDGE_NAMES[b]
                    if edge in ('west', 'east'):
                        ix = 0 if edge == 'west' else -1
                        edge_data[name][b] = f2d[:, ix]
                    else:
                        ix = 0 if edge == 'south' else -1
                        edge_data[name][b] = f2d[ix, :]

            src_time = _get_var(fp, 'ocean_time', 0)
            bry_time[t] = float(np.atleast_1d(src_time)[0])

            # 陆地点设为 fill_value（与 cmems_bc 一致）
            fill_val = 1e37
            for b, active in enumerate(boundaries):
                if not active:
                    continue
                edge = EDGE_NAMES[b]

                # 获取对应边界的 mask
                if edge in ('west', 'east'):
                    bm = dst_grd.get('mask_rho', np.ones_like(dst_grd['h']))[:, 0 if edge == 'west' else -1]
                else:
                    bm = dst_grd.get('mask_rho', np.ones_like(dst_grd['h']))[0 if edge == 'south' else -1, :]

                for vn in edge_data:
                    if edge_data[vn][b] is None:
                        continue
                    if vn in ('zeta', 'ubar', 'vbar'):
                        edge_data[vn][b][bm == 0] = fill_val
                    else:
                        edge_data[vn][b][bm == 0] = fill_val

            if t == 0:
                vgrid_params = {'Vtransform': Vtransform, 'Vstretching': Vstretching,
                                'theta_s': theta_s, 'theta_b': theta_b,
                                'Tcline': Tcline, 'hc': Tcline}
                write_bry_file(bry_file, dst_grd, vgrid_params, boundaries,
                               edge_data, bry_time[:t + 1])
            else:
                ds = nc4.Dataset(bry_file, 'a')
                ds.variables['bry_time'][t] = bry_time[t]
                for var_name in edge_data:
                    for b, active in enumerate(boundaries):
                        if not active or edge_data[var_name][b] is None:
                            continue
                        edge = EDGE_NAMES[b]
                        key = f'{var_name}_{edge}'
                        ds.variables[key][t] = edge_data[var_name][b]
                ds.close()

            t += 1

    print(f"\nWritten: {bry_file}")
