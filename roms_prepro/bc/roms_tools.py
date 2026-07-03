"""
ROMS Python Tools - 边界条件制作工具箱
实现 stretching, set_depth, obc_mercator, roms_vectors, uv_barotropic 等函数
"""

import numpy as np
from scipy.interpolate import griddata, interp1d, LinearNDInterpolator
from scipy.ndimage import distance_transform_edt


# ============================================================
# 垂直坐标函数
# ============================================================

def stretching(Vstretching, theta_s, theta_b, hc, N, kgrid):
    """
    ROMS垂直拉伸函数
    
    Parameters
    ----------
    Vstretching : int
        拉伸函数类型 (1-5)
    theta_s : float
        表面控制参数
    theta_b : float
        底部控制参数
    hc : float
        拉伸宽度
    N : int
        垂直层数
    kgrid : int
        0=RHO-points, 1=W-points
    
    Returns
    -------
    s : ndarray
        垂直拉伸坐标
    Cs : ndarray
        垂直拉伸函数
    """
    if kgrid == 0:
        # RHO-points
        s = -1.0 + (np.arange(1, N+1) - 0.5) / N
    else:
        # W-points
        s = -1.0 + np.arange(0, N+1) / N

    if Vstretching == 1:
        # Song and Haidvogel (1994)
        if theta_s > 0:
            Cs = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        else:
            Cs = -s**2
    elif Vstretching == 2:
        # Shchepetkin and McWilliams (2005)
        if theta_s > 0:
            Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        else:
            Csur = -s**2
        if theta_b > 0:
            Cbot = (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
            Cs = Cbot
        else:
            Cs = Csur
    elif Vstretching == 3:
        # Shchepetkin (2008) - UCLA-ROMS
        if theta_s > 0:
            Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        else:
            Csur = -s**2
        if theta_b > 0:
            Cbot = (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
            Cs = Cbot
        else:
            Cs = Csur
    elif Vstretching == 4:
        # A. Shchepetkin (2010) - UCLA-ROMS
        if theta_s > 0:
            Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        else:
            Csur = -s**2
        if theta_b > 0:
            Cbot = (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
            Cs = Cbot
        else:
            Cs = Csur
    elif Vstretching == 5:
        # Geyer et al. (2009)
        if theta_s > 0:
            Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        else:
            Csur = -s**2
        if theta_b > 0:
            Cbot = (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
            Cs = Cbot
        else:
            Cs = Csur
    else:
        raise ValueError(f"Unknown Vstretching: {Vstretching}")

    return s, Cs


def set_depth(Vtransform, Vstretching, theta_s, theta_b, hc, N, igrid, h, ssh):
    """
    计算ROMS垂直坐标深度
    
    Parameters
    ----------
    Vtransform : int
        垂直变换方程 (1 or 2)
    Vstretching : int
        垂直拉伸函数
    theta_s, theta_b, hc : float
        S坐标参数
    N : int
        垂直层数
    igrid : int
        网格类型 (1=RHO, 3=U, 4=V, 5=W)
    h : ndarray
        水深
    ssh : ndarray
        海面高度
    
    Returns
    -------
    z : ndarray
        3D深度数组
    """
    s, Cs = stretching(Vstretching, theta_s, theta_b, hc, N, igrid)
    
    h = np.atleast_2d(h)
    ssh = np.atleast_2d(ssh)
    
    # 对U/V网格进行交错平均
    if igrid == 3:  # U-points: xi方向交错
        h = 0.5 * (h[:, :-1] + h[:, 1:])
        ssh = 0.5 * (ssh[:, :-1] + ssh[:, 1:])
    elif igrid == 4:  # V-points: eta方向交错
        h = 0.5 * (h[:-1, :] + h[1:, :])
        ssh = 0.5 * (ssh[:-1, :] + ssh[1:, :])
    
    Lp, Mp = h.shape
    
    if igrid == 5:
        # W-points
        Nlev = N + 1
    else:
        # RHO-points
        Nlev = N
    
    z = np.zeros((Lp, Mp, Nlev))
    
    for k in range(Nlev):
        if Vtransform == 1:
            z0 = hc * (s[k] - Cs[k]) + h * Cs[k]
            z[:, :, k] = z0 + ssh * (1.0 + z0 / h)
        elif Vtransform == 2:
            z0 = (hc * s[k] + h * Cs[k]) / (hc + h)
            z[:, :, k] = ssh + (ssh + h) * z0
    
    return z


# ============================================================
# 边界插值函数
# ============================================================

def obc_mercator(Vname, S, Finp, lon, lat, mask, depth=None):
    """
    将CMEMS数据插值到ROMS边界
    
    Parameters
    ----------
    Vname : str
        变量名 ('zeta', 'temp', 'salt', 'u', 'v')
    S : dict
        ROMS网格结构
    Finp : ndarray
        输入数据 (2D或3D)
    lon, lat : ndarray
        输入数据经纬度 (2D)
    mask : ndarray
        输入数据掩膜 (2D或3D)
    depth : ndarray, optional
        输入数据深度 (1D, 仅3D变量需要)
    
    Returns
    -------
    Fout : dict
        边界数据 {'west': ..., 'east': ..., 'south': ..., 'north': ...}
    """
    is2d = (Vname == 'zeta')
    
    # 获取ROMS网格坐标
    rlon = S['lon_rho']
    rlat = S['lat_rho']
    rmask = S['mask_rho']
    
    if not is2d:
        if Vname == 'u':
            rlon = S['lon_u']
            rlat = S['lat_u']
            rmask = S['mask_u']
            z_r = S['z_u']
        elif Vname == 'v':
            rlon = S['lon_v']
            rlat = S['lat_v']
            rmask = S['mask_v']
            z_r = S['z_v']
        else:
            z_r = S['z_r']
    
    # 裁剪CMEMS数据到ROMS范围
    ii = np.where((lon[:, 0] > (rlon.min() - 1)) & (lon[:, 0] < (rlon.max() + 1)))[0]
    jj = np.where((lat[0, :] > (rlat.min() - 1)) & (lat[0, :] < (rlat.max() + 1)))[0]
    
    x = lon[np.ix_(ii, jj)]
    y = lat[np.ix_(ii, jj)]
    
    Fout = {}
    
    if is2d:
        # 2D插值
        M = mask[np.ix_(ii, jj)]
        wet = np.where(M.ravel() > 0)[0]
        F = Finp[np.ix_(ii, jj)]
        
        src_points = np.column_stack([x.ravel()[wet], y.ravel()[wet]])
        src_values = F.ravel()[wet]
        target_points = np.column_stack([rlon.ravel(), rlat.ravel()])
        
        try:
            interp = LinearNDInterpolator(src_points, src_values)
            Fwrk = interp(target_points).reshape(rlon.shape)
        except Exception:
            Fwrk = griddata(src_points, src_values, (rlon, rlat), method='linear')
        
        Fwrk[np.isnan(Fwrk) & (rmask == 0)] = 0
        
        # 替换NaN为最近邻
        Fwrk = _fillnan_boundary(Fwrk, rmask, rlon, rlat, S['boundary'])
        
        # 提取边界
        Fout = _extract_boundary_2d(Fwrk, S['boundary'], rmask)
    
    else:
        # 3D插值
        # Finp shape: (depth, lat, lon)
        Nlev = Finp.shape[0]
        
        # 确定有效层数
        for k in range(Nlev-1, 0, -1):
            if np.all(Finp[k, :, :] == 0):
                Nlev = k
        
        # 预计算插值器（只构建一次三角剖分）
        # 找到所有层共有的wet点
        M_all = mask[:Nlev, :, :][:, ii, :][:, :, jj]
        wet_all = np.where(np.all(M_all > 0, axis=0).ravel())[0]
        
        if len(wet_all) == 0:
            # 如果没有共同wet点，逐层处理
            wet_all = None
        
        # 水平插值到ROMS网格
        Flev = np.zeros((rlon.shape[0], rlon.shape[1], Nlev))
        
        x_sub = x.ravel()
        y_sub = y.ravel()
        target_points = np.column_stack([rlon.ravel(), rlat.ravel()])
        
        for k in range(Nlev):
            M = mask[k, :, :][np.ix_(ii, jj)]
            wet = np.where(M.ravel() > 0)[0]
            
            if len(wet) == 0:
                continue
            
            F = Finp[k, :, :][np.ix_(ii, jj)]
            
            # 使用LinearNDInterpolator（比griddata快很多）
            src_points = np.column_stack([x_sub[wet], y_sub[wet]])
            src_values = F.ravel()[wet]
            
            try:
                interp = LinearNDInterpolator(src_points, src_values)
                Fwrk = interp(target_points).reshape(rlon.shape)
            except Exception:
                Fwrk = griddata(src_points, src_values, (rlon, rlat), method='linear')
            
            Fwrk[np.isnan(Fwrk) & (rmask == 0)] = 0
            
            # 替换NaN为最近邻
            Fwrk = _fillnan_boundary(Fwrk, rmask, rlon, rlat, S['boundary'])
            
            Flev[:, :, k] = Fwrk
        
        # 垂直插值到ROMS S坐标
        depth_neg = -np.abs(depth[:Nlev])
        depth_neg[0] = 0.0
        
        Fout = _vertical_interp_boundary(Flev, depth_neg, z_r, rmask, S['boundary'], Vname)
    
    return Fout


def _fillnan_boundary(Fwrk, rmask, rlon, rlat, boundary):
    """用最近邻替换NaN"""
    nan_mask = np.isnan(Fwrk) & (rmask > 0)
    if not np.any(nan_mask):
        return Fwrk
    
    valid = ~np.isnan(Fwrk) & (rmask > 0)
    if not np.any(valid):
        return Fwrk
    
    _, idx = distance_transform_edt(~valid, return_indices=True)
    Fwrk[nan_mask] = Fwrk[idx[0][nan_mask], idx[1][nan_mask]]
    
    return Fwrk


def _extract_boundary_2d(Fwrk, boundary, rmask):
    """从2D场提取边界"""
    Fout = {}
    
    if boundary[0]:  # west
        Fout['west'] = Fwrk[0, :]
    if boundary[1]:  # east
        Fout['east'] = Fwrk[-1, :]
    if boundary[2]:  # south
        Fout['south'] = Fwrk[:, 0]
    if boundary[3]:  # north
        Fout['north'] = Fwrk[:, -1]
    
    return Fout


def _vertical_interp_boundary(Flev, depth, z_r, rmask, boundary, Vname):
    """垂直插值到ROMS边界"""
    Im, Jm, N = z_r.shape
    Fout = {}
    
    for ib, bname in enumerate(['west', 'east', 'south', 'north']):
        if not boundary[ib]:
            continue
        
        if ib == 0:  # west
            if Vname in ('u', 'v'):
                bmask = rmask[0:2, :]
                z = z_r[0:2, :, :]
                Fobc = np.zeros_like(z)
                for j in range(Jm):
                    for i in range(2):
                        if bmask[i, j] > 0:
                            Fobc[i, j, :] = interp1(depth, Flev[0, j, :], z[i, j, :], 
                                                      bounds_error=False, fill_value=0)
            else:
                bmask = rmask[0, :]
                z = z_r[0, :, :]
                Fobc = np.zeros_like(z)
                for j in range(Jm):
                    if bmask[j] > 0:
                        Fobc[j, :] = interp1(depth, Flev[0, j, :], z[j, :], 
                                              bounds_error=False, fill_value=0)
        
        elif ib == 1:  # east
            if Vname in ('u', 'v'):
                bmask = rmask[-2:, :]
                z = z_r[-2:, :, :]
                Fobc = np.zeros_like(z)
                for j in range(Jm):
                    for i in range(2):
                        if bmask[i, j] > 0:
                            Fobc[i, j, :] = interp1(depth, Flev[-1, j, :], z[i, j, :], 
                                                      bounds_error=False, fill_value=0)
            else:
                bmask = rmask[-1, :]
                z = z_r[-1, :, :]
                Fobc = np.zeros_like(z)
                for j in range(Jm):
                    if bmask[j] > 0:
                        Fobc[j, :] = interp1(depth, Flev[-1, j, :], z[j, :], 
                                              bounds_error=False, fill_value=0)
        
        elif ib == 2:  # south
            if Vname in ('u', 'v'):
                bmask = rmask[:, 0:2]
                z = z_r[:, 0:2, :]
                Fobc = np.zeros_like(z)
                for i in range(Im):
                    for j in range(2):
                        if bmask[i, j] > 0:
                            Fobc[i, j, :] = interp1(depth, Flev[i, 0, :], z[i, j, :], 
                                                      bounds_error=False, fill_value=0)
            else:
                bmask = rmask[:, 0]
                z = z_r[:, 0, :]
                Fobc = np.zeros_like(z)
                for i in range(Im):
                    if bmask[i] > 0:
                        Fobc[i, :] = interp1(depth, Flev[i, 0, :], z[i, :], 
                                              bounds_error=False, fill_value=0)
        
        elif ib == 3:  # north
            if Vname in ('u', 'v'):
                bmask = rmask[:, -2:]
                z = z_r[:, -2:, :]
                Fobc = np.zeros_like(z)
                for i in range(Im):
                    for j in range(2):
                        if bmask[i, j] > 0:
                            Fobc[i, j, :] = interp1(depth, Flev[i, -1, :], z[i, j, :], 
                                                      bounds_error=False, fill_value=0)
            else:
                bmask = rmask[:, -1]
                z = z_r[:, -1, :]
                Fobc = np.zeros_like(z)
                for i in range(Im):
                    if bmask[i] > 0:
                        Fobc[i, :] = interp1(depth, Flev[i, -1, :], z[i, :], 
                                              bounds_error=False, fill_value=0)
        
        Fout[bname] = Fobc
    
    return Fout


# ============================================================
# 速度处理函数
# ============================================================

def roms_vectors(Urho, Vrho, angle, mask_u, mask_v, boundary):
    """
    将速度旋转到ROMS C-grid
    """
    u = {}
    v = {}
    
    for ib, bname in enumerate(['west', 'east', 'south', 'north']):
        if not boundary[ib]:
            continue
        
        if bname not in Urho:
            continue
        
        U = Urho[bname]
        V = Vrho[bname]
        
        # 获取对应边界的角度
        if ib == 0:  # west
            a = angle[0:2, :] if U.ndim == 3 else angle[0, :]
        elif ib == 1:  # east
            a = angle[-2:, :] if U.ndim == 3 else angle[-1, :]
        elif ib == 2:  # south
            a = angle[:, 0:2] if U.ndim == 3 else angle[:, 0]
        elif ib == 3:  # north
            a = angle[:, -2:] if U.ndim == 3 else angle[:, -1]
        
        # 旋转
        if U.ndim == 2:
            # 2D: (space, depth)
            if ib in [0, 1]:  # west/east: a shape (2, eta) or (eta,)
                u[bname] = U * np.cos(a[..., None]) + V * np.sin(a[..., None])
                v[bname] = V * np.cos(a[..., None]) - U * np.sin(a[..., None])
            else:  # south/north: a shape (xi, 2) or (xi,)
                u[bname] = U * np.cos(a[..., None]) + V * np.sin(a[..., None])
                v[bname] = V * np.cos(a[..., None]) - U * np.sin(a[..., None])
        else:
            # 3D: already handled above
            u[bname] = U * np.cos(a[..., None]) + V * np.sin(a[..., None])
            v[bname] = V * np.cos(a[..., None]) - U * np.sin(a[..., None])
    
    return u, v


def uv_barotropic(u, v, Hz, boundary):
    """
    计算正压速度（垂向平均）
    Hz在RHO网格上，内部自动交错到U/V网格
    
    Parameters
    ----------
    u, v : dict
        边界速度
    Hz : ndarray
        垂直层厚度 (Lp, Mp, N) - RHO网格
    boundary : list
        边界开关
    
    Returns
    -------
    ubar, vbar : dict
        正压速度
    """
    ubar = {}
    vbar = {}
    
    for ib, bname in enumerate(['west', 'east', 'south', 'north']):
        if not boundary[ib]:
            continue
        
        if bname not in u:
            continue
        
        u_3d = u[bname]
        v_3d = v[bname]
        
        if ib == 0:  # west: xi=0, eta方向
            # u在xi方向交错，取Hz的平均
            Hz_u = 0.5 * (Hz[0, :-1, :] + Hz[0, 1:, :])
            Hz_v = Hz[0, :, :]
        elif ib == 1:  # east: xi=end
            Hz_u = 0.5 * (Hz[-1, :-1, :] + Hz[-1, 1:, :])
            Hz_v = Hz[-1, :, :]
        elif ib == 2:  # south: eta=0
            Hz_u = Hz[:, 0, :]
            Hz_v = 0.5 * (Hz[:-1, 0, :] + Hz[1:, 0, :])
        elif ib == 3:  # north: eta=end
            Hz_u = Hz[:, -1, :]
            Hz_v = 0.5 * (Hz[:-1, -1, :] + Hz[1:, -1, :])
        
        # 垂向积分
        u_sum = np.sum(u_3d * Hz_u, axis=-1)
        v_sum = np.sum(v_3d * Hz_v, axis=-1)
        H_sum_u = np.sum(Hz_u, axis=-1)
        H_sum_v = np.sum(Hz_v, axis=-1)
        
        ubar[bname] = u_sum / H_sum_u
        vbar[bname] = v_sum / H_sum_v
    
    return ubar, vbar


# ============================================================
# NaN填充函数
# ============================================================

def fill_nan_2d(data):
    """
    用最近邻填充NaN (2D或3D)
    3D时逐层处理，避免distance_transform在3D上太慢
    """
    if data.ndim == 2:
        return _fill_nan_single(data)
    else:
        result = data.copy()
        for k in range(data.shape[0]):  # depth是第0维
            result[k] = _fill_nan_single(data[k])
        return result


def _fill_nan_single(layer):
    """填充单层NaN"""
    nan_mask = np.isnan(layer)
    if not np.any(nan_mask):
        return layer
    if np.all(nan_mask):
        return layer
    
    valid = ~nan_mask
    _, idx = distance_transform_edt(~valid, return_indices=True)
    result = layer.copy()
    result[nan_mask] = layer[idx[0][nan_mask], idx[1][nan_mask]]
    return result
