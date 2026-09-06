import numpy as np
import netCDF4 as nc
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
from matplotlib.path import Path
from tqdm import tqdm


DEFAULT_CONSTITUENTS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1']

TPXOHARMONICS = {
    'Q1':  np.array([1, -3,  1,  1,  0,  0, 270, 13.3986607]),
    'O1':  np.array([1, -2,  1,  0,  0,  0, 270, 13.9430351]),
    'P1':  np.array([1,  0, -1,  0,  0,  0, 270, 14.9589310]),
    'K1':  np.array([1,  0,  1,  0,  0,  0,  90, 15.0410690]),
    'N2':  np.array([2, -3,  2,  1,  0,  0,   0, 28.4397297]),
    'M2':  np.array([2, -2,  2,  0,  0,  0,   0, 28.9841042]),
    'S2':  np.array([2,  0,  0,  0,  0,  0,   0, 30.0000000]),
    'K2':  np.array([2,  0,  2,  0,  0,  0,   0, 30.0821381]),
}


def tpxo_to_roms_tide(roms_grid_file, out_file, t0, ndays=365,
                       tpxo_dir=None, constituents=None):
    """
    生成 ROMS 潮汐强迫文件（TPXO8 atlas_30 -> ROMS grid）。

    Parameters
    ----------
    roms_grid_file : str
        ROMS 网格文件路径（包含 lon_rho, lat_rho, mask_rho）。
    out_file : str
        输出潮汐强迫文件路径。
    t0 : datetime or str
        参考时间。datetime 对象或 ISO 格式字符串（如 '2000-01-01'）。
    ndays : int, optional
        模拟长度（天），用于计算 nodal 因子的参考时间 t0 + ndays/2。默认 365。
    tpxo_dir : str, optional
        TPXO 数据目录。默认与 roms_grid_file 同目录。
    constituents : list of str, optional
        分潮列表。默认 ['M2','S2','N2','K2','K1','O1','P1','Q1']。
    """
    # --- 参数处理 ---
    if isinstance(t0, str):
        t0 = datetime.fromisoformat(t0.replace(' ', 'T'))

    if constituents is None:
        constituents = [c for c in DEFAULT_CONSTITUENTS if c in TPXOHARMONICS]

    for c in constituents:
        if c not in TPXOHARMONICS:
            raise ValueError(f'Unknown constituent: {c}')

    if tpxo_dir is None:
        import os
        tpxo_dir = os.path.dirname(roms_grid_file) or '.'

    # --- 读取 ROMS 网格 ---
    g = nc.Dataset(roms_grid_file)
    lonR = np.mod(g.variables['lon_rho'][:], 360)
    latR = g.variables['lat_rho'][:]
    maskR = g.variables['mask_rho'][:]
    g.close()

    L, M = lonR.shape  # (eta_rho, xi_rho)
    N = len(constituents)

    # --- 边界多边形 ---
    bndx = [lonR[0, 0], lonR[-1, 0], lonR[-1, -1], lonR[0, -1]]
    bndy = [latR[0, 0], latR[-1, 0], latR[-1, -1], latR[0, -1]]

    # --- 读取 TPXO 数据 ---
    tpxo = _read_tpxo_data(tpxo_dir, constituents, lonR, latR, bndx, bndy)

    # --- Nodal 因子和平衡相位 ---
    # MATLAB: dnum = datenum(t0)，参考时间 = t0 + ndays/2
    dnum = _datetime_to_datenum(t0)
    t_half = dnum + ndays / 2.0
    fFac, uFac = _tpxo_nodal_factors(t_half, constituents)
    Vdeg = _vphase(dnum, constituents)

    periods = np.array([360.0 / TPXOHARMONICS[c][7] for c in constituents])

    # --- 分配数组 ---
    zamp = np.zeros((L, M, N))
    zpha = np.zeros((L, M, N))
    uamp = np.zeros((L, M, N))
    upha = np.zeros((L, M, N))
    vamp = np.zeros((L, M, N))
    vpha = np.zeros((L, M, N))
    major = np.zeros((L, M, N))
    ecc = np.zeros((L, M, N))
    inc = np.zeros((L, M, N))
    phase = np.zeros((L, M, N))

    iswet = maskR == 1

    # --- 插值 + 计算 ---
    pbar = tqdm(enumerate(constituents), desc='Tidal constituents', unit='con', total=N)
    for ki, con in pbar:
        # 高程
        ei = _interp_tpxo(tpxo['h'], ki, lonR, latR, iswet)
        zamp[:, :, ki] = np.abs(ei) * fFac[ki]
        zpha[:, :, ki] = np.mod(-np.angle(ei, deg=False) * 180/np.pi - uFac[ki] - Vdeg[ki], 360)

        # U 分量
        ei = _interp_tpxo(tpxo['U'], ki, lonR, latR, iswet)
        uamp[:, :, ki] = np.abs(ei) * fFac[ki]
        upha[:, :, ki] = np.mod(-np.angle(ei, deg=False) * 180/np.pi - uFac[ki] - Vdeg[ki], 360)

        # V 分量
        ei = _interp_tpxo(tpxo['V'], ki, lonR, latR, iswet)
        vamp[:, :, ki] = np.abs(ei) * fFac[ki]
        vpha[:, :, ki] = np.mod(-np.angle(ei, deg=False) * 180/np.pi - uFac[ki] - Vdeg[ki], 360)

        # 椭圆参数
        maj, ecc_i, inc_i, pha_i = _ap2ep(uamp[:, :, ki], upha[:, :, ki],
                                            vamp[:, :, ki], vpha[:, :, ki])
        ecc_i = np.nan_to_num(ecc_i, 0)
        major[:, :, ki] = maj
        ecc[:, :, ki] = ecc_i
        inc[:, :, ki] = inc_i
        phase[:, :, ki] = pha_i

    minor = major * np.abs(ecc)

    # --- 输出 ---
    print(f'Writing {out_file}...')
    _write_tide_nc(out_file, L, M, N, constituents, periods,
                   zamp, zpha, major, minor, inc, phase,
                   uamp, upha, vamp, vpha, t0)
    print(f'Tidal forcing written: {out_file}')


# ---------------------------------------------------------------------------
# 时间转换
# ---------------------------------------------------------------------------

def _datetime_to_datenum(dt):
    """datetime -> MATLAB datenum（天数，从 0000-01-01 起）。"""
    return dt.toordinal() + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24.0


# ---------------------------------------------------------------------------
# TPXO 数据读取
# ---------------------------------------------------------------------------

def _read_tpxo_data(tpxo_dir, constituents, lonR, latR, bndx, bndy):
    """读取 TPXO8 atlas_30 数据并裁剪到 ROMS 区域。"""
    lon_min, lon_max = lonR.min(), lonR.max()
    lat_min, lat_max = latR.min(), latR.max()
    margin = 0.5

    # 读取 TPXO 网格
    grd = nc.Dataset(f'{tpxo_dir}/grid_tpxo8atlas_30.nc')
    lon_z = grd.variables['lon_z'][:]
    lat_z = grd.variables['lat_z'][:]
    lon_u = grd.variables['lon_u'][:]
    lat_u = grd.variables['lat_u'][:]
    lon_v = grd.variables['lon_v'][:]
    lat_v = grd.variables['lat_v'][:]
    hz = grd.variables['hz'][:]
    hu = grd.variables['hu'][:]
    hv = grd.variables['hv'][:]
    grd.close()

    # meshgrid: MATLAB [X,Y] = meshgrid(X1,Y1) -> X 是 lon 方向, Y 是 lat 方向
    X_z, Y_z = np.meshgrid(lon_z, lat_z)
    X_u, Y_u = np.meshgrid(lon_u, lat_u)
    X_v, Y_v = np.meshgrid(lon_v, lat_v)

    # 裁剪索引
    I_z, J_z = np.where((Y_z >= lat_min - margin) & (Y_z <= lat_max + margin) &
                         (X_z >= lon_min - margin) & (X_z <= lon_max + margin))
    I_z, J_z = np.unique(I_z), np.unique(J_z)

    I_u, J_u = np.where((Y_u >= lat_min - margin) & (Y_u <= lat_max + margin) &
                         (X_u >= lon_min - margin) & (X_u <= lon_max + margin))
    I_u, J_u = np.unique(I_u), np.unique(J_u)

    I_v, J_v = np.where((Y_v >= lat_min - margin) & (Y_v <= lat_max + margin) &
                         (X_v >= lon_min - margin) & (X_v <= lon_max + margin))
    I_v, J_v = np.unique(I_v), np.unique(J_v)

    # 多边形掩膜
    poly = Path(np.column_stack([bndx, bndy]))

    def _process(var, J, I, X, Y, depth_full, const_files):
        """
        处理一个变量（h/U/V）。
        X/Y 来自 meshgrid，shape 为 (lat, lon)。
        TPXO 数据 (Re, Im, depth) 存储为 (lon, lat)。
        """
        # 裁剪: X(I,J) 是 (lat_sub, lon_sub)，转置为 (lon_sub, lat_sub)
        x_sub = X[np.ix_(I, J)].T
        y_sub = Y[np.ix_(I, J)].T
        depth_sub = depth_full[np.ix_(J, I)]  # depth 是 (lon, lat)

        mask_sub = poly.contains_points(np.column_stack([x_sub.ravel(), y_sub.ravel()]))
        mask_sub = mask_sub.reshape(x_sub.shape) & (depth_sub > 0)

        result = {'x': x_sub, 'y': y_sub, 'depth': depth_sub, 'mask': mask_sub}

        re_grid = np.zeros((len(constituents),) + x_sub.shape, dtype=np.complex128)
        for ki, (con, fname) in enumerate(const_files):
            ds = nc.Dataset(fname)
            vr = var.lower() if var != 'h' else 'h'
            Re = ds.variables[f'{vr}Re'][:]
            Im = ds.variables[f'{vr}Im'][:]
            ds.close()
            Re_sub = Re[np.ix_(J, I)].astype('f8')
            Im_sub = Im[np.ix_(J, I)].astype('f8')
            z = Re_sub + 1j * Im_sub

            if var == 'h':
                z = z / 1000.0      # mm -> m
            else:
                z = z / 10000.0     # cm/s -> m/s (transport)
                z = z / depth_sub  # transport / depth = velocity

            re_grid[ki] = z

        result['z'] = re_grid
        return result

    h_files = [(c, f'{tpxo_dir}/hf.{c.lower()}_tpxo8_atlas_30c.nc') for c in constituents]
    uv_files = [(c, f'{tpxo_dir}/uv.{c.lower()}_tpxo8_atlas_30c.nc') for c in constituents]

    tpxo = {}
    tpxo['h'] = _process('h', J_z, I_z, X_z, Y_z, hz, h_files)
    tpxo['U'] = _process('U', J_u, I_u, X_u, Y_u, hu, uv_files)
    tpxo['V'] = _process('V', J_v, I_v, X_v, Y_v, hv, uv_files)

    return tpxo


# ---------------------------------------------------------------------------
# 插值
# ---------------------------------------------------------------------------

def _interp_tpxo(tpxo_var, ki, lonR, latR, iswet):
    """
    两步插值（匹配 MATLAB interpTPXO）：
    1. 最近邻填充（scatteredInterpolant）
    2. 规则网格插值（griddedInterpolant）
    仅对湿点插值。
    """
    x = tpxo_var['x']
    y = tpxo_var['y']
    z = tpxo_var['z'][ki]
    m = tpxo_var['mask']

    # 步骤1: 最近邻填充
    pts_valid = np.column_stack([x[m].ravel(), y[m].ravel()])
    vals_valid = z[m].ravel()
    pts_full = np.column_stack([x.ravel(), y.ravel()])
    nn = NearestNDInterpolator(pts_valid, vals_valid)
    z_filled = nn(pts_full).reshape(x.shape)

    # 步骤2: 规则网格插值
    # x 是 (lon, lat): 每行共享同一 lon，每列共享同一 lat
    x1d = x[:, 0]
    y1d = y[0, :]
    if x1d[0] > x1d[-1]:
        x1d = x1d[::-1]
        z_filled = z_filled[::-1, :]
    if y1d[0] > y1d[-1]:
        y1d = y1d[::-1]
        z_filled = z_filled[:, ::-1]

    interp = RegularGridInterpolator(
        (x1d, y1d), z_filled,
        method='linear', bounds_error=False, fill_value=0
    )

    result = np.zeros(lonR.shape, dtype=np.complex128)
    pts_wet = np.column_stack([lonR[iswet].ravel(), latR[iswet].ravel()])
    if len(pts_wet) > 0:
        result[iswet] = interp(pts_wet).ravel()

    return result


# ---------------------------------------------------------------------------
# Nodal 因子（匹配 MATLAB TPXOnodalfactors）
# ---------------------------------------------------------------------------

def _tpxo_nodal_factors(dnum, constituents):
    """
    计算 nodal 因子 f 和 u。
    dnum: MATLAB datenum（从 0000-01-01 起的天数）。
    """
    t = (dnum + 0.5 - datetime(1900, 1, 1).toordinal()) / 36525.0

    VN = np.mod(360 * (0.719954 - 5.372617 * t + 0.000006 * t * t), 360)
    VN = np.deg2rad(VN)

    cN = np.cos(VN)
    c2N = np.cos(2 * VN)
    c3N = np.cos(3 * VN)
    sN = np.sin(VN)
    s2N = np.sin(2 * VN)
    s3N = np.sin(3 * VN)

    f_dict = {}
    u_dict = {}

    f_dict['M2'] = 1.0004 - 0.0373 * cN + 0.0002 * c2N
    u_dict['M2'] = -0.0374 * sN

    f_dict['S2'] = 1.0
    u_dict['S2'] = 0.0

    f_dict['N2'] = f_dict['M2']
    u_dict['N2'] = u_dict['M2']

    f_dict['K2'] = 1.0241 + 0.2863 * cN + 0.0083 * c2N - 0.0015 * c3N
    u_dict['K2'] = -0.3096 * sN + 0.0119 * s2N - 0.0007 * s3N

    f_dict['K1'] = 1.0060 + 0.1150 * cN - 0.0088 * c2N + 0.0006 * c3N
    u_dict['K1'] = -0.1546 * sN + 0.0119 * s2N - 0.0012 * s3N

    f_dict['O1'] = 1.0089 + 0.1871 * cN - 0.0147 * c2N + 0.0014 * c3N
    u_dict['O1'] = 0.1885 * sN - 0.0234 * s2N + 0.0033 * s3N

    f_dict['P1'] = f_dict['K1']
    u_dict['P1'] = u_dict['K1']

    f_dict['Q1'] = f_dict['O1']
    u_dict['Q1'] = u_dict['O1']

    f_out = np.array([f_dict[c] for c in constituents])
    u_out = np.array([np.mod(u_dict[c] * 180 / np.pi, 360) for c in constituents])
    return f_out, u_out


# ---------------------------------------------------------------------------
# 平衡相位（匹配 MATLAB Vphase）
# ---------------------------------------------------------------------------

def _vphase(dnum, constituents):
    """
    计算平衡相位 Vdeg。
    dnum: MATLAB datenum。
    """
    t = (dnum + 0.5 - datetime(1900, 1, 1).toordinal()) / 36525.0
    t_hour = (dnum % 1) * 24

    Vs = np.mod(360 * (0.751206 + 1336.855231 * t - 0.000003 * t * t), 360)
    Vh = np.mod(360 * (0.776935 + 100.002136 * t + 0.000001 * t * t), 360)
    Vp = np.mod(360 * (0.928693 + 11.302872 * t - 0.000029 * t * t), 360)
    VN = np.mod(360 * (0.719954 - 5.372617 * t + 0.000006 * t * t), 360)
    Vp1 = np.mod(360 * (0.781169 + 0.004775 * t + 0.000001 * t * t), 360)

    result = []
    for con in constituents:
        DN = TPXOHARMONICS[con]
        V = (t_hour * DN[7] + Vs * DN[1] + Vh * DN[2] +
             Vp * DN[3] + VN * DN[4] + Vp1 * DN[5] + DN[6])
        V = np.mod(V, 360)
        result.append(V)
    return np.array(result)


# ---------------------------------------------------------------------------
# ap2ep（匹配 MATLAB ap2ep）
# ---------------------------------------------------------------------------

def _ap2ep(Au, PHIu, Av, PHIv):
    """
    将振幅/相位转换为椭圆参数。
    匹配 MATLAB ap2ep（Zhigang Xu 原版）。
    """
    PHIu_r = np.deg2rad(PHIu)
    PHIv_r = np.deg2rad(PHIv)

    u = Au * np.exp(-1j * PHIu_r)
    v = Av * np.exp(-1j * PHIv_r)

    wp = (u + 1j * v) / 2
    wm = np.conj(u - 1j * v) / 2

    Wp = np.abs(wp)
    Wm = np.abs(wm)
    THETAp = np.angle(wp)
    THETAm = np.angle(wm)

    SEMA = Wp + Wm
    SEMI = Wp - Wm
    ECC = SEMI / (SEMA + 1e-30)
    PHA = np.rad2deg(THETAm - THETAp) / 2
    INC = np.rad2deg(THETAm + THETAp) / 2

    PHA = np.mod(PHA, 360)
    INC = np.mod(INC, 360)

    return SEMA, ECC, INC, PHA


# ---------------------------------------------------------------------------
# NetCDF 输出（NETCDF3_CLASSIC 格式，兼容 ROMS 老版本）
# ---------------------------------------------------------------------------

def _write_tide_nc(fn, L, M, N, constituents, periods,
                   zamp, zpha, cmax, cmin, cangle, cphase,
                   uamp, upha, vamp, vpha, ini_date):
    """
    写入 ROMS 潮汐强迫文件。
    维度顺序: (xi_rho, eta_rho, tide_period) — 与 MATLAB 一致。
    使用 NETCDF3_CLASSIC 格式确保 ROMS 老版本可读取。
    """
    ds = nc.Dataset(fn, 'w', format='NETCDF3_CLASSIC')

    # 全局属性
    ROMStitle = f'ROMS TPXO data for {ini_date.strftime("%b %d %Y")}'
    ds.title = ROMStitle
    ds.Creation_date = ini_date.strftime('%Y%m%d')
    ds.grd_file = fn
    ds.type = 'ROMS forcing file from TPXO'
    dnum = _datetime_to_datenum(ini_date)
    ds.ini_date_datenumber = dnum
    ds.ini_date_mjd = dnum - datetime(1968, 5, 23).toordinal()
    components_str = ' '.join([f'{c} ' for c in constituents])
    ds.components = components_str

    # 维度
    ds.createDimension('tide_period', N)
    ds.createDimension('eta_rho', L)
    ds.createDimension('xi_rho', M)

    # zero_phase_date（ROMS 惯例）
    zpd = float(ini_date.strftime('%Y%m%d.%f'))
    zv = ds.createVariable('zero_phase_date', 'f8', ())
    zv.long_name = 'tidal reference date for zero phase'
    zv.units = 'days as %Y%m%d.%f'
    zv.C_format = '%13.4f'
    zv.FORTRAN_format = '(f13.4)'
    zv[:] = zpd

    # tide_period
    v = ds.createVariable('tide_period', 'f8', ('tide_period',))
    v.long_name = 'Tide angular period'
    v.units = 'hours'
    v[:] = periods

    # 所有潮汐场: MATLAB 写入 (xi_rho, eta_rho, tide_period)
    # 我们的数据是 (L, M, N) = (eta_rho, xi_rho, tide_period)
    # 需要转置: data.transpose(1, 0, 2) -> (xi_rho, eta_rho, tide_period)
    def _reorder(arr):
        return np.ascontiguousarray(arr.transpose(1, 0, 2))

    def _write_vec(name, data, long_name, units):
        v = ds.createVariable(name, 'f8', ('xi_rho', 'eta_rho', 'tide_period'))
        v.long_name = long_name
        v.units = units
        v[:] = _reorder(data)

    _write_vec('tide_Eamp', zamp,
               'Tide elevation amplitude', 'meters')
    _write_vec('tide_Ephase', zpha,
               'Tide elevation phase angle', 'degrees')

    _write_vec('tide_Cmax', cmax,
               'Tidal current ellipse semi-major axis', 'meter second-1')
    _write_vec('tide_Cmin', cmin,
               'Tidal current ellipse semi-minor axis', 'meter second-1')
    _write_vec('tide_Cangle', cangle,
               'Tidal current ellipse inclination angle',
               'degrees between semi-major axis and east')
    _write_vec('tide_Cphase', cphase,
               'Tidal current phase angle', 'degrees')

    _write_vec('tide_Uamp', uamp,
               'Tidal current U-component amplitude', 'meter second-1')
    _write_vec('tide_Uphase', upha,
               'Tidal current U-component phase', 'degrees')
    _write_vec('tide_Vamp', vamp,
               'Tidal current V-component amplitude', 'meter second-1')
    _write_vec('tide_Vphase', vpha,
               'Tidal current V-component phase', 'degrees')

    ds.close()
