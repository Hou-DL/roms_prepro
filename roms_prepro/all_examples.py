"""
========================================================================
  ROMS Preprocessing Toolkit — 总示例 / Master Example
========================================================================

功能: 一份文件覆盖全部模块的用法索引
       One file indexing the usage of every module

模块 / Modules
--------------
  grid       网格生成 (GEBCO 水深 -> ROMS grid)、垂向坐标、度量
  ic         初始场: CMEMS 分变量文件 / 单文件 (Mercator/HYCOM) / ROMS→ROMS
  bc         边界场: CMEMS→ROMS / ROMS→ROMS
  forcing    ERA5 大气强迫 (GRIB/NC, 风场/温压/辐射/降水)
  tide       TPXO8 潮汐强迫 (8 主要分潮)
  river      河流强迫 (月气候态 / 指定时段日序列)
  remapping  sigma→z 坐标转换 (整文件一键 / 单数组 / 站点)

用法 / Usage
-----------
    python roms_prepro/all_examples.py                # 列出全部示例
    python roms_prepro/all_examples.py grid           # 运行指定示例
    python roms_prepro/all_examples.py ic_cmems bry_cmems remap_file ...

注意: 各示例里的路径均为占位符，运行前改成自己的数据路径。
Note: paths are placeholders — edit them before running.

依赖 / Dependencies: numpy, scipy, netCDF4, xarray (ERA5), tqdm
========================================================================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ======================================================================
# 1. grid — ROMS 网格生成
# ======================================================================

def example_grid():
    """
    从四角坐标 + GEBCO 水深生成 ROMS 网格文件。
    全流程: lon/lat 生成 -> 度量 (pm/pn/angle/f/dndx/dmde) -> 水深插值
            -> 掩膜 -> 光滑 (rx0 判据) -> 写标准 ROMS 网格 NetCDF
    """
    from roms_prepro.grid import create_roms_grid

    # ---------- edit these parameters ----------
    grid_file = 'my_grid.nc'

    # 域四角 [SW, NW, NE, SE]
    lon_corners = [119.0, 119.0, 125.7, 125.7]
    lat_corners = [33.5, 37.1, 37.1, 33.5]
    nx, ny = 300, 200            # RHO 点数
    min_depth = 5.0              # 最小水深 (m)
    rx0max = 0.2                 # 地形粗糙度判据

    # GEBCO 水深 (规则网格; source_depth 形状 (nlat, nlon), 陆地为正)
    # 读取示例:
    import netCDF4 as nc
    with nc.Dataset('gebco_subset.nc') as g:
        source_lon = g.variables['lon'][:]
        source_lat = g.variables['lat'][:]
        source_depth = g.variables['elevation'][:]
    # -------------------------------------------

    create_roms_grid(
        lon_corners=lon_corners, lat_corners=lat_corners,
        nx=nx, ny=ny, grid_file=grid_file,
        source_lon=source_lon, source_lat=source_lat, source_depth=source_depth,
        min_depth=min_depth, rx0max=rx0max,
        vgrid_params={'Vtransform': 2, 'Vstretching': 4},
    )


def example_grid_metrics():
    """已有 lon_rho/lat_rho (eta,xi) 二维数组时, 单独计算全部网格度量。"""
    from roms_prepro.grid import make_grid_corner, compute_metrics

    lon_rho, lat_rho = make_grid_corner(
        [119.0, 119.0, 125.7, 125.7],   # [SW, NW, NE, SE]
        [33.5, 37.1, 37.1, 33.5],
        nx=300, ny=200)

    m = compute_metrics(lon_rho, lat_rho)
    # 返回 dict: pm/pn/angle/f/dndx/dmde/x_rho/y_rho/lon_u/lat_u/...
    # ROMS 约定: 1/pm = XI 方向间距, 1/pn = ETA 方向间距
    print('pm range:', m['pm'].min(), m['pm'].max())


def example_vgrid():
    """
    垂向坐标底层函数 (与 pyroms/ROMS set_scoord 数值一致, VS1-4)。
    s: -1(底)~0(面); Cs: 拉伸曲线; z: 负值 (海面 0, 向下为负)。
    """
    from roms_prepro.grid import stretching, set_depth, s_rho, s_w

    N, theta_s, theta_b, Tcline = 30, 7.0, 0.1, 20.0
    Vtransform, Vstretching = 2, 4

    s_r, Cs_r = stretching(Vstretching, theta_s, theta_b, N, kgrid=0)  # RHO 点
    s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, N, kgrid=1)  # W 点 (N+1)
    print('s_rho 端点:', s_r[0], s_r[-1])      # 底/面
    print('s_w   端点:', s_w[0], s_w[-1])      # -1.0 / 0.0

    h = np.full((5, 5), 2000.0)                # 水深 (eta, xi)
    zeta = np.zeros_like(h)                    # 海面高度 (可为时变 2D 场)

    z_r = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, zeta=zeta, igrid=1)     # (N, eta, xi)   RHO 层深
    z_u = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, zeta=zeta, igrid=3)     # (N, eta, xi-1) U 点
    z_v = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, zeta=zeta, igrid=4)     # (N, eta-1, xi) V 点
    z_w = set_depth(Vtransform, Vstretching, theta_s, theta_b, Tcline, N,
                    h, zeta=zeta, igrid=5)     # (N+1, eta, xi) W 界面
    print('z_r 表层/底层:', z_r[-1, 0, 0], z_r[0, 0, 0])


# ======================================================================
# 2. ic — 初始场
# ======================================================================

def example_ic_cmems():
    """
    CMEMS 分变量文件 -> ROMS IC (推荐路径, 端到端验证)。
    三种文件组织: A) 目录自动搜索  B) 指定各变量文件  C) 全变量合并单文件
    """
    from roms_prepro.ic import cmems_to_roms_ini

    grid_file = 'my_grid.nc'
    ini_file = 'my_ini_cmems.nc'

    # ---------- edit these parameters ----------
    # 方式 A: 目录 (自动匹配 cmems_<var>_*.nc 或合并文件)
    data_dir = '/data/hdl/oceanfiles/CMEMS/202505/'

    # 方式 B: 逐文件指定 (取消注释并注释掉 data_dir)
    # zeta_file, temp_file = 'cmems_zos_202505.nc', 'cmems_thetao_202505.nc'
    # salt_file, u_file, v_file = 'cmems_so_202505.nc', 'cmems_uo_202505.nc', 'cmems_vo_202505.nc'

    Vtransform, Vstretching = 2, 3      # 必须与 ROMS 运行参数一致
    theta_s, theta_b, Tcline = 2.5, 1.0, 25.0
    N = 30
    init_date = '2025-05-01'            # 制作哪一天 (自动匹配最近时间步)
    time_ref = None                     # None=自动从 CMEMS 时间单位读取
    # -------------------------------------------

    cmems_to_roms_ini(
        roms_grid_file=grid_file, ini_file=ini_file,
        data_dir=data_dir,
        # zeta_file=zeta_file, temp_file=temp_file, salt_file=salt_file,
        # u_file=u_file, v_file=v_file,
        Vtransform=Vtransform, Vstretching=Vstretching,
        theta_s=theta_s, theta_b=theta_b, Tcline=Tcline, N=N,
        init_date=init_date, time_ref=time_ref,
    )


def example_ic_mercator():
    """单文件源 (所有变量在同一文件, Mercator/HYCOM/CMEMS 合并下载) -> ROMS IC。"""
    from roms_prepro.ic import mercator_to_roms_ini

    mercator_to_roms_ini(
        roms_grid_file='my_grid.nc',
        source_file='cmems_glo_phy_20220823.nc',
        ini_file='my_ini.nc',
        Vtransform=2, Vstretching=4,
        theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
        init_date='2025-01-01',          # None = 取第一个时间步
    )


def example_ic_roms2roms():
    """父网格 ROMS 输出 -> 子网格 IC (嵌套; 经中间 z 层重网格, 自动旋转流速)。"""
    from roms_prepro.ic import roms_to_roms_ini

    roms_to_roms_ini(
        src_grid_file='parent_grid.nc',
        src_hist_dir='./parent_output/',   # 目录自动搜索; 或 src_hist_file='ocean_his_0001.nc'
        dst_grid_file='my_grid.nc',
        ini_file='my_ini_from_parent.nc',
        Vtransform=2, Vstretching=4,       # 子网格垂向参数 (父网格参数自动读取)
        theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
        init_date='2020-01-15',            # 制作哪一天
    )


# ======================================================================
# 3. bc — 边界场
# ======================================================================

def example_bry_cmems():
    """
    CMEMS -> ROMS 随时间变化的边界场 (端到端验证的主路径)。
    配置方式: 修改模块全局变量后调用 main()。
    """
    import roms_prepro.bc.d_obc_cmems as cfg
    from roms_prepro.bc.d_obc_cmems import main as cmems_bry_main

    # ---------- edit these parameters ----------
    cfg.DATA_DIR = '/data/hdl/oceanfiles/CMEMS/2025'   # 分变量文件目录
    # cfg.CMEMS_FILE = 'cmems_all_2025.nc'             # 或指定合并单文件
    cfg.GRD_NAME = 'my_grid.nc'
    cfg.BRY_NAME = 'my_bry_cmems.nc'

    cfg.BOUNDARY = [0, 1, 1, 0]           # 开边界 [西, 东, 南, 北]

    cfg.N_LEVELS = 30                     # 垂向参数与 ROMS 运行一致
    cfg.VTRANSFORM = 2
    cfg.VSTRETCHING = 3
    cfg.THETA_S = 2.5
    cfg.THETA_B = 1.0
    cfg.TCLINE = 25.0

    cfg.TIME_START = '2025-05-01 00:00:00'
    cfg.TIME_END = '2025-09-30 23:00:00'
    cfg.ROMS_TIME_REF = '1990-01-01 00:00:00'
    # -------------------------------------------

    cmems_bry_main()


def example_bry_roms2roms():
    """父网格 ROMS 输出 -> 子网格时间依赖边界场。"""
    from roms_prepro.bc import roms_to_roms_bry

    roms_to_roms_bry(
        src_grid_file='parent_grid.nc',
        src_hist_dir='./parent_output/',   # 或 src_hist_files=['ocean_avg_001.nc', ...]
        dst_grid_file='my_grid.nc',
        bry_file='my_bry_from_parent.nc',
        Vtransform=2, Vstretching=4,
        theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
        boundaries=(True, True, True, True),   # (西, 东, 南, 北)
        start_date='2020-01-01', end_date='2020-01-31',
        # time_ref=None,                      # 自动从源文件 ocean_time 单位读取
    )


# ======================================================================
# 4. forcing — ERA5 大气强迫
# ======================================================================

def example_forcing_era5():
    """
    ERA5 (GRIB 或 NC) -> ROMS 大气强迫。
    支持变量: u10 v10 t2m d2m msl tp tcc r, 辐射 msdrswrf/msdwswrf/ssrd,
              msdwlwrf/msnlwrf; 累积量 (J/m2, m) 自动按时间步长换算为通量。
    输出两个文件: 风场 (romsforc_era5_wind.nc) + 其它强迫 (romsforc_era5_forcing.nc)
    """
    from roms_prepro.forcing import ERA5toROMS

    converter = ERA5toROMS(
        in_dir='/data/hdl/oceanfiles/era5/in/',      # ERA5 文件目录 (自动搜索)
        out_dir='./forcing_output',
        time_start='2025-01-01 00:00:00',            # None = 全部时间
        time_end='2025-02-20 23:00:00',
        base_date='1990-01-01 00:00:00',             # ROMS 时间基准
        grid_file=None,                              # None=保持 ERA5 网格;
                                                      # 传网格文件+rotate_wind=True 可旋转风场
        variables=['u10', 'v10', 't2m', 'd2m', 'msl', 'tp',
                   'msdrswrf', 'msdwlwrf'],
        rotate_wind=False,
        # wind_file='era5_wind_2025.nc', forcing_file='era5_forc_2025day.nc',
    )
    converter.process()


# ======================================================================
# 5. tide — TPXO8 潮汐强迫
# ======================================================================

def example_tide():
    """
    TPXO8 atlas_30 -> ROMS 潮汐强迫文件。
    默认 8 分潮 M2 S2 N2 K2 K1 O1 P1 Q1; 文件名自动兼容 _v1 后缀。
    """
    from roms_prepro.tide import tpxo_to_roms_tide

    tpxo_to_roms_tide(
        'my_grid.nc',
        'my_tide.nc',
        t0='2000-01-01',            # 相位参考时间
        ndays=365,                  # 模拟长度 (nodal 因子取 t0+ndays/2)
        tpxo_dir='/data/hdl/oceanfiles/tpxo8_atlas_30/',
        constituents=None,          # None = 8 标准分潮; 或 ['M2', 'S2', ...]
    )


# ======================================================================
# 6. river — 河流强迫
# ======================================================================

def example_river():
    """
    月气候态流量 -> ROMS 河流强迫 (含 river_flag/Udirection/Vdirection 等必需变量)。
    两种时间轴: 默认 12 个月记录 + cycle_length=365.25 (ROMS 内部循环);
                或 time_start/time_end 生成日序列 (气候态周期插值)。
    """
    from roms_prepro.river import create_river_file, yangtze_river, find_river_mouth
    import netCDF4 as nc

    grid_file = 'my_grid.nc'
    with nc.Dataset(grid_file) as g:
        lon_rho = g.variables['lon_rho'][:]
        lat_rho = g.variables['lat_rho'][:]
        mask_rho = g.variables['mask_rho'][:]

    rivers = [
        # 内置月气候态 (长江/黄河), 河口位置自动定位
        yangtze_river(lon_rho, lat_rho, mask_rho, discharge_m3s=31000, salt_psu=0.5),
        # huanghe_river(lon_rho, lat_rho, mask_rho),

        # 自定义河流: I/J 为网格索引 (xi/eta), direction 0=沿+XI 1=沿+ETA
        # {'name': 'Custom River', 'I': 100, 'J': 50,
        #  'discharge': 800.0, 'salt': 1.0, 'direction': 0},
    ]

    # 也可以只给河口经纬度, 自动找最近湿点:
    # i, j = find_river_mouth(lon_rho, lat_rho, mask_rho, 121.5, 31.5)

    create_river_file(
        roms_grid_file=grid_file,
        out_file='my_river.nc',
        rivers=rivers,
        # time_start='2025-01-01', time_end='2025-12-31',  # 日序列模式
    )


# ======================================================================
# 7. remapping — sigma → z 坐标转换
# ======================================================================

def example_remap_file():
    """
    整文件一键转换 (默认变量 temp/salt/u/v/zeta; zeta 表层场直接拷贝)。
    垂直参数自动从文件读取, 可逐键覆盖; 深度自动过滤超过最大水深的层。
    """
    from roms_prepro.remapping import process_file, DEFAULT_STD_DEPTHS

    # 默认: 全部变量换到 rho 网格, 坐标名 lon/lat, 输出精简易读
    process_file('ocean_avg_0003.nc', 'ocean_avg_0003_z.nc')

    # 完整参数示例:
    process_file(
        'ocean_avg_0003.nc',
        'ocean_avg_0003_z.nc',
        variables=['temp', 'salt', 'u', 'v', 'zeta'],   # None = 默认五变量
        std_depths=[0, 10, 25, 50, 100, 200, 500, 1000, 2000, 3000],  # None = DEFAULT_STD_DEPTHS
        vgrid_params=None,      # None=自动读取; 可覆盖 {'N': 30, 'theta_s': 5.0, ...}
        to_rho=True,            # True=全 rho 网格; False=保留交错网格
    )

    # CLI 批处理 / CLI batch:
    #   python -m roms_prepro.remapping.roms2z_levels -i 'ocean_avg_*.nc' -d ./z_out/ \
    #       --depths 0 10 50 100 500 --vars temp salt
    #   python -m roms_prepro.remapping.roms2z_levels -i in.nc -o out.nc --native-grid


def example_remap_array():
    """单个 sigma 坐标数组 (N, eta, xi) -> z 层 (内存中, 不落盘)。"""
    from roms_prepro.remapping import sigma_to_z_levels

    # 网格文件: 演示用仓库自带示例网格; 实际使用换成自己的网格
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grid_file = os.path.join(repo_root, 'examples', 'NSCS_grd_operational_adjust.nc')
    if not os.path.exists(grid_file):
        grid_file = 'my_grid.nc'

    # var_sigma: (N, eta, xi), 形状须与网格一致; 例如 ds.variables['temp'][0]
    import netCDF4 as nc
    with nc.Dataset(grid_file) as g:
        ny, nx = g.variables['h'].shape
    var_sigma = np.random.rand(30, ny, nx)

    z_levels = np.arange(-5, -500, -5)          # 负值向下
    var_z = sigma_to_z_levels(
        var_sigma, grid_file, z_levels,
        Cpos='rho',                             # 'rho' | 'u' | 'v' | 'w'
        Vtransform=2, Vstretching=4,
        theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
    )
    print('var_z shape:', var_z.shape)          # (nz, eta, xi)


def example_remap_station():
    """站点剖面 (N_sigma, n_sta) -> z 层 (如浮标/断面数据)。"""
    from roms_prepro.remapping import station_to_z_levels

    # sigma 深度: 每站各层的实际深度 (负值向下), 可由 set_depth 计算
    from roms_prepro.grid import set_depth
    n_sta = 3
    h_sta = np.full(n_sta, 500.0)                       # 各站水深
    z_r = set_depth(2, 4, 7.0, 0.1, 20.0, 30, h_sta, igrid=1)   # (N, n_sta)
    sigma_depth = z_r

    var_sigma = np.random.rand(30, n_sta)               # (N_sigma, n_sta)
    z_levels = np.arange(-5, -500, -5)
    var_z = station_to_z_levels(var_sigma, sigma_depth, z_levels)
    print('var_z shape:', var_z.shape)                  # (nz, n_sta)


# ======================================================================
# main
# ======================================================================

ALL_EXAMPLES = {
    'grid':          ('网格生成 (GEBCO -> ROMS grid)',            example_grid),
    'grid_metrics':  ('已有 lon/lat 计算度量',                     example_grid_metrics),
    'vgrid':         ('垂向坐标 stretching/set_depth',             example_vgrid),
    'ic_cmems':      ('初始场: CMEMS 分变量文件',                  example_ic_cmems),
    'ic_mercator':   ('初始场: 单文件 (Mercator/HYCOM)',           example_ic_mercator),
    'ic_roms2roms':  ('初始场: 父网格 ROMS -> 子网格 (嵌套)',      example_ic_roms2roms),
    'bry_cmems':     ('边界场: CMEMS -> ROMS',                     example_bry_cmems),
    'bry_roms2roms': ('边界场: 父网格 ROMS -> 子网格',             example_bry_roms2roms),
    'forcing_era5':  ('大气强迫: ERA5 -> ROMS',                    example_forcing_era5),
    'tide':          ('潮汐强迫: TPXO8 -> ROMS',                   example_tide),
    'river':         ('河流强迫 (月气候态/日序列)',                example_river),
    'remap_file':    ('整文件一键 sigma -> z',                     example_remap_file),
    'remap_array':   ('单个数组 sigma -> z',                       example_remap_array),
    'remap_station': ('站点剖面 sigma -> z',                       example_remap_station),
}


def main():
    if len(sys.argv) <= 1:
        print(__doc__)
        print('可用示例 / Available examples:')
        for name, (desc, _) in ALL_EXAMPLES.items():
            print(f'  {name:15s} {desc}')
        print('\n运行 / Run:  python roms_prepro/all_examples.py <name> [name ...]')
        return

    for name in sys.argv[1:]:
        if name not in ALL_EXAMPLES:
            print(f'未知示例 / unknown example: {name}')
            continue
        desc, func = ALL_EXAMPLES[name]
        print(f'\n{"=" * 60}\n  EXAMPLE: {name} — {desc}\n{"=" * 60}')
        func()


if __name__ == '__main__':
    main()
