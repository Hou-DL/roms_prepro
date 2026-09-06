"""
========================================================================
  CMEMS -> ROMS Initial Condition (IC) 制作模板
  CMEMS -> ROMS Initial Condition (IC) Template
========================================================================

功能: 从 CMEMS (Copernicus Marine Service) 数据制作 ROMS 初始场
Input:  5个CMEMS变量文件 (zos, thetao, so, uo, vo)
Output: ROMS IC NetCDF 文件

支持两种数据组织模式:
  Mode A: 分别指定各变量文件路径 (zeta_file, temp_file, ...)
  Mode B: 指定 data_dir 目录，自动搜索文件

用法:
    python cmems2roms_ini.py

依赖:
    numpy, scipy, netCDF4, roms_prepro

Verified: 2025-07-06 (NSCS domain, CMEMS 2025-05)
========================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from roms_prepro.ic import cmems_to_roms_ini

# ===================================================================
# 1. 路径设置 / Path Settings
# ===================================================================

# ROMS 网格文件
# ROMS grid file (must contain: h, lon_rho, lat_rho, lon_u, lat_u, lon_v, lat_v,
#                                    mask_rho, mask_u, mask_v, angle)
GRDNAME = "/data/hdl/roms_prepro/examples/NSCS_grd_operational_adjust.nc"

# 输出 IC 文件路径
# Output IC file path
ININAME = "/data/hdl/roms_prepro/examples/NSCS_ini20250522.nc"

# -------------------------------------------------------------------
# 模式 A: 分别指定各变量文件 (Mode A: individual file paths)
# 如果以下5个路径全部指定，则直接使用，忽略 DATA_DIR
# -------------------------------------------------------------------
ZFILE = "/mnt/q1/public_ocean_data/Mercator/2025/cmems_zos_202505.nc"    # 海面高度 SSH
TFILE = "/mnt/q1/public_ocean_data/Mercator/2025/cmems_thetao_202505.nc"  # 温度 Temperature
SFILE = "/mnt/q1/public_ocean_data/Mercator/2025/cmems_so_202505.nc"      # 盐度 Salinity
UFILE = "/mnt/q1/public_ocean_data/Mercator/2025/cmems_uo_202505.nc"      # U流速 U velocity
VFILE = "/mnt/q1/public_ocean_data/Mercator/2025/cmems_vo_202505.nc"      # V流速 V velocity

# -------------------------------------------------------------------
# 模式 B: 目录自动搜索 (Mode B: auto-detect from directory)
# 将 ZFILE/TFILE/SFILE/UFILE/VFILE 全部设为 None，则使用 DATA_DIR 模式
# DATA_DIR 目录下自动搜索: cmems_zos_*.nc, cmems_thetao_*.nc, ...
# -------------------------------------------------------------------
DATA_DIR = None  # e.g. "/mnt/q1/public_ocean_data/Mercator/2025"

# ===================================================================
# 2. CMEMS 变量名 / CMEMS Variable Names
# ===================================================================
# 如果CMEMS文件中的变量名不是标准名，在此修改
# Modify if your CMEMS files use non-standard variable names

ZVAR = 'zos'       # SSH: zos, sla, adt
TVAR = 'thetao'    # Temperature: thetao, t, temperature
SVAR = 'so'        # Salinity: so, s, salinity
UVAR = 'uo'        # U-velocity: uo, u, water_u
VVAR = 'vo'        # V-velocity: vo, v, water_v

# ===================================================================
# 3. ROMS 垂直网格参数 / Vertical Grid Parameters
# ===================================================================
# 设为 None 则自动从 GRDNAME 网格文件中读取
# Set to None to auto-read from grid file

VTRANSFORM  = 2      # 垂直变换公式: 1=旧版, 2=新版 (ROMS default: 2)
VSTRETCHING = 3      # 拉伸函数: 1-4 (常用: 3 or 4; ROMS default: 4)
THETA_S     = 2.5    # 表层拉伸参数 (0-10, 越大表层越密)
THETA_B     = 1.0    # 底层拉伸参数 (0-4, 越大底层越密)
TCLINE      = 25.0   # 临界深度 (m), 表层/底层的分界深度
N           = 30     # 垂直 sigma 层数

# ===================================================================
# 4. 时间设置 / Time Settings
# ===================================================================

# 初始场日期 (IC date)
# 程序自动从CMEMS文件中匹配该日期对应的时间步
# Set to None to use first time step in file
IC_DATE = '2025-05-22'

# ROMS 时间基准 (ROMS time reference)
# ocean_time 的起算日期, 格式: 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
# Set to None to auto-detect from CMEMS file time units
TIME_REF = None  # e.g. '1990-01-01'

# 显式时间索引 (Explicit time index)
# 设置此值将覆盖 IC_DATE 的自动匹配
# Set to None to use IC_DATE auto-detection
TIME_INDEX = None  # e.g. 0, 1, 2, ...

# ===================================================================
# 5. QC / 质量控制选项 / Quality Control Options
# ===================================================================
# 以下硬编码在 cmems_to_roms_ini() 内部，此处仅作说明:
# - 温度范围: -100 ~ 50 °C (超出设为NaN后填充)
# - 盐度范围: 0 ~ 50 PSU
# - 速度范围: -100 ~ 100 m/s
# - 深度扩展: 当ROMS最大水深 > CMEMS最大水深时，自动延拓
# - NaN填充: 先水平最近邻填充，再垂直填充

# ===================================================================
# 6. 运行 / Run
# ===================================================================

if __name__ == '__main__':
    result = cmems_to_roms_ini(
        # ---- 必需参数 ----
        roms_grid_file=GRDNAME,
        ini_file=ININAME,

        # ---- 数据源 (Mode A: 分别指定) ----
        zeta_file=ZFILE,
        temp_file=TFILE,
        salt_file=SFILE,
        u_file=UFILE,
        v_file=VFILE,

        # ---- 数据源 (Mode B: 目录自动搜索) ----
        # data_dir=DATA_DIR,

        # ---- CMEMS变量名 ----
        zeta_var=ZVAR,
        temp_var=TVAR,
        salt_var=SVAR,
        u_var=UVAR,
        v_var=VVAR,

        # ---- 垂直网格参数 (None=自动从网格文件读取) ----
        Vtransform=VTRANSFORM,
        Vstretching=VSTRETCHING,
        theta_s=THETA_S,
        theta_b=THETA_B,
        Tcline=TCLINE,
        N=N,

        # ---- 时间设置 ----
        init_date=IC_DATE,
        time_ref=TIME_REF,
        time_index=TIME_INDEX,
    )

    if result:
        print('\n' + '=' * 60)
        print('Success! IC file created.')
        print(f'Output: {ININAME}')
        print('=' * 60)
    else:
        print('\nFailed!')
