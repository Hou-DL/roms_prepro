"""
========================================================================
  CMEMS -> ROMS Boundary Condition (BC) 制作模板
  CMEMS -> ROMS Boundary Condition (BC) Template
========================================================================

功能: 从 CMEMS 数据制作 ROMS 随时间变化的边界场
Input:  CMEMS 月报/日报文件 (zos, thetao, so, uo, vo)
Output: ROMS BRY NetCDF 文件 (含 bry_time 维度)

支持两种数据组织模式:
  Mode A: CMEMS_FILE 指定单文件 (全变量合并文件)
  Mode B: DATA_DIR 目录自动搜索 (分离变量文件或合并文件)

用法:
    python example_bc_cmems.py

依赖:
    numpy, scipy, xarray, netCDF4, roms_prepro

Verified: 2025-07-06 (NSCS domain, CMEMS 2024)
========================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roms_prepro.bc.d_obc_cmems as cfg
from roms_prepro.bc.d_obc_cmems import main as cmems_bry_main

# ===================================================================
# 1. 路径设置 / Path Settings
# ===================================================================

# ROMS 网格文件
# ROMS grid file (must contain: h, lon_rho, lat_rho, lon_u, lat_u,
#                                    lon_v, lat_v, mask_rho, mask_u, mask_v, angle)
GRID_FILE = "/data/hdl/roms_prepro/examples/NSCS_grd_operational_adjust.nc"

# 输出 BRY 文件路径
# Output boundary file path
BRY_FILE = "/data/hdl/roms_prepro/examples/NSCS_bry2024.nc"

# -------------------------------------------------------------------
# 模式 A: 单文件模式 (Mode A: single multi-variable file)
# 设置 CMEMS_FILE 为包含全部变量 (zos, thetao, so, uo, vo) 的文件路径
# -------------------------------------------------------------------
CMEMS_FILE = None  # e.g. "/path/to/cmems_merged_202405.nc"

# -------------------------------------------------------------------
# 模式 B: 目录自动搜索 (Mode B: auto-detect from directory)
# CMEMS_FILE = NULL 时使用此模式
# 优先搜索 CMEMS_MERGED_PATTERN 匹配的合并文件
# 若无合并文件，则搜索分离变量文件 (cmems_zos_*.nc, ...)
# -------------------------------------------------------------------
CMEMS_DIR = "/mnt/q1/public_ocean_data/Mercator/2024"

# 合并文件通配符 (Merged file glob pattern)
CMEMS_MERGED_PATTERN = 'cmems_glo_phy_*.nc'

# ===================================================================
# 2. 边界开关 / Boundary Switches
# ===================================================================
# BOUNDARY = [west, east, south, north]
# 每个元素: 1=开启该边界, 0=关闭
#
#  示例:  NSCS (南海) 一般开 east + south
#        BOUNDARY = [0, 1, 1, 0]
#
#  示意图:
#            North (3)
#              ↑
#   West (0) ←  → East (1)
#              ↓
#            South (2)
#
#  注意: 边界位置取自网格文件:
#    West  = xi=0 (第一列)
#    East  = xi=-1 (最后一列)
#    South = eta=0 (第一行)
#    North = eta=-1 (最后一行)
BOUNDARY = [0, 1, 1, 0]

# ===================================================================
# 3. 垂直网格参数 / Vertical Grid Parameters
# ===================================================================
# 设为 None 则自动从 GRID_FILE 网格文件中读取
# Set to None to auto-read from grid file

N_LEVELS    = 50     # 垂直 sigma 层数
VTRANSFORM  = 2      # 垂直变换公式: 1=旧版, 2=新版 (ROMS default: 2)
VSTRETCHING = 4      # 拉伸函数: 1-4 (常用: 3 or 4; ROMS default: 4)
THETA_S     = 5.0    # 表层拉伸参数 (0-10, 越大表层越密)
THETA_B     = 2.0    # 底层拉伸参数 (0-4, 越大底层越密)
TCLINE      = 200.0  # 临界深度 (m), 表层/底层的分界深度

# ===================================================================
# 4. 时间设置 / Time Settings
# ===================================================================
# 时间格式: 'YYYY-MM-DD HH:MM:SS'
# Time format: 'YYYY-MM-DD HH:MM:S'

TIME_START = '2024-09-01 00:00:00'  # 起始时间 (包含)
TIME_END   = '2024-11-08 23:00:00'  # 结束时间 (包含)

# ROMS 时间基准 (ROMS time reference)
# bry_time 的起算日期, 格式: 'YYYY-MM-DD HH:MM:SS'
ROMS_TIME_REF = '1970-01-01 00:00:00'

# ===================================================================
# 5. 填充值 / Fill Value
# ===================================================================
# NetCDF 陆地点填充值 (ROMS standard: 1e37)
FILL_VAL = 1.0e+37

# ===================================================================
# 6. 处理说明 / Processing Notes
# ===================================================================
# 内部执行流程:
#   [1/5] 读取 ROMS 网格
#   [2/5] 发现 CMEMS 文件 (先合并文件，后分离文件)
#   [2.5] 按时间范围预过滤文件
#   [3/5] 创建输出 NetCDF (根据 BOUNDARY 开关创建变量)
#   [4/5] 逐文件逐时间步处理:
#         - 读取 + NaN填充
#         - 水平插值到边界 (线性 + 最近邻)
#         - 垂直插值到 ROMS sigma 层
#         - 速度旋转 (地理坐标 -> ROMS C-grid)
#         - 计算正压速度 (ubar, vbar)
#         - 写入输出文件
#   [5/5] 完成
#
# 输出变量 (以 east 边界为例):
#   zeta_east(bry_time, eta_rho)        - 海面高度
#   temp_east(bry_time, eta_rho, s_rho) - 温度
#   salt_east(bry_time, eta_rho, s_rho) - 盐度
#   u_east(bry_time, eta_rho, s_rho)    - U速度 (ROMS C-grid)
#   v_east(bry_time, eta_rho, s_rho)    - V速度 (ROMS C-grid)
#   ubar_east(bry_time, eta_rho)        - 正压U速度
#   vbar_east(bry_time, eta_rho)        - 正压V速度

# ===================================================================
# 7. 应用配置 / Apply Configuration & Run
# ===================================================================

if __name__ == '__main__':
    # 将配置写入模块变量 (Write config to module variables)
    cfg.DATA_DIR = CMEMS_DIR
    cfg.GRD_NAME = GRID_FILE
    cfg.BRY_NAME = BRY_FILE
    cfg.BOUNDARY = BOUNDARY
    cfg.N_LEVELS = N_LEVELS
    cfg.VTRANSFORM = VTRANSFORM
    cfg.VSTRETCHING = VSTRETCHING
    cfg.THETA_S = THETA_S
    cfg.THETA_B = THETA_B
    cfg.TCLINE = TCLINE
    cfg.CMEMS_FILE = CMEMS_FILE
    cfg.CMEMS_MERGED_PATTERN = CMEMS_MERGED_PATTERN
    cfg.TIME_START = TIME_START
    cfg.TIME_END = TIME_END
    cfg.ROMS_TIME_REF = ROMS_TIME_REF
    cfg.FILL_VAL = FILL_VAL

    cmems_bry_main()
