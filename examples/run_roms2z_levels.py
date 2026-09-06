"""
========================================================================
  ROMS Sigma -> Z-Level 垂直坐标转换模板
  ROMS Sigma-Coordinate to Standard Z-Levels Remapping Template
========================================================================

功能: 将 ROMS sigma 坐标场插值到标准深度层
Input:  ROMS 输出文件 (IC/BC/history/average)
Output: 标准 z-level NetCDF 文件

用途:
  - 可视化 (标准深度等值线图)
  - 与非 ROMS 模式/观测数据的比较
  - 数据后处理与分析

插值方法:
  - 逐点一维插值 (scipy interp1d)
  - 表层处理: 浅于最上 sigma 层的深度直接用顶层值 (避免表层NaN)
  - 深层处理: 深于最下 sigma 层的深度设为 NaN

用法:
    python roms2z_levels.py

CLI用法 (batch processing):
    python -m roms_prepro.remapping.roms2z_levels -i ocean_avg_*.nc -d ./output/ --depths 0 10 50 100 200

依赖:
    numpy, scipy, netCDF4, roms_prepro

Verified: 2025-07-06 (NSCS domain)
========================================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roms_prepro.remapping import process_file, DEFAULT_STD_DEPTHS
import numpy as np

# ===================================================================
# 1. 输入/输出路径 / Input/Output Paths
# ===================================================================

# 输入 ROMS 文件
# Input ROMS file (IC, BC, history, average, etc.)
INPUT_FILE = "/data/hdl/roms_prepro/examples/NSCS_ini20250522.nc"

# 输出文件路径
# Set to None to auto-generate: <input_basename><suffix><ext>
OUTPUT_FILE = "/data/hdl/roms_prepro/examples/NSCS_ini20250522_z.nc"

# 文件名后缀 (当 OUTPUT_FILE=None 时使用)
# File suffix when auto-generating output name
SUFFIX = '_z'

# ===================================================================
# 2. 深度层配置 / Depth Configuration
# ===================================================================

# -------------------------------------------------------------------
# 方式 A: 使用默认深度层 (DEFAULT_STD_DEPTHS, 0-5500m)
# 默认层数较多，适合全深度分析
# -------------------------------------------------------------------
USE_DEFAULT_DEPTHS = True

# -------------------------------------------------------------------
# 方式 B: 自定义深度层 (Custom depth levels)
# 设为 USE_DEFAULT_DEPTHS = False 并修改 CUSTOM_DEPTHS
# -------------------------------------------------------------------
CUSTOM_DEPTHS = [
    0, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000
]

# 快速生成等间隔深度:
# CUSTOM_DEPTHS = list(np.arange(0, 5000, 50))  # 每50m一层

# 快速生成双层加密 (表层密、深层疏):
# CUSTOM_DEPTHS = list(np.arange(0, 100, 5)) + list(np.arange(100, 5000, 100))

# ===================================================================
# 3. 处理选项 / Processing Options
# ===================================================================

# 要处理的变量: None = 默认 ['temp', 'salt', 'u', 'v', 'zeta']
#   3D 变量 (s_rho) 插值到 z 层; 2D 变量 (如 zeta 表层) 直接拷贝
# 或显式指定/裁剪，如 ['temp', 'salt'] / ['temp', 'salt', 'rho', 'AKv']
VARIABLES = None

# 垂直坐标参数覆盖: None = 自动从输入文件读取
# 只需给出要覆盖的键，如 {'N': 32, 'theta_s': 5.0}
VGRID_PARAMS = None

# 水平网格: True = 全部变量换到 rho 网格（默认，输出更精简，坐标为 lon/lat）
#           False = 保留原交错网格（u 在 eta_u/xi_u，坐标为 lon_rho/lon_u/...）
TO_RHO = True

# 垂直参数 (自动从输入文件读取)
# Vtransform, Vstretching, theta_s, theta_b, N are auto-detected from input file

# 深度过滤: 自动去除大于最大水深的深度层
# Depths deeper than max bathymetry are automatically filtered out

# ===================================================================
# 4. 运行 / Run
# ===================================================================

if __name__ == '__main__':
    # 选择深度层
    if USE_DEFAULT_DEPTHS:
        std_depths = None  # 使用默认 (DEFAULT_STD_DEPTHS)
        depth_label = 'default'
    else:
        std_depths = CUSTOM_DEPTHS
        depth_label = f'custom ({len(CUSTOM_DEPTHS)} levels)'

    print(f'ROMS Sigma -> Z-Level Interpolation')
    print(f'  Input:  {INPUT_FILE}')
    print(f'  Output: {OUTPUT_FILE}')
    print(f'  Depth levels: {depth_label}')
    print()

    result = process_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        std_depths=std_depths,
        suffix=SUFFIX,
        variables=VARIABLES,
        vgrid_params=VGRID_PARAMS,
        to_rho=TO_RHO,
    )

    if result:
        print(f'\nSuccess! Output: {result}')
    else:
        print('\nFailed!')


# ===================================================================
# 5. CLI 批处理示例 / CLI Batch Processing Examples
# ===================================================================
#
# 单文件:
#   python -m roms_prepro.remapping.roms2z_levels \
#       -i NSCS_ini20250522.nc -o output_z.nc
#
# 多文件批量:
#   python -m roms_prepro.remapping.roms2z_levels \
#       -i ocean_avg_0001.nc ocean_avg_0002.nc -d ./z_output/
#
# 自定义深度:
#   python -m roms_prepro.remapping.roms2z_levels \
#       -i ocean_avg_*.nc -d ./z_output/ \
#       --depths 0 10 20 50 100 200 500 1000 2000 3000
#
# 并行处理 (多文件):
#   python -m roms_prepro.remapping.roms2z_levels \
#       -i ocean_avg_*.nc -d ./z_output/ -j 8
#
# ===================================================================
