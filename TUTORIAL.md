# ROMS Preprocessing Toolkit 使用教程

## 概述

ROMS Preprocessing Toolkit 是一个用于 ROMS (Regional Ocean Modeling System) 前处理的 Python 工具包，支持：

- **网格生成** — 从 GEBCO 水深数据创建 ROMS 网格
- **初始条件 (IC)** — 从 CMEMS/Mercator/HYCOM 或其他 ROMS 运行创建初始场
- **边界条件 (BC)** — 创建时间依赖的边界场
- **大气强迫** — 从 ERA5 数据创建大气强迫场
- **潮汐强迫** — 从 TPXO8 数据创建潮汐强迫
- **河流输入** — 创建河流强迫场
- **坐标转换** — sigma ↔ z 坐标插值

## 环境配置

```bash
conda env create -f environment.yml
conda activate roms_prepro
```

## 输入文件要求

### CMEMS 数据

CMEMS (Copernicus Marine Environment Monitoring Service) 数据支持以下几种组织方式：

1. **单文件模式**：一个文件包含所有变量（zos, thetao, so, uo, vo）
2. **多文件模式**：每个变量一个文件（CMEMS 标准下载格式）
3. **目录模式**：指定目录，自动搜索匹配的文件

**文件命名模式（自动识别）**：

| 变量 | 文件名模式 |
|------|-----------|
| SSH (zos) | `cmems_zos_*.nc`, `cmems_merged_*.nc` |
| 温度 (thetao) | `cmems_thetao_*.nc`, `cmems_merged_*.nc` |
| 盐度 (so) | `cmems_so_*.nc`, `cmems_merged_*.nc` |
| U流速 (uo) | `cmems_uo_*.nc`, `cmems_merged_*.nc` |
| V流速 (vo) | `cmems_vo_*.nc`, `cmems_merged_*.nc` |

**变量名（自动识别）**：

| 标准名 | 支持的变量名 |
|--------|-------------|
| zos | zos, sla, adt |
| thetao | thetao, t, temperature, votemper |
| so | so, s, salinity, vosaline |
| uo | uo, u, water_u, vozocrtx |
| vo | vo, v, water_v, vomecrty |

**维度名（自动识别）**：

| 维度 | 支持的名字 |
|------|-----------|
| 经度 | longitude, lon, nav_lon |
| 纬度 | latitude, lat, nav_lat |
| 深度 | depth, lev, deptht |
| 时间 | time, time_counter |

### ROMS 数据（用于 ROMS→ROMS 转换）

ROMS 历史文件应包含以下变量：`temp`, `salt`, `zeta`, `u`, `v`, `angle`
以及维度：`ocean_time`, `s_rho`, `eta_rho`, `xi_rho` 等。

**注意**：所有功能均支持以下增强特性：
- **自动时间参考检测**：`time_ref` 参数自动从源文件的时间变量单位属性读取，无需手动设置
- **日期选择 IC**：通过 `init_date` 参数指定制作哪一天的 IC，自动查找匹配的时间步
- **目录自动搜索**：支持指定目录路径，自动搜索匹配的 `.nc` 文件

包目录结构：

```
roms_prepro/
├── grid/          # 网格生成
├── ic/            # 初始场
├── bc/            # 边界场
├── forcing/       # ERA5 大气强迫
├── tide/          # 潮汐强迫
├── river/         # 河流输入
└── remapping/     # sigma↔z 坐标转换
```

## 快速开始

### 1. 网格生成

```python
from roms_prepro.grid import create_roms_grid

create_roms_grid(
    lon_corners=[119.0, 119.0, 125.7, 125.7],
    lat_corners=[33.5, 37.1, 37.1, 33.5],
    nx=300, ny=200,
    grid_file='my_grid.nc',
    source_lon=slon, source_lat=slat, source_depth=sdep,
    min_depth=5.0, rx0max=0.2,
    vgrid_params={'Vtransform': 2, 'Vstretching': 4}
)
```

### 2. 初始条件 (IC)

#### 方式一：单文件 (Mercator/HYCOM/CMEMS)

适用于包含所有变量的单个 NetCDF 文件：

```python
from roms_prepro.ic import mercator_to_roms_ini

mercator_to_roms_ini(
    roms_grid_file='my_grid.nc',
    source_file='cmems_data.nc',
    ini_file='my_ini.nc',
    Vtransform=2, Vstretching=4,
    theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
    init_date='2025-01-01',         # 指定制作哪一天的 IC
    # time_ref=None,                 # 自动从源文件时间单位读取
)
```

#### 方式二：CMEMS 分变量文件

适用于 CMEMS 下载的分变量文件。支持三种用法：

**用法 A：指定目录自动搜索（推荐）**

```python
from roms_prepro.ic import cmems_to_roms_ini

cmems_to_roms_ini(
    roms_grid_file='my_grid.nc',
    ini_file='my_ini_cmems.nc',
    data_dir='/path/to/CMEMS/',
    Vtransform=2, Vstretching=3,
    theta_s=2.5, theta_b=1.0, Tcline=25.0, N=30,
    init_date='2025-05-01',         # 指定制作哪一天的 IC
    # time_ref=None,                 # 自动从 CMEMS 文件时间单位读取
)
```

**用法 B：指定各变量文件路径**

```python
from roms_prepro.ic import cmems_to_roms_ini

cmems_to_roms_ini(
    roms_grid_file='my_grid.nc',
    ini_file='my_ini_cmems.nc',
    zeta_file='zos_20250101.nc',
    temp_file='thetao_20250101.nc',
    salt_file='so_20250101.nc',
    u_file='uo_20250101.nc',
    v_file='vo_20250101.nc',
    Vtransform=2, Vstretching=3,
    theta_s=2.5, theta_b=1.0, Tcline=25.0, N=30,
    init_date='2025-05-01',
)
```

**用法 C：指定单文件（所有变量在一个文件中）**

```python
from roms_prepro.ic import cmems_to_roms_ini

cmems_to_roms_ini(
    roms_grid_file='my_grid.nc',
    ini_file='my_ini_cmems.nc',
    zeta_file='cmems_all.nc',      # 所有变量在同一个文件中
    temp_file='cmems_all.nc',
    salt_file='cmems_all.nc',
    u_file='cmems_all.nc',
    v_file='cmems_all.nc',
    Vtransform=2, Vstretching=3,
    theta_s=2.5, theta_b=1.0, Tcline=25.0, N=30,
    init_date='2025-05-01',
)
```

#### 方式三：从其他 ROMS 运行创建

支持指定单个文件或目录自动搜索：

```python
from roms_prepro.ic import roms_to_roms_ini

# 方式 A：指定目录自动搜索
roms_to_roms_ini(
    src_grid_file='parent_grid.nc',
    dst_grid_file='my_grid.nc',
    ini_file='my_ini_from_parent.nc',
    src_hist_dir='./parent_output/',     # 自动搜索 *.nc 文件
    Vtransform=2, Vstretching=4,
    theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
    init_date='2020-01-15',              # 指定制作哪一天的 IC
    # time_ref=None,                      # 自动从 ocean_time 单位读取
)

# 方式 B：指定单个文件
roms_to_roms_ini(
    src_grid_file='parent_grid.nc',
    dst_grid_file='my_grid.nc',
    ini_file='my_ini_from_parent.nc',
    src_hist_file='parent_hist.nc',
    init_date='2020-01-15',
)
```

### 3. 边界条件 (BC)

适用于 CMEMS 下载的分变量文件（zos, thetao, so, uo, vo）：

```python
import roms_prepro.bc.d_obc_cmems as cfg
from roms_prepro.bc.d_obc_cmems import main as cmems_bry_main

cfg.DATA_DIR = '/path/to/CMEMS/2025'
cfg.GRD_NAME = 'my_grid.nc'
cfg.BRY_NAME = 'my_bry_cmems.nc'

cfg.BOUNDARY = [0, 1, 1, 0]  # [W, E, S, N]

cfg.N_LEVELS = 30
cfg.VTRANSFORM = 2
cfg.VSTRETCHING = 3
cfg.THETA_S = 2.5
cfg.THETA_B = 1.0
cfg.TCLINE = 25.0

cfg.TIME_START = '2025-05-01 00:00:00'
cfg.TIME_END = '2025-09-30 23:00:00'
cfg.ROMS_TIME_REF = '1990-01-01 00:00:00'

cmems_bry_main()
```

#### 边界条件：从其他 ROMS 运行创建

```python
from roms_prepro.bc import roms_to_roms_bry

# 方式 A：指定目录自动搜索
roms_to_roms_bry(
    src_grid_file='parent_grid.nc',
    dst_grid_file='my_grid.nc',
    bry_file='my_bry_from_parent.nc',
    src_hist_dir='./parent_output/',
    Vtransform=2, Vstretching=4,
    theta_s=7.0, theta_b=0.1, Tcline=20.0, N=30,
    boundaries=(True, True, True, True),
    start_date='2020-01-01',
    end_date='2020-01-31',
    # time_ref=None,                      # 自动从 ocean_time 单位读取
)

# 方式 B：指定文件列表
roms_to_roms_bry(
    src_grid_file='parent_grid.nc',
    dst_grid_file='my_grid.nc',
    bry_file='my_bry_from_parent.nc',
    src_hist_files=['ocean_avg_001.nc', 'ocean_avg_002.nc'],
    start_date='2020-01-01',
    end_date='2020-01-31',
)
```

### 4. 大气强迫 (ERA5)

```python
from roms_prepro.forcing import ERA5toROMS

conv = ERA5toROMS(
    in_dir='/path/to/era5/',        # ERA5 GRIB/NC 文件目录
    out_dir='./forcing_output',
    time_start='2024-12-27 00:00:00',
    time_end='2025-02-20 23:00:00',
    base_date='1990-01-01 00:00:00',  # ROMS 时间基准
    grid_file=None,                 # None = 保持 ERA5 原始网格
    variables=['u10', 'v10', 't2m', 'd2m', 'msl', 'tp',
               'msdrswrf', 'msdwlwrf'],
    rotate_wind=False,
)
conv.process()
```

配置模板见 `roms_prepro/forcing/d_era2roms.py`。

### 5. 潮汐强迫 (TPXO8)

```python
from roms_prepro.tide import tpxo_to_roms_tide

tpxo_to_roms_tide(
    'my_grid.nc',
    'my_tide.nc',
    t0='2000-01-01',           # 相位参考时间
    ndays=365,                 # 模拟长度（nodal 因子取 t0+ndays/2）
    tpxo_dir='/path/to/tpxo/',
    constituents=['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'],
)
```

### 6. 河流输入

可用 `yangtze_river()` / `huanghe_river()` 快速生成内置月气候态河流字典，
或手工构造 `{'name', 'I', 'J', 'discharge', 'salt', 'direction'}` 字典。

```python
from roms_prepro.river import create_river_file

create_river_file(
    roms_grid_file='my_grid.nc',
    out_file='my_river.nc',
    rivers=rivers,  # 河流数据列表（I/J 网格索引 + discharge 等）
)
```

### 7. sigma → z 坐标转换

单变量（内存中的数组）：

```python
from roms_prepro.remapping import sigma_to_z_levels
import numpy as np

z_levels = np.arange(-5, -500, -5)
var_z = sigma_to_z_levels(
    var_sigma, 'my_grid.nc', z_levels,
    Vtransform=2, Vstretching=4, N=30
)
```

整文件一键转换（垂直参数自动从文件读取，可覆盖；深度自动过滤超过最大水深的层）：

```python
from roms_prepro.remapping import process_file

process_file('ocean_his_0001.nc', 'ocean_his_0001_z.nc')   # 全部默认

process_file(                          # 指定变量 / 深度 / 垂直参数
    'ocean_his_0001.nc', 'out_z.nc',
    variables=['temp', 'salt'],
    std_depths=[0, 10, 25, 50, 100, 200, 500, 1000],
    vgrid_params={'Vtransform': 2, 'Vstretching': 4,
                  'theta_s': 7.0, 'theta_b': 0.1, 'Tcline': 20.0, 'N': 30},
)
```

CLI 批处理：

```bash
python -m roms_prepro.remapping.roms2z_levels -i 'ocean_avg_*.nc' -d ./z_out/ \
    --depths 0 10 50 100 500 --vars temp salt
```

## 使用 main.py

`main.py` 是一个完整的工作流脚本，包含所有功能的示例：

```bash
# 运行所有步骤
python main.py

# 运行特定步骤
python main.py grid           # 网格生成
python main.py ic_mercator    # 单文件 IC
python main.py ic_cmems       # 多文件 CMEMS IC
python main.py bry_cmems      # CMEMS 边界条件
python main.py ic_roms2roms   # ROMS→ROMS IC
python main.py bry_roms2roms  # ROMS→ROMS BC
python main.py forcing_era5   # ERA5 大气强迫
```

## 常用参数说明

### 垂直坐标参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `Vtransform` | 垂直变换方程 | 2 |
| `Vstretching` | 拉伸函数 | 4 |
| `theta_s` | 表面控制参数 | 7.0 |
| `theta_b` | 底部控制参数 | 0.1 |
| `Tcline` | 拉伸层宽度 (m) | 20.0 |
| `N` | 垂直层数 | 30 |

### 时间参考

`time_ref` 参数指定输出文件的时间参考：

```python
# 格式 1：完整单位字符串
time_ref = 'seconds since 2000-01-01 00:00:00'

# 格式 2：仅日期（用于 cmems_to_roms_ini）
time_ref = '1990-01-01'
```

### 边界条件

`boundaries` 参数是一个 4 元素的布尔元组，表示 [西, 东, 南, 北] 边界：

```python
boundaries = (True, True, True, True)   # 所有边界
boundaries = (False, True, True, True)  # 无西边界
```

## 数据源

### CMEMS (Copernicus Marine Environment Monitoring Service)

- 网址：https://marine.copernicus.eu/
- 变量：zos (SSH), thetao (温度), so (盐度), uo/vo (流速)
- 格式：每个变量一个文件

### HYCOM (Hybrid Coordinate Ocean Model)

- 网址：https://www.hycom.org/
- 格式：单文件包含所有变量

### ERA5

- 网址：https://cds.climate.copernicus.eu/
- 变量：u10, v10, t2m, msl, msdwlwrf, msnswrf, tp, d2m

## 注意事项

1. **内存需求**：处理大区域时建议 ≥ 16GB 内存
2. **网格分辨率**：建议 nx, ny 不超过 500×500
3. **垂直层数**：通常 N=30 足够
4. **时间格式**：日期格式为 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'

## 故障排除

### 常见错误

1. **内存不足**
   - 减小网格分辨率
   - 减少时间步数

2. **插值错误**
   - 检查源数据覆盖范围
   - 确保网格在源数据范围内

3. **垂直坐标错误**
   - 检查 Vtransform 和 Vstretching 设置
   - 确保 Tcline 合理

## 示例脚本

参考 `examples/` 目录中的示例脚本。
