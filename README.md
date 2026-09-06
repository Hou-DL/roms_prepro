# ROMS Preprocessing Toolkit

<!-- 中文 -->

**Demo 版本** — 适用于 [ROMS](https://www.myroms.org/) (Regional Ocean Modeling System) 的前处理工具包，覆盖网格生成、初始场/边界场、大气强迫、潮汐强迫、河流输入、sigma↔z 坐标转换的全流程数据准备。

<!-- English -->

**Demo version** — A preprocessing toolkit for [ROMS](https://www.myroms.org/) (Regional Ocean Modeling System). Supports the full workflow: grid generation, initial/boundary conditions, atmospheric forcing, tidal forcing, river input, and sigma↔z coordinate remapping.

---

## 模块 / Modules

| Module | 中文功能 | Description |
|--------|----------|-------------|
| `grid/` | ROMS 网格生成与质量控制 | Grid generation and quality control |
| `ic/` | 初始场 (IC) — CMEMS/Mercator/HYCOM 或 ROMS→ROMS | Initial conditions from CMEMS/Mercator/HYCOM or another ROMS run |
| `bc/` | 边界场 (BC) — CMEMS→ROMS 或 ROMS→ROMS | Boundary conditions from CMEMS or another ROMS run |
| `forcing/` | ERA5 大气强迫 (风、气压、辐射、降水、湿度) | ERA5 atmospheric forcing (wind, pressure, radiation, precipitation, humidity) |
| `tide/` | TPXO8 潮汐强迫 (8 主要分潮) | TPXO8 tidal forcing (8 major constituents) |
| `river/` | 河流输入 (月均流量) | River forcing (monthly climatology) |
| `remapping/` | sigma↔z 坐标插值 (网格场 + 站点) | Sigma↔z coordinate interpolation (gridded fields and station profiles) |

---

## 快速开始 / Quick Start

### 环境配置 / Environment setup

```bash
# 从 environment.yml 创建 conda 环境 / Create conda environment
conda env create -f environment.yml
conda activate roms_prepro
```

或者手动 / Or manually:

```bash
conda create -n roms_prepro python=3.10
conda activate roms_prepro
pip install numpy scipy netCDF4 xarray tqdm
```

### 代码示例 / Code example

```python
from roms_prepro.grid import create_roms_grid
from roms_prepro.ic import cmems_to_roms_ini, mercator_to_roms_ini, roms_to_roms_ini
from roms_prepro.forcing import ERA5toROMS
from roms_prepro.tide import tpxo_to_roms_tide
from roms_prepro.river import create_river_file
from roms_prepro.remapping import sigma_to_z_levels, process_file

# 1. 网格 / Grid
# create_roms_grid(lon_corners=[...], lat_corners=[...], nx=300, ny=200,
#                  grid_file='my_grid.nc', source_lon=..., source_lat=...,
#                  source_depth=..., min_depth=5.0, rx0max=0.2)

# 2. 初始场 / Initial conditions
# cmems_to_roms_ini('my_grid.nc', 'my_ini.nc', data_dir='/path/to/CMEMS/',
#                   Vtransform=2, Vstretching=3, theta_s=2.5, theta_b=1.0,
#                   Tcline=25.0, N=30, init_date='2025-05-01')
# mercator_to_roms_ini('my_grid.nc', 'src.nc', 'ini.nc', N=30)   # 单文件源
# roms_to_roms_ini(src_grid, src_hist, dst_grid, 'ini.nc')       # ROMS→ROMS

# 3. 边界场 / Boundary conditions
# import roms_prepro.bc.d_obc_cmems as cfg                      # CMEMS→ROMS
# from roms_prepro.bc.d_obc_cmems import main as cmems_bry_main
# cfg.DATA_DIR = '/path/to/CMEMS/'; cfg.GRD_NAME = 'my_grid.nc'
# cfg.BRY_NAME = 'my_bry.nc'; cmems_bry_main()
# from roms_prepro.bc import roms_to_roms_bry                   # ROMS→ROMS

# 4. ERA5 大气强迫 / Atmospheric forcing
# conv = ERA5toROMS(in_dir='/path/to/era5/', out_dir='./out',
#                   base_date='1990-01-01 00:00:00')
# conv.process()

# 5. TPXO8 潮汐 / Tidal forcing (8 constituents)
# tpxo_to_roms_tide('my_grid.nc', 'tide.nc', t0='2000-01-01', ndays=365,
#                   tpxo_dir='/path/to/tpxo')

# 6. 河流 / River forcing
# create_river_file('my_grid.nc', 'river.nc', rivers)

# 7. sigma → z 坐标转换 / Coordinate remapping
# import numpy as np
# zlevs = np.arange(-5, -500, -5)
# var_z = sigma_to_z_levels(var_sigma, 'my_grid.nc', zlevs, N=30)
# process_file('ocean_his_0001.nc', 'ocean_his_0001_z.nc')      # 整文件转换 + CLI
```

`main.py` 提供全部步骤的参数模板（`python main.py grid|ic_cmems|bry_cmems|...`）；
可运行的配置模板见 `examples/` 目录。

---

## 依赖 / Dependencies

| Package | 用途 | Purpose |
|---------|------|---------|
| Python ≥ 3.9 | — | — |
| numpy | 数值计算 | Numerical arrays |
| netCDF4 | NetCDF 文件读写 | NetCDF I/O |
| scipy | 插值 | Interpolation |
| tqdm | 进度条 | Progress bars |
| xarray, cfgrib | ERA5 (GRIB/NC) 数据读取 | ERA5 dataset handling |

---

## 说明 / Notes

### 中文
- 垂直坐标 stretching/set_depth 与 pyroms（ROMS set_scoord）数值一致，支持 Vstretching 1-4（Vstretching=5 为均匀 s 变体）
- **ERA5 风场** — 默认不旋转到曲线网格坐标（`rotate_wind=True` 可开启）
- **ERA5 累积量** — 辐射 (J/m²) 与降水 (m) 自动按数据步长换算为通量
- **TPXO 潮汐** — 默认 8 个标准分潮 (M2, S2, N2, K2, K1, O1, P1, Q1)
- **河流输入** — 基于月气候态流量，含 river_flag/Udirection/Vdirection 等 ROMS 必需变量
- **坐标转换** — `remapping` 模块支持网格场 (`sigma_to_z_levels`)、整文件 (`process_file`) 和站点数据 (`station_to_z_levels`)

### English
- Vertical stretching/set_depth match pyroms (ROMS set_scoord); Vstretching 1-4 supported (Vstretching=5 uses a uniform-s variant)
- **ERA5 wind** — no rotation to the curvilinear grid by default (`rotate_wind=True` to enable)
- **ERA5 accumulations** — radiation (J/m²) and precipitation (m) auto-converted to fluxes using the data time step
- **TPXO tides** — 8 standard constituents (M2, S2, N2, K2, K1, O1, P1, Q1)
- **River input** — monthly climatological discharge with the ROMS-required river_flag/Udirection/Vdirection variables
- **Remapping** — `sigma_to_z_levels` for gridded fields, `process_file` for whole files, `station_to_z_levels` for station profiles

---

## 注意事项 / Caveats

- 大文件处理需要足够内存（建议 ≥ 16GB） / Large file processing requires adequate RAM (≥ 16 GB recommended)
- CDS API 下载 ERA5 数据需单独配置（本机无法直接下载） / ERA5 download via CDS API requires local setup (not available on this server)
- TPXO8 潮汐数据需自行获取 / TPXO8 tidal data must be obtained separately
