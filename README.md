# ROMS Preprocessing Toolkit

<!-- 中文 -->

**Demo 版本** — 适用于 [ROMS](https://www.myroms.org/) (Regional Ocean Modeling System) 的前处理工具包，覆盖网格生成、初始场/边界场、大气强迫、潮汐强迫、河流输入、sigma↔z 坐标转换的全流程数据准备。

<<<<<<< HEAD
| 模块 | 功能 |
|------|------|
| `roms_prepro/grid/` | ROMS 网格生成与质量控制 |
| `roms_prepro/icbc/` | 初始场 (IC) 和边界场 (BC) — 支持 ROMS/Mercator/HYCOM 数据源 |
| `roms_prepro/forcing/` | ERA5 大气强迫场 (风、气压、辐射、降水、湿度) |
| `roms_prepro/tide/` | 潮汐强迫 (TPXO8) |
| `roms_prepro/river/` | 河流输入 (月均流量) |
| `roms_prepro/utils/` | 工具函数 |
=======
---
>>>>>>> 34158e6 (Add sta2z, bilingual README, update main.py)

<!-- English -->

**Demo version** — A preprocessing toolkit for [ROMS](https://www.myroms.org/) (Regional Ocean Modeling System). Supports the full workflow: grid generation, initial/boundary conditions, atmospheric forcing, tidal forcing, river input, and sigma↔z coordinate remapping.

---

## 模块 / Modules

| Module | 中文功能 | Description |
|--------|----------|-------------|
| `grid/` | ROMS 网格生成与质量控制 | Grid generation and quality control |
| `icbc/` | 初始场 (IC) 和边界场 (BC) — CMEMS/Mercator/HYCOM | Initial and boundary conditions from CMEMS/Mercator/HYCOM |
| `forcing/` | ERA5 大气强迫 (风、气压、辐射、降水、湿度) | ERA5 atmospheric forcing (wind, pressure, radiation, precipitation, humidity) |
| `tide/` | TPXO8 潮汐强迫 (8 主要分潮) | TPXO8 tidal forcing (8 major constituents) |
| `river/` | 河流输入 (月均流量) | River forcing (monthly climatology) |
| `remapping/` | sigma↔z 坐标插值 (网格场 + 站点) | Sigma↔z coordinate interpolation (gridded fields and station profiles) |
| `utils/` | 工具函数 | Utility functions |

---

## 快速开始 / Quick Start

```python
from roms_prepro.grid import create_roms_grid
from roms_prepro.icbc import mercator_to_roms_ini, mercator_to_roms_bry
from roms_prepro.forcing import era5_to_roms_forcing
from roms_prepro.tide import tpxo_to_roms_tide
from roms_prepro.river import create_river_file
from roms_prepro.remapping import sigma_to_z_levels, station_to_z_levels

# 1. 网格 / Grid
# create_roms_grid(bathy_file, 'my_grid.nc', ...)

# 2. 初始场 / Initial conditions
# mercator_to_roms_ini('my_grid.nc', 'src.nc', 'ini.nc', N=30)

# 3. 边界场 / Boundary conditions
# mercator_to_roms_bry('my_grid.nc', ['src.nc'], 'bry.nc',
#                      start_date='2025-01-01', end_date='2025-01-31', N=30)

# 4. ERA5 大气强迫 / Atmospheric forcing
# era5_to_roms_forcing('my_grid.nc', era5_files, 'forcing.nc',
#                      start_date='2025-01-01', end_date='2025-01-31')

# 5. TPXO8 潮汐 / Tidal forcing (8 constituents)
# tpxo_to_roms_tide('my_grid.nc', '/path/to/tpxo', 'tide.nc')

# 6. 河流 / River forcing
# create_river_file('my_grid.nc', rivers, 'river.nc')

# 7. sigma → z 坐标转换 / Coordinate remapping
# import numpy as np
# zlevs = np.arange(-5, -500, -5)
# var_z = sigma_to_z_levels(var_sigma, 'my_grid.nc', zlevs, N=30)
```

---

## 依赖 / Dependencies

| Package | 用途 | Purpose |
|---------|------|---------|
| Python ≥ 3.9 | — | — |
| numpy | 数值计算 | Numerical arrays |
| netCDF4 | NetCDF 文件读写 | NetCDF I/O |
| scipy | 插值 | Interpolation |
| tqdm | 进度条 | Progress bars |
| (可选) xarray, dask | ERA5 大数据处理 | Large ERA5 dataset handling |

---

## 说明 / Notes

### 中文
- **Demo 版本** — 功能完整，可用于实际 ROMS 模拟前处理
- **ERA5 风场** — 使用原始 eastward/northward 分量，**不做角度旋转**
- **TPXO 潮汐** — 默认 8 个标准分潮 (M2, S2, N2, K2, K1, O1, P1, Q1)，M4 可通过 `constituents` 参数手动添加
- **河流输入** — 基于月气候态流量
- **坐标转换** — `remapping` 模块支持网格场 (`sigma_to_z_levels`) 和站点数据 (`station_to_z_levels`)

### English
- **Demo version** — Fully functional for real ROMS preprocessing
- **ERA5 wind** — Uses raw eastward/northward components, **no grid-angle rotation**
- **TPXO tides** — Default 8 standard constituents (M2, S2, N2, K2, K1, O1, P1, Q1); M4 available via the `constituents` parameter
- **River input** — Based on monthly climatological discharge
- **Remapping** — `sigma_to_z_levels` for gridded fields, `station_to_z_levels` for station profiles

---

## 注意事项 / Caveats

- 大文件处理需要足够内存（建议 ≥ 16GB） / Large file processing requires adequate RAM (≥ 16 GB recommended)
- CDS API 下载 ERA5 数据需单独配置（本机无法直接下载） / ERA5 download via CDS API requires local setup (not available on this server)
- TPXO8 潮汐数据需自行获取 / TPXO8 tidal data must be obtained separately