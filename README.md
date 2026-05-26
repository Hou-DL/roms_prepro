# ROMS Preprocessing Toolkit (Demo)

**Demo Version** — 适用于 ROMS (Regional Ocean Modeling System) 的前处理工具包，支持网格、初始场/边界场、大气强迫、潮汐强迫、河流输入的全流程数据准备。

## 功能模块

| 模块 | 功能 |
|------|------|
| `roms_prepro/grid/` | ROMS 网格生成与质量控制 |
| `roms_prepro/icbc/` | 初始场 (IC) 和边界场 (BC) — 支持 CMEMS/Mercator/HYCOM 数据源 |
| `roms_prepro/forcing/` | ERA5 大气强迫场 (风、气压、辐射、降水、湿度) |
| `roms_prepro/remapping/` | 潮汐强迫 (TPXO8) |
| `roms_prepro/river/` | 河流输入 (月均流量) |
| `roms_prepro/utils/` | 工具函数 |

## 快速示例

```python
from roms_prepro.grid import make_grid
from roms_prepro.icbc import mercator_to_roms_ini, mercator_to_roms_bry
from roms_prepro.forcing.era5 import make_forcing

# 网格
make_grid(bathy_file, 'my_grid.nc')

# 初始场
mercator_to_roms_ini('my_grid.nc', 'mercator_data.nc', 'ini.nc',
                     init_date='2025-01-03', N=30)

# 边界场
mercator_to_roms_bry('my_grid.nc', ['mercator_data.nc'], 'bry.nc',
                     start_date='2025-01-01', end_date='2025-01-31', N=30)

# ERA5 强迫
make_forcing('my_grid.nc', 'era5_data.nc', 'forcing.nc',
             date_range=['2025-01-01', '2025-01-31'])
```

## 依赖

- Python ≥ 3.9
- numpy, netCDF4, scipy
- (可选) xarray, dask — 用于 ERA5 数据处理

## 说明

- Demo 版本功能完整，可用于实际 ROMS 模拟前处理
- ERA5 风场使用原始 eastward/northward 分量，不进行角度旋转
- TPXO 潮汐支持主要分潮 (M2, S2, K1, O1 等)
- 河流输入基于月气候态流量

## 注意事项

- 大文件处理需要足够内存（建议 ≥ 16GB）
- CDS API 下载 ERA5 数据需单独配置
- 潮汐数据 TPXO8 需自行获取
