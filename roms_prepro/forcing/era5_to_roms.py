"""
ERA5 to ROMS Forcing Converter (精简版)
========================================
参考 era5_2roms.m (Wang, 2023/6/30)

支持:
- GRIB/NC文件自动识别
- 多文件自动发现（不同变量可在不同文件中）
- 自动读取输入文件的经纬度范围
- 插值到ROMS网格（带风场旋转）或规则经纬度网格

使用:
    from era5_to_roms import ERA5toROMS
    converter = ERA5toROMS(
        in_dir='path/to/era5/data',
        out_dir='path/to/output',
        time_start='2025-05-01 00:00:00',
        time_end='2025-05-31 23:00:00',
        base_date='1990-01-01 00:00:00',
    )
    converter.process()
"""

import os
import glob
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc
import xarray as xr
import cfgrib


# ============================================================
# ERA5变量名 -> ROMS变量名 映射
# ============================================================
# GRIB短名 -> 我们的变量名
GRIB_SHORTNAME_MAP = {
    '10u': 'u10', '10v': 'v10', '2t': 't2m', '2d': 'd2m',
    'msl': 'msl', 'tp': 'tp', 'tcc': 'tcc', 'r': 'r',
    'avg_sdlwrf': 'msdwlwrf', 'avg_snlwrf': 'msnlwrf',
    'avg_snswrf': 'msdrswrf',
}

# 我们的变量名 -> ROMS输出信息
ROMS_VARINFO = {
    'u10':    {'roms': 'Uwind',    'units': 'meter second-1',    'long_name': 'surface u-wind component',     'file': 'wind',    'time_var': 'wind_time'},
    'v10':    {'roms': 'Vwind',    'units': 'meter second-1',    'long_name': 'surface v-wind component',     'file': 'wind',    'time_var': 'wind_time'},
    't2m':    {'roms': 'Tair',     'units': 'Celsius',           'long_name': 'surface air temperature',      'file': 'forcing', 'time_var': 'tair_time', 'convert': 'K2C'},
    'd2m':    {'roms': 'd2m',      'units': 'Celsius',           'long_name': 'surface dewpoint temperature', 'file': 'forcing', 'time_var': 'tair_time', 'convert': 'K2C'},
    'msl':    {'roms': 'Pair',     'units': 'millibar',          'long_name': 'surface air pressure',         'file': 'forcing', 'time_var': 'pair_time', 'convert': 'Pa2mb'},
    'tp':     {'roms': 'rain',     'units': 'kilogram meter-2 second-1', 'long_name': 'rain fall rate',          'file': 'forcing', 'time_var': 'rain_time', 'convert': 'm2kgms'},
    'ssrd':   {'roms': 'swrad',    'units': 'Watts meter-2',     'long_name': 'net solar shortwave radiation','file': 'forcing', 'time_var': 'srf_time'},
    'msdrswrf': {'roms': 'swrad',  'units': 'Watts meter-2',     'long_name': 'net solar shortwave radiation','file': 'forcing', 'time_var': 'srf_time'},
    'msdwswrf': {'roms': 'swrad',  'units': 'Watts meter-2',     'long_name': 'mean downward shortwave radiation', 'file': 'forcing', 'time_var': 'srf_time'},
    'msdwlwrf': {'roms': 'lwrad_down','units':'Watts meter-2',   'long_name': 'downward longwave radiation',  'file': 'forcing', 'time_var': 'lrf_time'},
    'msnlwrf': {'roms': 'lwrad',   'units': 'Watts meter-2',     'long_name': 'net downward longwave radiation','file': 'forcing', 'time_var': 'lrf_time'},
    'r':      {'roms': 'Qair',     'units': 'percentage',        'long_name': 'surface air relative humidity','file': 'forcing', 'time_var': 'qair_time'},
    'tcc':    {'roms': 'cloud',    'units': 'nondimensional',    'long_name': 'cloud fraction',               'file': 'forcing', 'time_var': 'cloud_time'},
}

def _convert(data, mode):
    """单位转换"""
    if mode == 'K2C':   return data - 273.15
    if mode == 'Pa2mb': return data / 100.0
    if mode == 'm2kgms': return data / 3600.0 * 1000.0
    return data


def _numeric_time_to_datetime(time, units=''):
    """
    Convert numeric time values to datetime.

    The units attribute of the time variable ('hours since 1950-01-01',
    'days since 1900-01-01 00:00:00', ...) is authoritative. Only when it
    is missing do we fall back to magnitude-based heuristics, which cannot
    distinguish e.g. 'days since 1950' from 'hours since 1900'.
    """
    vals = np.atleast_1d(np.asarray(time))
    if units and 'since' in units:
        try:
            dates = nc.num2date(vals, units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
            return np.array([datetime(d.year, d.month, d.day,
                                      d.hour, d.minute, d.second)
                             for d in np.atleast_1d(dates)])
        except Exception:
            pass
    tv = float(vals[0])
    if tv > 1e9:      # epoch seconds
        return np.array([datetime(1970, 1, 1) + timedelta(seconds=float(x)) for x in vals])
    if tv > 100000:   # hours since 1900-ish
        return np.array([datetime(1900, 1, 1) + timedelta(hours=float(x)) for x in vals])
    if tv > 100:      # hours since 1900 (small values)
        return np.array([datetime(1900, 1, 1) + timedelta(hours=float(x)) for x in vals])
    # days since base_date (ROMS-style output, e.g. 0..30)
    return np.array([datetime(1990, 1, 1) + timedelta(days=float(x)) for x in vals])


def _read_grib_files(file_list, var_names):
    """
    从GRIB文件列表中读取指定变量
    自动处理不同变量在不同文件中的情况
    """
    data = {}
    target_grib_names = set()
    for vn in var_names:
        for grib_name, our_name in GRIB_SHORTNAME_MAP.items():
            if our_name == vn:
                target_grib_names.add(grib_name)

    # 需要在气压层读取的变量（取1000hPa）
    pressure_level_vars = {'r': 1000.0}

    for fpath in file_list:
        for gname in sorted(target_grib_names):
            if gname in data:
                continue
            try:
                # 先尝试surface类型
                try:
                    ds = xr.open_dataset(fpath, engine='cfgrib',
                        backend_kwargs={'filter_by_keys': {
                            'typeOfLevel': 'surface', 'shortName': gname}})
                except Exception:
                    # 相对湿度等变量只存在于气压层文件中
                    if gname not in pressure_level_vars:
                        continue
                    ds = None

                actual = None
                if ds is not None:
                    for v in ds.data_vars:
                        if v in ROMS_VARINFO or GRIB_SHORTNAME_MAP.get(v, v) in ROMS_VARINFO:
                            actual = v
                            break

                if actual is None:
                    if ds is not None:
                        ds.close()
                    if gname in pressure_level_vars:
                        # 尝试气压层类型
                        ds = xr.open_dataset(fpath, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {
                                'typeOfLevel': 'isobaricInhPa',
                                'shortName': gname}})
                        for v in ds.data_vars:
                            if v in ROMS_VARINFO or GRIB_SHORTNAME_MAP.get(v, v) in ROMS_VARINFO:
                                actual = v
                                break
                        if actual is None:
                            ds.close()
                            continue
                        # 取指定气压层
                        level = pressure_level_vars[gname]
                        if 'isobaricInhPa' in ds.dims:
                            ds = ds.sel(isobaricInhPa=level)
                    else:
                        continue

                vn = GRIB_SHORTNAME_MAP.get(actual, actual)
                if vn not in var_names:
                    ds.close()
                    continue

                var_data = ds[actual].values
                var_time = ds['time'].values.astype('datetime64[ms]').astype(datetime)

                # 处理step维度（累积/平均变量有step维度）
                if 'step' in ds[actual].dims and var_data.ndim == 4:
                    base_times = var_time
                    steps = ds['step'].values
                    expanded_data = []
                    expanded_time = []
                    for i in range(len(base_times)):
                        for j in range(len(steps)):
                            td = timedelta(microseconds=int(steps[j]) / 1000)
                            valid_t = base_times[i] + td
                            expanded_data.append(var_data[i, j, :, :])
                            expanded_time.append(valid_t)
                    var_data = np.array(expanded_data)
                    var_time = np.array(expanded_time)
                    valid_mask = ~np.isnan(var_data).all(axis=(1, 2))
                    var_data = var_data[valid_mask]
                    var_time = var_time[valid_mask]

                data[vn] = {
                    'lon': ds['longitude'].values,
                    'lat': ds['latitude'].values,
                    'time': var_time,
                    'data': var_data,
                }
                print(f"  {gname} -> {vn}: shape={var_data.shape}")
                ds.close()
            except Exception:
                pass
    return data


def _read_nc_files(file_list, var_names):
    """从NC文件列表中读取指定变量（自动拼接多个文件的时间维度）"""
    data = {vn: {'lon': None, 'lat': None, 'time': [], 'data': [], 'units': ''} for vn in var_names}
    for fpath in file_list:
        try:
            ds = xr.open_dataset(fpath, engine='h5netcdf')
        except Exception:
            try:
                ds = nc.Dataset(fpath)
            except Exception as e:
                print(f"  跳过 {os.path.basename(fpath)}: {e}")
                continue

        # 找坐标
        lon = lat = time = None
        for n in ['longitude', 'lon', 'x']:
            if n in ds.variables: lon = ds.variables[n][:]; break
        for n in ['latitude', 'lat', 'y']:
            if n in ds.variables: lat = ds.variables[n][:]; break
        for n in ['time', 'valid_time']:
            if n in ds.variables: time = ds.variables[n][:]; break
        if lon is None or lat is None:
            ds.close()
            continue

        # 确保是numpy数组
        if hasattr(lon, 'values'): lon = lon.values
        if hasattr(lat, 'values'): lat = lat.values
        if hasattr(time, 'values'): time = time.values

        # 时间转换
        if time is not None and hasattr(time, 'dtype') and 'datetime' in str(time.dtype):
            t = time.astype('datetime64[ms]').astype(datetime)
        elif time is not None:
            # prefer the units attribute; heuristics are only a fallback
            t_units = ''
            for n in ['time', 'valid_time']:
                if n in ds.variables:
                    tv_ = ds.variables[n]
                    t_units = (getattr(tv_, 'units', '') or
                               (tv_.attrs.get('units', '') if hasattr(tv_, 'attrs') else ''))
                    break
            try:
                t = _numeric_time_to_datetime(time, t_units)
            except Exception:
                t = None
        else:
            t = None

        # 读变量，拼接时间维度
        for vn in var_names:
            if vn not in ds.variables:
                continue
            vdata = ds.variables[vn][:]
            if hasattr(vdata, 'values'): vdata = vdata.values
            if vdata.ndim > 3:
                vdata = vdata[0]

            if data[vn]['lon'] is None:
                data[vn]['lon'] = lon
                data[vn]['lat'] = lat
                data[vn]['units'] = str(getattr(ds.variables[vn], 'units', '') or '')

            data[vn]['time'].extend(t if t is not None else np.arange(vdata.shape[0]))
            data[vn]['data'].append(vdata)
            print(f"  {vn}: +{vdata.shape[0]}步 from {os.path.basename(fpath)}")

        ds.close()

    # 合并时间维度
    for vn in var_names:
        if data[vn]['data']:
            data[vn]['time'] = np.array(data[vn]['time'])
            data[vn]['data'] = np.concatenate(data[vn]['data'], axis=0)
            print(f"  {vn}: 合并后 shape={data[vn]['data'].shape}")
        else:
            del data[vn]
    return data


def _interpolate(data_3d, lon_in, lat_in, lon_rho, lat_rho):
    """将 (ntime, nlat, nlon) 数据插值到目标网格，返回 (ntime, Lp, Mp)"""
    # 确保升序
    if np.diff(lat_in).mean() < 0:
        lat_in = lat_in[::-1]
        data_3d = data_3d[:, ::-1, :]
    if np.diff(lon_in).mean() < 0:
        lon_in = lon_in[::-1]
        data_3d = data_3d[:, :, ::-1]

    # 经度约定统一：ERA5 是 0-360，目标网格可能是 -180~180（或反之）
    lon_rho = np.asarray(lon_rho)
    if lon_in.max() > 180.0 and lon_rho.min() < 0.0:
        lon_rho = np.mod(lon_rho, 360.0)
    elif lon_in.max() <= 180.0 and lon_rho.min() >= 0.0 and lon_rho.max() > 180.0:
        lon_in = np.where(lon_in < 0.0, lon_in + 360.0, lon_in)
        order = np.argsort(lon_in)
        lon_in = lon_in[order]
        data_3d = data_3d[:, :, order]

    target = np.column_stack([lat_rho.ravel(), lon_rho.ravel()])

    ntime = data_3d.shape[0]
    Lp, Mp = lon_rho.shape
    result = np.zeros((ntime, Lp, Mp))
    for t in range(ntime):
        # build per time step: RegularGridInterpolator.values became a
        # read-only property in scipy >= 1.14, so swapping interp.values
        # in place no longer works
        interp = RegularGridInterpolator(
            (lat_in, lon_in), data_3d[t],
            method='linear', bounds_error=False, fill_value=np.nan)
        result[t] = interp(target).reshape(Lp, Mp)
    return result


class ERA5toROMS:
    """
    ERA5 -> ROMS 强迫场转换器

    Parameters
    ----------
    in_dir : str
        ERA5数据目录（自动搜索grib/nc文件）
    out_dir : str
        输出目录
    time_start : str, optional
        起始时间 'YYYY-MM-DD HH:MM:SS'，不指定则用数据第一个时刻
    time_end : str, optional
        结束时间 'YYYY-MM-DD HH:MM:SS'，不指定则用数据最后一个时刻
    base_date : str, optional
        ROMS时间基准 'YYYY-MM-DD HH:MM:SS'，不指定则用数据第一个时刻
    grid_file : str, optional
        ROMS网格文件，不指定则使用输入数据的经纬度范围
    variables : list, optional
        要处理的变量列表，默认全部
    """

    def __init__(self, in_dir, out_dir, time_start=None, time_end=None,
                 base_date=None, grid_file=None, variables=None,
                 wind_file='romsforc_era5_wind.nc',
                 forcing_file='romsforc_era5_forcing.nc',
                 rotate_wind=False):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.time_start = datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S') if time_start else None
        self.time_end = datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S') if time_end else None
        self.base_date = datetime.strptime(base_date, '%Y-%m-%d %H:%M:%S') if base_date else None
        self.grid_file = grid_file
        self.variables = variables or list(ROMS_VARINFO.keys())
        self.wind_file = wind_file
        self.forcing_file = forcing_file
        self.rotate_wind = rotate_wind

    def _find_files(self):
        """自动搜索输入目录下的grib和nc文件（不搜索子目录）"""
        files = []
        for ext in ['*.grib', '*.grib2', '*.grb', '*.grb2', '*.nc',
                     '*.GRIB', '*.GRIB2', '*.GRB', '*.GRB2', '*.NC']:
            files.extend(glob.glob(os.path.join(self.in_dir, ext)))
        # 去重，排除输出文件
        files = sorted(set(os.path.normpath(f) for f in files))

        return files

    def _setup_grid(self, lon_1d, lat_1d):
        """设置目标网格"""
        if self.grid_file:
            with nc.Dataset(self.grid_file) as ds:
                self.lon_rho = ds.variables['lon_rho'][:]
                self.lat_rho = ds.variables['lat_rho'][:]
                self.angle_rho = ds.variables['angle'][:] if 'angle' in ds.variables \
                    else np.zeros_like(self.lon_rho)
            print(f"ROMS网格: {self.lon_rho.shape[0]}x{self.lon_rho.shape[1]}")
        else:
            # 使用输入数据的经纬度范围，创建规则网格
            # 数据shape是(ntime, nlat, nlon)，所以网格也要是(nlat, nlon)
            self.lon_rho, self.lat_rho = np.meshgrid(lon_1d, lat_1d)  # shape: (nlat, nlon)
            self.angle_rho = np.zeros_like(self.lon_rho)
            print(f"规则网格: {len(lon_1d)}x{len(lat_1d)}, "
                  f"lon=[{lon_1d[0]:.2f},{lon_1d[-1]:.2f}], "
                  f"lat=[{lat_1d[0]:.2f},{lat_1d[-1]:.2f}]")

    def _qc(self, data, roms_name):
        """简单质量控制"""
        data = np.where(np.isnan(data), 0, data)
        if roms_name in ('Uwind', 'Vwind', 'Tair'):
            data[data < -100] = 0
        elif roms_name == 'Pair':
            data[data < 0] = 0
        elif roms_name == 'Qair':
            data = np.clip(data, 0, 100)
        elif roms_name == 'rain':
            data[data < 0] = 0
        return data

    @staticmethod
    def _resample_time(data, times, common):
        """把 (ntime, Lp, Mp) 场线性插值到公共时间轴（逐格点）。"""
        if len(times) == len(common) and np.allclose(times, common):
            return data
        if len(times) == 1:
            return np.repeat(data, len(common), axis=0)
        out = np.empty((len(common),) + data.shape[1:], dtype=data.dtype)
        for k in range(data.shape[1]):
            for m in range(data.shape[2]):
                out[:, k, m] = np.interp(common, times, data[:, k, m])
        return out

    def _write_nc(self, filename, variables_dict):
        """写NetCDF文件（netCDF4格式，兼容ROMS）"""
        Lp, Mp = self.lon_rho.shape  # Lp=纬度数, Mp=经度数
        import tempfile, shutil

        # 先写到临时ASCII路径（绕过中文路径问题）
        tmp_fd, tmpfile = tempfile.mkstemp(suffix='.nc')
        os.close(tmp_fd)
        ncid = nc.Dataset(tmpfile, 'w', format='NETCDF4')

        # 维度名参考ROMS标准：Jm=纬度, Im=经度
        ncid.createDimension('Jm', Lp)
        ncid.createDimension('Im', Mp)

        # 经纬度 (Jm, Im) = (纬度, 经度)
        lon_var = ncid.createVariable('lon', 'f8', ('Jm', 'Im'))
        lon_var.long_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var[:] = self.lon_rho

        lat_var = ncid.createVariable('lat', 'f8', ('Jm', 'Im'))
        lat_var.long_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var[:] = self.lat_rho

        time_units = f"days since {self.base_date.strftime('%Y-%m-%d %H:%M:%S')}"

        # 所有变量共用一个 time 维度：取各变量时间轴的并集，
        # 短时间序列变量线性插值到公共轴（此前短序列尾部补 0 是错误值）
        common_time = np.unique(np.concatenate(
            [np.asarray(v['roms_time']) for v in variables_dict.values()]))
        ncid.createDimension('time', len(common_time))

        # 每个变量写入
        for var in variables_dict.values():
            roms_time = np.asarray(var['roms_time'])
            time_var_name = var['time_var']  # ROMS标准名: wind_time, tair_time等

            # 时间变量（维度是 'time'，变量名是 wind_time/tair_time 等）
            if time_var_name not in ncid.variables:
                tid = ncid.createVariable(time_var_name, 'f8', ('time',))
                tid.units = time_units
                tid.long_name = 'atmospheric forcing time'
                tid.field = f'{time_var_name}, scalar, series'
                tid[:] = common_time

            # 数据变量（维度是 'time', 'Jm', 'Im'）
            data = var['data']  # (ntime, Lp, Mp)
            data = self._resample_time(data, roms_time, common_time)
            vid = ncid.createVariable(var['roms'], 'f4', ('time', 'Jm', 'Im'))
            vid.long_name = var['long_name']
            vid.units = var['units']
            vid.coordinates = 'lon lat time'
            vid.time = time_var_name
            vid.field = f"{var['roms']}, scalar, series"
            vid[:] = data

        ncid.close()

        # 删除已存在的文件，复制到目标路径
        if os.path.exists(filename):
            os.remove(filename)
        shutil.copy2(tmpfile, filename)
        os.remove(tmpfile)

    def process(self):
        """主处理函数"""
        print("=" * 50)
        print("ERA5 -> ROMS Forcing Converter")
        print("=" * 50)

        # 1. 搜索文件
        files = self._find_files()
        if not files:
            raise FileNotFoundError(f"在 {self.in_dir} 中未找到数据文件")
        print(f"找到 {len(files)} 个文件: {[os.path.basename(f) for f in files]}")

        # 2. 分离grib和nc文件
        grib_files = [f for f in files if os.path.splitext(f)[1].lower() in ('.grib', '.grb', '.grib2', '.grb2')]
        nc_files = [f for f in files if os.path.splitext(f)[1].lower() == '.nc']

        # 3. 读取数据
        all_data = {}
        if grib_files:
            print("\n读取GRIB文件...")
            all_data.update(_read_grib_files(grib_files, self.variables))
        if nc_files:
            print("\n读取NC文件...")
            all_data.update(_read_nc_files(nc_files, self.variables))

        if not all_data:
            raise ValueError("未读取到任何数据")

        # 4. 自动获取经纬度范围并设置网格
        first_var = list(all_data.keys())[0]
        lon_1d = np.sort(np.unique(all_data[first_var]['lon']))
        lat_1d = np.sort(np.unique(all_data[first_var]['lat']))
        self._setup_grid(lon_1d, lat_1d)

        # 5. 时间处理
        time_raw = all_data[first_var]['time']

        # 自动设置base_date：默认用数据第一个时刻
        if self.base_date is None:
            self.base_date = time_raw[0]
            print(f"base_date自动设为: {self.base_date}")

        # 自动设置时间范围：默认用数据全部时刻
        if self.time_start is None:
            self.time_start = time_raw[0]
        if self.time_end is None:
            self.time_end = time_raw[-1]

        time_mask = (time_raw >= self.time_start) & (time_raw <= self.time_end)
        time_dt = time_raw[time_mask]
        print(f"时间: {time_dt[0]} ~ {time_dt[-1]}, {len(time_dt)}步")

        # 6. 逐变量处理（注意：不同变量可能有不同时间步数）
        wind_data = {}
        forcing_data = {}
        # 每个变量独立的时间映射：记录哪个时间步被选中
        var_time_masks = {}

        for vn in self.variables:
            if vn not in all_data:
                print(f"  {vn}: 未找到，跳过")
                continue
            info = ROMS_VARINFO.get(vn)
            if info is None:
                continue

            print(f"\n处理 {vn} -> {info['roms']}")

            # 每个变量用自己的时间做过滤（解决不同时间分辨率问题）
            var_time = all_data[vn]['time']
            var_mask = (var_time >= self.time_start) & (var_time <= self.time_end)
            var_time_filtered = var_time[var_mask]
            var_roms_time = np.array([(t - self.base_date).total_seconds() / 86400.0
                                      for t in var_time_filtered])
            raw = all_data[vn]['data'][var_mask]

            # 单位转换：ERA5 累积量（J/m²、m）按数据实际时间步长换算为通量
            src_units = str(all_data[vn].get('units') or '')
            u_low = src_units.lower().strip()
            deltas = [(var_time_filtered[i + 1] - var_time_filtered[i]).total_seconds()
                      for i in range(len(var_time_filtered) - 1)]
            dt_hours = (float(np.median(deltas)) / 3600.0) if deltas else 1.0
            if (info['roms'] in ('swrad', 'lwrad', 'lwrad_down')
                    and 'j' in u_low and 'w' not in u_low):
                print(f"  累积辐射({src_units.strip()}) -> W m-2 (÷{dt_hours:g}h)")
                raw = raw / (dt_hours * 3600.0)
            elif info['roms'] == 'rain' and u_low in ('m', 'meters',
                    'm of water equivalent', 'm of water equivilent'):
                print(f"  累积降水({src_units.strip()}) -> kg m-2 s-1 (÷{dt_hours:g}h)")
                raw = raw / (dt_hours * 3600.0) * 1000.0
            elif 'convert' in info:
                raw = _convert(raw, info['convert'])

            # 判断是否需要插值（输入输出网格相同时跳过）
            in_lon = all_data[vn]['lon']
            in_lat = all_data[vn]['lat']
            # 网格shape是(nlat, nlon)，lon_rho[0,:]是lon一维，lon_rho[:,0]是lat一维
            same_grid = (len(in_lat) == self.lon_rho.shape[0] and
                         len(in_lon) == self.lon_rho.shape[1] and
                         np.allclose(in_lat, self.lat_rho[:, 0]) and
                         np.allclose(in_lon, self.lon_rho[0, :]))

            if same_grid:
                print(f"  网格相同，跳过插值")
                proc_data = raw
            else:
                print(f"  插值到目标网格...")
                proc_data = _interpolate(raw, in_lon, in_lat, self.lon_rho, self.lat_rho)

            # 风场需要特殊处理
            if vn in ('u10', 'v10'):
                # 暂存原始数据和时间
                wind_data[vn] = proc_data
                var_time_masks[vn] = var_roms_time
                if len([k for k in wind_data if k in ('u10', 'v10')]) == 2:
                    u_raw, v_raw = wind_data['u10'], wind_data['v10']
                    wind_time = var_time_masks.get('u10', var_roms_time)
                    if self.rotate_wind:
                        print("  旋转风场...")
                        a = self.angle_rho
                        u_rot = u_raw * np.cos(a) + v_raw * np.sin(a)
                        v_rot = v_raw * np.cos(a) - u_raw * np.sin(a)
                        u_raw, v_raw = u_rot, v_rot
                    else:
                        print("  不旋转（直接输出u10/v10分量）")
                    for vn2, data in [('Uwind', u_raw), ('Vwind', v_raw)]:
                        wind_data[vn2] = {'roms': vn2, 'data': self._qc(data, vn2).astype('f4'),
                                          'units': 'meter second-1',
                                          'long_name': f'surface {"u" if vn2=="Uwind" else "v"}-wind component',
                                          'roms_time': wind_time,
                                          'time_var': 'wind_time'}
                    wind_data.pop('u10', None)
                    wind_data.pop('v10', None)
                continue

            # 质量控制
            proc_data = self._qc(proc_data, info['roms'])
            info_out = {'roms': info['roms'], 'data': proc_data.astype('f4'),
                        'units': info['units'], 'long_name': info['long_name'],
                        'roms_time': var_roms_time,
                        'time_var': info['time_var']}

            if info['file'] == 'wind':
                wind_data[info['roms']] = info_out
            else:
                forcing_data[info['roms']] = info_out

        # 7. 写文件
        # u10/v10 必须成对（缺一时 wind_data 里会残留裸数组导致写文件崩溃）
        if 'u10' in wind_data or 'v10' in wind_data:
            missing = 'v10' if 'u10' in wind_data else 'u10'
            raise ValueError(f"风场 u10/v10 必须成对提供：缺少 {missing}")

        os.makedirs(self.out_dir, exist_ok=True)

        if wind_data:
            outf = os.path.join(self.out_dir, self.wind_file)
            self._write_nc(outf, wind_data)
            print(f"\n写入: {outf}")

        if forcing_data:
            outf = os.path.join(self.out_dir, self.forcing_file)
            self._write_nc(outf, forcing_data)
            print(f"写入: {outf}")

        print("\n完成!")
