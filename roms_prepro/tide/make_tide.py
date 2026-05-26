import numpy as np
import netCDF4 as nc
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


CONSTITUENTS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4']

DOODSON = {
    'M2':  np.array([2, -2,  2,  0, 0, 0,   0, 28.9841042]),
    'S2':  np.array([2,  0,  0,  0, 0, 0,   0, 30.0000000]),
    'N2':  np.array([2, -3,  2,  1, 0, 0,   0, 28.4397297]),
    'K2':  np.array([2,  0,  2,  0, 0, 0,   0, 30.0821381]),
    'K1':  np.array([1,  0,  1,  0, 0, 0,  90, 15.0410690]),
    'O1':  np.array([1, -2,  1,  0, 0, 0, 270, 13.9430351]),
    'P1':  np.array([1,  0, -1,  0, 0, 0, 270, 14.9589310]),
    'Q1':  np.array([1, -3,  1,  1, 0, 0, 270, 13.3986607]),
    'M4':  np.array([4, -4,  4,  0, 0, 0,   0, 57.9682083]),
}


def tpxo_to_roms_tide(roms_grid_file, tpxo_dir, out_file,
                     constituents=None, ini_date=None):
    if constituents is None:
        constituents = [c for c in CONSTITUENTS if c in DOODSON]

    for c in constituents:
        if c not in DOODSON:
            raise ValueError(f'Unknown constituent: {c}')

    g = nc.Dataset(roms_grid_file)
    lon_rho = np.mod(g.variables['lon_rho'][:], 360)
    lat_rho = g.variables['lat_rho'][:]
    mask_rho = g.variables['mask_rho'][:]
    h_roms = g.variables['h'][:]
    g.close()

    

    Lp, Mp = lon_rho.shape
    N = len(constituents)

    periods = np.array([360.0 / DOODSON[c][7] for c in constituents])

    lon_min, lon_max = lon_rho.min(), lon_rho.max()
    lat_min, lat_max = lat_rho.min(), lat_rho.max()

    zamp = np.zeros((Lp, Mp, N))
    zpha = np.zeros((Lp, Mp, N))
    uamp = np.zeros((Lp, Mp, N))
    upha = np.zeros((Lp, Mp, N))
    vamp = np.zeros((Lp, Mp, N))
    vpha = np.zeros((Lp, Mp, N))

    t0 = datetime.now() if ini_date is None else datetime.fromisoformat(ini_date.replace(' ', 'T'))
    t_days = (t0 - datetime(1900, 1, 1)).days + (t0.hour + t0.minute / 60.0) / 24.0
    t = (t_days - datetime(1900, 1, 1).toordinal()) / 36525.0

    fFac = _nodal_factors(t, constituents)
    uFac = _nodal_phase_correction(t, constituents)
    Vdeg = _equilibrium_phase(t0, constituents)

    pbar = tqdm(constituents, desc='Tidal constituents', unit='con')
    for ki, con in enumerate(pbar):

        hf_file = f'{tpxo_dir}/hf.{con.lower()}_tpxo8_atlas_30c.nc'
        uv_file = f'{tpxo_dir}/uv.{con.lower()}_tpxo8_atlas_30c.nc'

        h_complex = _read_hf_complex(hf_file, lon_min, lon_max, lat_min, lat_max)
        u_complex = _read_uv_complex(uv_file, lon_min, lon_max, lat_min, lat_max, var='u')
        v_complex = _read_uv_complex(uv_file, lon_min, lon_max, lat_min, lat_max, var='v')

        z_interp = _interp_tpxo(h_complex, lon_rho, lat_rho, mask_rho)
        u_interp = _interp_tpxo(u_complex, lon_rho, lat_rho, mask_rho)
        v_interp = _interp_tpxo(v_complex, lon_rho, lat_rho, mask_rho)

        h_safe = np.where(mask_rho == 0, np.nan, h_roms)
        u_interp = u_interp / h_safe
        v_interp = v_interp / h_safe
        u_interp = np.nan_to_num(u_interp, 0)
        v_interp = np.nan_to_num(v_interp, 0)

        f = fFac[ki]
        u = uFac[ki]
        v0 = Vdeg[ki]

        zamp[:, :, ki] = np.abs(z_interp) * f
        zpha[:, :, ki] = np.mod(-np.angle(z_interp) * 180 / np.pi - u - v0, 360)

        uamp[:, :, ki] = np.abs(u_interp) * f
        upha[:, :, ki] = np.mod(-np.angle(u_interp) * 180 / np.pi - u - v0, 360)

        vamp[:, :, ki] = np.abs(v_interp) * f
        vpha[:, :, ki] = np.mod(-np.angle(v_interp) * 180 / np.pi - u - v0, 360)

    major, ecc, inc, ph = _ap2ep(uamp, upha, vamp, vpha)
    minor = major * np.abs(ecc)

    print(f'Writing {out_file}...')
    _write_tide_nc(out_file, Lp, Mp, N, constituents, periods,
                   zamp, zpha, major, minor, inc, ph, uamp, upha, vamp, vpha,
                   title=f'ROMS tidal forcing from TPXO8 atlas')

    print(f'Tidal forcing written: {out_file}')


def _read_uv_complex(uv_file, lon_min, lon_max, lat_min, lat_max, var='u'):
    ds = nc.Dataset(uv_file)
    if var == 'u':
        lon_var = ds.variables['lon_u'][:]
        lat_var = ds.variables['lat_u'][:]
        Re = ds.variables['uRe'][:]
        Im = ds.variables['uIm'][:]
    else:
        lon_var = ds.variables['lon_v'][:]
        lat_var = ds.variables['lat_v'][:]
        Re = ds.variables['vRe'][:]
        Im = ds.variables['vIm'][:]
    ds.close()

    complex_data = (Re + 1j * Im) / 10000.0

    margin = 1.0
    i = np.where((lon_var >= lon_min - margin) & (lon_var <= lon_max + margin))[0]
    j = np.where((lat_var >= lat_min - margin) & (lat_var <= lat_max + margin))[0]
    if len(i) == 0 or len(j) == 0:
        raise ValueError(f'No TPXO {var} data for region')

    lon_sub = lon_var[i]
    lat_sub = lat_var[j]
    data_sub = complex_data[np.ix_(i, j)]

    lon_sub = np.sort(lon_sub)
    lat_sub = np.sort(lat_sub)
    si_lon = np.argsort(lon_var[i])
    si_lat = np.argsort(lat_var[j])
    data_sorted = np.take(np.take(data_sub, si_lon, axis=0), si_lat, axis=1)

    return (lon_sub, lat_sub, data_sorted)


def _read_hf_complex(hf_file, lon_min, lon_max, lat_min, lat_max):
    ds = nc.Dataset(hf_file)
    lon_z = ds.variables['lon_z'][:]
    lat_z = ds.variables['lat_z'][:]
    hRe = ds.variables['hRe'][:]
    hIm = ds.variables['hIm'][:]
    ds.close()

    h_complex = (hRe + 1j * hIm) / 1000.0

    margin = 1.0
    i = np.where((lon_z >= lon_min - margin) & (lon_z <= lon_max + margin))[0]
    j = np.where((lat_z >= lat_min - margin) & (lat_z <= lat_max + margin))[0]
    if len(i) == 0 or len(j) == 0:
        raise ValueError(f'No TPXO data覆盖 ROMS 区域: lon=[{lon_min:.1f},{lon_max:.1f}], lat=[{lat_min:.1f},{lat_max:.1f}]')

    lon_sub = lon_z[i]
    lat_sub = lat_z[j]
    h_sub = h_complex[np.ix_(i, j)]

    lon_sub = np.sort(lon_sub)
    lat_sub = np.sort(lat_sub)
    si_lon = np.argsort(lon_z[i])
    si_lat = np.argsort(lat_z[j])
    data_sorted = np.take(np.take(h_sub, si_lon, axis=0), si_lat, axis=1)

    return (lon_sub, lat_sub, data_sorted)


def _interp_tpxo(src, dst_lon, dst_lat, mask):
    lon_1d, lat_1d, data = src
    interp = RegularGridInterpolator(
        (lon_1d, lat_1d), data,
        method='linear', bounds_error=False, fill_value=0
    )
    points = np.column_stack([dst_lon.ravel(), dst_lat.ravel()])
    result = interp(points).reshape(dst_lon.shape)
    result = np.where(mask == 0, 0, result)
    return result


def _nodal_factors(t, constituents):
    VN = np.mod(360 * (0.719954 - 5.372617 * t + 0.000006 * t * t), 360)
    VN = VN * np.pi / 180
    cN = np.cos(VN)
    c2N = np.cos(2 * VN)
    c3N = np.cos(3 * VN)
    sN = np.sin(VN)
    s2N = np.sin(2 * VN)
    s3N = np.sin(3 * VN)

    f = {}
    f['M2'] = 1.0004 - 0.0373 * cN + 0.0002 * c2N
    f['S2'] = 1.0
    f['N2'] = f['M2']
    f['K2'] = 1.0241 + 0.2863 * cN + 0.0083 * c2N - 0.0015 * c3N
    f['K1'] = 1.0060 + 0.1150 * cN - 0.0088 * c2N + 0.0006 * c3N
    f['O1'] = 1.0089 + 0.1871 * cN - 0.0147 * c2N + 0.0014 * c3N
    f['P1'] = f['K1']
    f['Q1'] = f['O1']
    f['M4'] = f['M2'] ** 2

    return np.array([f[c] for c in constituents])


def _nodal_phase_correction(t, constituents):
    VN = np.mod(360 * (0.719954 - 5.372617 * t + 0.000006 * t * t), 360)
    VN = VN * np.pi / 180
    sN = np.sin(VN)
    s2N = np.sin(2 * VN)
    s3N = np.sin(3 * VN)

    u = {}
    u['M2'] = -0.0374 * sN
    u['S2'] = 0.0
    u['N2'] = u['M2']
    u['K2'] = -0.3096 * sN + 0.0119 * s2N - 0.0007 * s3N
    u['K1'] = -0.1546 * sN + 0.0119 * s2N - 0.0012 * s3N
    u['O1'] = 0.1885 * sN - 0.0234 * s2N + 0.0033 * s3N
    u['P1'] = u['K1']
    u['Q1'] = u['O1']
    u['M4'] = 2 * u['M2']

    return np.array([np.mod(u[c] * 180 / np.pi, 360) for c in constituents])


def _equilibrium_phase(dt, constituents):
    t = (dt - datetime(1900, 1, 1)).total_seconds() / 86400.0
    t = t / 36525.0
    hour = dt.hour + dt.minute / 60.0

    Vs = np.mod(360 * (0.751206 + 1336.855231 * t - 3.6e-6 * t * t), 360)
    Vh = np.mod(360 * (0.776935 + 100.002136 * t + 1.0e-6 * t * t), 360)
    Vp = np.mod(360 * (0.928693 + 11.302872 * t - 2.9e-5 * t * t), 360)
    VN = np.mod(360 * (0.719954 - 5.372617 * t + 6.0e-6 * t * t), 360)
    Vp1 = np.mod(360 * (0.781169 + 0.004775 * t + 1.0e-6 * t * t), 360)
    Vs = np.mod(Vs, 360)
    Vh = np.mod(Vh, 360)
    Vp = np.mod(Vp, 360)
    VN = np.mod(VN, 360)
    Vp1 = np.mod(Vp1, 360)

    result = []
    for con in constituents:
        DN = DOODSON[con]
        V = (hour * DN[7] + Vs * DN[1] + Vh * DN[2] +
             Vp * DN[3] + VN * DN[4] + Vp1 * DN[5] + DN[6])
        V = np.mod(V, 360)
        result.append(V)
    return np.array(result)


def _ap2ep(Au, PHIu, Av, PHIv):
    PHIu = PHIu * np.pi / 180
    PHIv = PHIv * np.pi / 180
    u = Au * np.exp(-1j * PHIu)
    v = Av * np.exp(-1j * PHIv)
    wp = (u + 1j * v) / 2
    wm = np.conj(u - 1j * v) / 2
    Wp = np.abs(wp)
    Wm = np.abs(wm)
    THETAp = np.angle(wp)
    THETAm = np.angle(wm)
    SEMA = Wp + Wm
    SEMI = Wp - Wm
    ECC = SEMI / (SEMA + 1e-30)
    PHA = np.mod((THETAm - THETAp) / 2 * 180 / np.pi, 360)
    INC = np.mod((THETAm + THETAp) / 2 * 180 / np.pi, 360)
    return SEMA, ECC, INC, PHA


def _write_tide_nc(fn, Lp, Mp, N, constituents, periods,
                  zamp, zpha, cmax, cmin, cangle, cphase,
                  uamp, upha, vamp, vpha, title=''):
    xi_sz = Mp
    eta_sz = Lp
    ds = nc.Dataset(fn, 'w', format='NETCDF3_64BIT')
    ds.title = title
    ds.created = datetime.now().isoformat()
    ds.components = ' '.join(constituents)

    xi_dim = ds.createDimension('xi_rho', xi_sz)
    eta_dim = ds.createDimension('eta_rho', eta_sz)
    t_dim = ds.createDimension('tide_period', N)

    def add_var(name, dims, data, long_name, units):
        v = ds.createVariable(name, 'f8', dims)
        v.long_name = long_name
        v.units = units
        v[:] = data
        return v

    add_var('tide_period', ('tide_period',), periods,
            'Tide angular period', 'hours')

    add_var('tide_Eamp', ('xi_rho', 'eta_rho', 'tide_period'), zamp.transpose(1, 0, 2),
            'Tide elevation amplitude', 'meters')
    add_var('tide_Ephase', ('xi_rho', 'eta_rho', 'tide_period'), zpha.transpose(1, 0, 2),
            'Tide elevation phase angle', 'degrees')

    add_var('tide_Cmax', ('xi_rho', 'eta_rho', 'tide_period'), cmax.transpose(1, 0, 2),
            'Tidal current ellipse semi-major axis', 'meter second-1')
    add_var('tide_Cmin', ('xi_rho', 'eta_rho', 'tide_period'), cmin.transpose(1, 0, 2),
            'Tidal current ellipse semi-minor axis', 'meter second-1')
    add_var('tide_Cangle', ('xi_rho', 'eta_rho', 'tide_period'), cangle.transpose(1, 0, 2),
            'Tidal current ellipse inclination angle', 'degrees between semi-major axis and east')
    add_var('tide_Cphase', ('xi_rho', 'eta_rho', 'tide_period'), cphase.transpose(1, 0, 2),
            'Tidal current ellipse phase angle', 'degrees')

    add_var('tide_Uamp', ('xi_rho', 'eta_rho', 'tide_period'), uamp.transpose(1, 0, 2),
            'Tidal current U-component amplitude', 'meters')
    add_var('tide_Uphase', ('xi_rho', 'eta_rho', 'tide_period'), upha.transpose(1, 0, 2),
            'Tidal current U-component phase', 'degrees')
    add_var('tide_Vamp', ('xi_rho', 'eta_rho', 'tide_period'), vamp.transpose(1, 0, 2),
            'Tidal current V-component amplitude', 'meters')
    add_var('tide_Vphase', ('xi_rho', 'eta_rho', 'tide_period'), vpha.transpose(1, 0, 2),
            'Tidal current V-component phase', 'degrees')

    ds.close()