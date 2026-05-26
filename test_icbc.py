"""Quick test for icbc module - synthetic data."""
import sys, os, tempfile
sys.path.insert(0, r'D:\ROMS\roms_prepro')
import numpy as np
import netCDF4 as nc4
from roms_prepro.grid import create_roms_grid
from roms_prepro.icbc._core import (horizontal_interp, z_to_sigma, sigma_to_z,
                                     rotate_uv, uv_to_cgrid)

# Create a small grid to test with
print("=== Testing _core functions ===\n")

# 1. horizontal_interp
src_lon, src_lat = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 15))
dst_lon, dst_lat = np.meshgrid(np.linspace(1, 9, 10), np.linspace(2, 8, 8))
src_2d = np.sin(src_lon * 0.5) * np.cos(src_lat * 0.5)
result_2d = horizontal_interp(src_lon, src_lat, src_2d, dst_lon, dst_lat)
assert result_2d.shape == (8, 10), f"shape {result_2d.shape}"

# 3D
src_3d = np.array([src_2d * 1.0, src_2d * 2.0, src_2d * 3.0])
result_3d = horizontal_interp(src_lon, src_lat, src_3d, dst_lon, dst_lat)
assert result_3d.shape == (3, 8, 10)
print("[PASS] horizontal_interp 2D and 3D")

# 2. z_to_sigma
z_lev = np.array([-100, -80, -60, -40, -20, -5, -1, 0])
nz, ny, nx = 8, 4, 5
var_z = np.random.randn(nz, ny, nx)
s_dep = np.linspace(-90, -2, 20).reshape(20, 1, 1) * np.ones((1, ny, nx))
var_s = z_to_sigma(var_z, z_lev, s_dep)
assert var_s.shape == (20, ny, nx)
print("[PASS] z_to_sigma")

# 3. sigma_to_z
var_z_back = sigma_to_z(var_s, s_dep, z_lev)
assert var_z_back.shape == (nz, ny, nx)
print("[PASS] sigma_to_z")

# 4. rotate_uv
u, v = np.ones((4, 5)), np.zeros((4, 5))
angle = np.full((4, 5), np.pi / 2)
u_rot, v_rot = rotate_uv(u, v, 0.0, angle)
assert np.allclose(u_rot, 0, atol=1e-10)
assert np.allclose(v_rot, -1, atol=1e-10)
print("[PASS] rotate_uv")

# 5. uv_to_cgrid
u_rho_3d = np.ones((10, 4, 5))
v_rho_3d = np.zeros((10, 4, 5))
mask_u = np.ones((4, 4)); mask_v = np.ones((3, 5))
u_cg, v_cg = uv_to_cgrid(u_rho_3d, v_rho_3d, mask_u, mask_v)
assert u_cg.shape == (10, 4, 4)
assert v_cg.shape == (10, 3, 5)
print("[PASS] uv_to_cgrid")

print("\n=== Testing mercator_to_roms_ini ===\n")

# Create a synthetic source file
src_file = os.path.join(tempfile.gettempdir(), '_test_source.nc')
ds = nc4.Dataset(src_file, 'w')
ds.createDimension('lon', 20); ds.createDimension('lat', 15)
ds.createDimension('depth', 5)
lons = np.linspace(119, 126, 20); lats = np.linspace(33, 38, 15)
ds.createVariable('longitude', 'f8', ('lon',))[:] = lons
ds.createVariable('latitude', 'f8', ('lat',))[:] = lats
ds.createVariable('depth', 'f8', ('depth',))[:] = np.array([0, -10, -30, -80, -200])
ds.createVariable('zos', 'f8', ('lat', 'lon'))[:] = np.random.randn(15, 20) * 0.1
ds.createVariable('thetao', 'f8', ('depth', 'lat', 'lon'))[:] = 15 + np.random.randn(5, 15, 20) * 2
ds.createVariable('so', 'f8', ('depth', 'lat', 'lon'))[:] = 35 + np.random.randn(5, 15, 20) * 0.5
ds.createVariable('uo', 'f8', ('depth', 'lat', 'lon'))[:] = np.random.randn(5, 15, 20) * 0.1
ds.createVariable('vo', 'f8', ('depth', 'lat', 'lon'))[:] = np.random.randn(5, 15, 20) * 0.1
ds.close()

# Create a ROMS grid
grid_file = os.path.join(tempfile.gettempdir(), '_test_icbc_grid.nc')
create_roms_grid(
    lon_corners=[120, 120, 125, 125],
    lat_corners=[34, 37, 37, 34],
    nx=30, ny=20,
    grid_file=grid_file,
    source_lon=lons, source_lat=lats,
    source_depth=-(15 + np.random.randn(15, 20) * 2),
    min_depth=10.0,
    smooth_method='shapiro',
    coastline_smooth=False
)

# Run mercator_to_roms_ini
ini_file = os.path.join(tempfile.gettempdir(), '_test_ini.nc')
from roms_prepro.icbc import mercator_to_roms_ini
result = mercator_to_roms_ini(
    roms_grid_file=grid_file,
    source_file=src_file,
    ini_file=ini_file,
    N=20
)
print(f"[PASS] mercator_to_roms_ini → {ini_file}")
print(f"  zeta shape: {result['zeta'].shape}")
print(f"  temp shape: {result['temp'].shape}")
print(f"  u shape:    {result['u'].shape}")

# Verify the output file
ds = nc4.Dataset(ini_file)
for v in ['zeta', 'temp', 'salt', 'u', 'v', 'ubar', 'vbar']:
    assert v in ds.variables, f"missing {v}"
ds.close()
print("[PASS] Output file has all required variables")

# -------------------------------------------------------------------
# Test boundary conditions
# -------------------------------------------------------------------
print("\n=== Testing mercator_to_roms_bry ===\n")

# Create two time-stamped source files
src_files = []
for t in [0, 1]:
    tf = os.path.join(tempfile.gettempdir(), f'_test_source_t{t}.nc')
    ds = nc4.Dataset(tf, 'w')
    ds.createDimension('lon', 20); ds.createDimension('lat', 15)
    ds.createDimension('depth', 5)
    ds.createVariable('longitude', 'f8', ('lon',))[:] = lons
    ds.createVariable('latitude', 'f8', ('lat',))[:] = lats
    ds.createVariable('depth', 'f8', ('depth',))[:] = np.array([0, -10, -30, -80, -200])
    ds.createVariable('time', 'f8', ())[:] = float(t * 86400)
    ds.createVariable('zos', 'f8', ('lat', 'lon'))[:] = np.random.randn(15, 20) * (0.1 + t * 0.05)
    ds.createVariable('thetao', 'f8', ('depth', 'lat', 'lon'))[:] = 15 + t + np.random.randn(5, 15, 20)
    ds.createVariable('so', 'f8', ('depth', 'lat', 'lon'))[:] = 35 + np.random.randn(5, 15, 20) * 0.5
    ds.createVariable('uo', 'f8', ('depth', 'lat', 'lon'))[:] = np.random.randn(5, 15, 20) * 0.1
    ds.createVariable('vo', 'f8', ('depth', 'lat', 'lon'))[:] = np.random.randn(5, 15, 20) * 0.1
    ds.close()
    src_files.append(tf)

bry_file = os.path.join(tempfile.gettempdir(), '_test_bry.nc')
from roms_prepro.icbc import mercator_to_roms_bry
mercator_to_roms_bry(
    roms_grid_file=grid_file,
    source_files=src_files,
    bry_file=bry_file,
    boundaries=(True, True, True, True),
    N=20,
    Vstretching=4,
)

# Verify BC file
ds = nc4.Dataset(bry_file)
print("\nBC file dimensions:", {k: len(v) for k, v in ds.dimensions.items()})
print("BC file variables:", sorted(ds.variables.keys()))
assert 'bry_time' in ds.variables
assert len(ds.variables['bry_time'][:]) == 2
for v in ['zeta', 'temp', 'salt', 'u', 'v', 'ubar', 'vbar']:
    for edge in ['west', 'east', 'south', 'north']:
        vname = f'{v}_{edge}'
        assert vname in ds.variables, f"missing {vname}"
print("[PASS] BC file has all required edge variables")
ds.close()

print("\n=== All icbc tests passed ===")