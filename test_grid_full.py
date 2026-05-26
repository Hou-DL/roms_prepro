"""
Comprehensive tests for grid creation, bathymetry interpolation,
smoothing, and roughness metrics (rx0, rx1).
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from roms_prepro.grid import (
    make_grid_corner, compute_metrics, compute_mask,
    rx0, rx1, roughness_report, rfactor,
    interp_bathymetry, fill_bathymetry,
    smooth_bathymetry, shapiro_filter,
    set_depth, create_roms_grid
)


# ===========================================================================
# 1.  make_grid_corner
# ===========================================================================

def test_make_grid_corner_shape():
    lon, lat = make_grid_corner([0, 0, 10, 10], [0, 10, 10, 0], nx=50, ny=40)
    assert lon.shape == (40, 50)
    assert lat.shape == (40, 50)
    # Corners should match
    assert abs(lon[0, 0] - 0) < 1e-10
    assert abs(lat[0, 0] - 0) < 1e-10
    assert abs(lat[-1, -1] - 10) < 1e-10
    print("[PASS] make_grid_corner shape & corners")


# ===========================================================================
# 2.  compute_metrics — shapes and consistency
# ===========================================================================

def test_compute_metrics():
    lon, lat = make_grid_corner([100, 100, 110, 110], [20, 30, 30, 20],
                                 nx=10, ny=8)
    m = compute_metrics(lon, lat)

    assert m['lon_rho'].shape == (8, 10)
    assert m['lon_u'].shape == (8, 9)       # xi_u = xi_rho - 1
    assert m['lon_v'].shape == (7, 10)       # eta_v = eta_rho - 1
    assert m['lon_psi'].shape == (7, 9)
    assert m['pm'].shape == (8, 10)
    assert m['pn'].shape == (8, 10)
    assert m['f'].shape == (8, 10)
    assert m['spherical'] == 'T'

    # f should increase with latitude
    assert np.all(np.diff(m['f'], axis=0) > 0)
    # pm, pn positive
    assert np.all(m['pm'] > 0)
    assert np.all(m['pn'] > 0)

    print("[PASS] compute_metrics shapes & basic sanity")


# ===========================================================================
# 3.  compute_mask
# ===========================================================================

def test_compute_mask():
    mask = np.ones((5, 6), dtype=float)
    mask[2, 2:4] = 0  # land block at centre
    u, v, p = compute_mask(mask)

    assert u.shape == (5, 5)
    assert v.shape == (4, 6)
    assert p.shape == (4, 5)

    # Check that land propagates correctly
    assert u[2, 1] == 0  # adjacent to land at (2,2) and (2,3)
    assert u[2, 2] == 0
    assert v[2, 2] == 0
    assert v[2, 3] == 0
    assert p[2, 2] == 0

    print("[PASS] compute_mask")


# ===========================================================================
# 4.  rx0 and rx1
# ===========================================================================

def test_rx0_flat():
    h = np.ones((10, 10)) * 100.0
    r = rx0(h)
    assert np.allclose(r, 0.0)
    print("[PASS] rx0: flat bathymetry = 0")

def test_rx0_step():
    h = np.ones((10, 10)) * 100.0
    h[5:, :] = 1.0  # step from 100 to 1
    r = rx0(h)
    # The step gives: |100-1|/(100+1) = 99/101 ≈ 0.980
    max_r = r.max()
    assert abs(max_r - 99/101) < 1e-10, f"rx0 max step: {max_r}"
    print(f"[PASS] rx0: step = {max_r:.4f}")

def test_rx1_flat():
    h = np.ones((10, 10)) * 100.0
    mask = np.ones_like(h)
    from roms_prepro.grid import set_depth
    z_w = set_depth(2, 4, 5.0, 0.4, 20.0, 10, h, igrid=5)
    r = rx1(z_w)
    assert np.allclose(r, 0.0), f"rx1 flat max: {r.max()}"
    print("[PASS] rx1: flat = 0")

def test_rx1_stepped():
    h = np.ones((10, 10)) * 100.0
    h[5:, :] = 10.0
    mask = np.ones_like(h)
    z_w = set_depth(2, 4, 5.0, 0.4, 20.0, 10, h, igrid=5)
    r = rx1(z_w)
    # rx1 should be > 0 at the step boundary
    assert r.max() > 0.1
    print(f"[PASS] rx1: stepped = {r.max():.4f}")


# ===========================================================================
# 5.  Bathymetry interpolation (simple synthetic)
# ===========================================================================

def test_interp_bathymetry():
    # Create a synthetic source: lon [0..10], lat [0..10], depth = 100 + lat
    slon = np.linspace(-5, 15, 100)
    slat = np.linspace(-5, 15, 100)
    slon2, slat2 = np.meshgrid(slon, slat)
    sdep = 100.0 + slat2.T  # (lon, lat)

    # Target: simple rect grid
    lon_t, lat_t = make_grid_corner([1, 1, 9, 9], [1, 9, 9, 1], nx=20, ny=16)
    h = interp_bathymetry(lon_t, lat_t, slon, slat, sdep)

    # At mid-lat 5°, depth should be ~105
    mid = h[8, 10]
    assert abs(mid - 105) < 2.0, f"depth at mid: {mid}"
    # Monotonic in lat direction
    assert np.all(np.diff(h, axis=0) >= -1.0)

    print("[PASS] interp_bathymetry synthetic test")


def test_fill_bathymetry():
    h = np.array([[3.0, 8.0], [12.0, 0.0]])
    mask = np.array([[1, 1], [1, 0]])
    hf = fill_bathymetry(h, mask, min_depth=5.0)
    assert hf[0, 0] == 5.0         # below min
    assert hf[0, 1] == 8.0         # unchanged
    assert hf[1, 0] == 12.0        # unchanged
    assert hf[1, 1] == 2.5         # land -> min_depth/2
    print("[PASS] fill_bathymetry")


# ===========================================================================
# 6.  Shapiro filter
# ===========================================================================

def test_shapiro_filter():
    h = np.random.rand(20, 20) * 200
    hs = shapiro_filter(h, order=4, napp=3)
    # Filter should reduce the variance
    assert hs.std() < h.std() * 0.99  # rough check
    print(f"[PASS] shapiro_filter: std reduced {h.std():.1f} -> {hs.std():.1f}")


# ===========================================================================
# 7.  rx0-based smoothing
# ===========================================================================

def test_smooth_bathymetry_rx0():
    """Test rfactor matches expectations and smoothing is callable."""
    h = 50 + np.random.rand(10, 10) * 5
    h[5:, :] = 200

    mask = np.ones_like(h)
    r = rfactor(h, mask)
    assert r.shape == (9, 9)  # PSI-point dimensions
    assert r.max() > 0.1  # should have roughness at the step

    # Smoothing should complete without error
    h_s = smooth_bathymetry(h, mask, method='rx0', rx0max=0.3, max_iter=50)
    assert h_s.shape == h.shape
    assert np.all(h_s[mask > 0] >= 5)  # min depth preserved
    print(f"[PASS] rfactor + smooth_bathymetry: r_max={r.max():.4f}, "
          f"rfactor shape={r.shape}")


# ===========================================================================
# 8.  High-level create_roms_grid (full workflow)
# ===========================================================================

def test_create_roms_grid():
    with tempfile.TemporaryDirectory() as tmp:
        grid_file = os.path.join(tmp, 'test_grid.nc')

        # Create a synthetic source
        slon = np.linspace(-5, 120, 200)
        slat = np.linspace(-5, 50, 200)
        slon2, slat2 = np.meshgrid(slon, slat)
        sdep = 2000.0 + 1000 * np.sin(slat2.T * np.pi / 30)

        metrics, h, mask_rho = create_roms_grid(
            lon_corners=[110, 110, 120, 120],
            lat_corners=[20, 30, 30, 20],
            nx=20, ny=16,
            grid_file=grid_file,
            source_lon=slon, source_lat=slat, source_depth=sdep,
            min_depth=5.0,
            smooth_method='rx0', rx0max=0.2,
            coastline_smooth=False,
            vgrid_params={'Vtransform': 2, 'Vstretching': 4}
        )

        # Verify the file was written
        import netCDF4 as nc4
        ds = nc4.Dataset(grid_file)
        assert 'lon_rho' in ds.variables
        assert 'lat_rho' in ds.variables
        assert 'lon_u' in ds.variables
        assert 'lat_u' in ds.variables
        assert 'lon_v' in ds.variables
        assert 'lon_psi' in ds.variables
        assert 'h' in ds.variables
        assert 'pm' in ds.variables
        assert 'pn' in ds.variables
        assert 'angle' in ds.variables
        assert 'f' in ds.variables
        assert 'mask_rho' in ds.variables
        assert 'mask_u' in ds.variables
        assert 'mask_v' in ds.variables

        # Check bathymetry dimensions
        h_nc = ds.variables['h'][:]
        assert h_nc.shape == (16, 20)
        assert np.all(h_nc >= 5.0), f"min depth = {h_nc.min()}"

        # Check all-water mask (no coast in this example)
        assert np.all(ds.variables['mask_rho'][:] == 1.0)
        ds.close()

    print("[PASS] create_roms_grid full workflow")


# ===========================================================================
# 9.  Grid roughness validation on a more realistic domain
# ===========================================================================

def test_grid_roughness_with_vgrid():
    """
    Create a grid with variable bathymetry, compute 3-D depths,
    and check both rx0 and rx1 quality metrics.
    """
    # Domain: simple box with a shelf slope
    np.random.seed(42)
    nx, ny = 25, 20
    lon_c = [100, 100, 110, 110]
    lat_c = [10, 20, 20, 10]

    # Synthetic bathymetry: shallow shelf -> deep ocean
    lons, lats = make_grid_corner(lon_c, lat_c, nx, ny)
    dist_from_shore = np.arange(ny)[:, None] * np.ones(nx)  # eta direction
    h_raw = 20 + (dist_from_shore / ny) ** 2 * 1980  # 20..2000 m

    mask = np.ones_like(h_raw)

    # Smooth it
    h_s = smooth_bathymetry(h_raw, mask, method='rx0', rx0max=0.2, max_iter=200, order=4, npass=4)

    # Compute depth (use moderate theta_s for stability in variable bathymetry)
    z_w = set_depth(2, 4, 3.0, 0.1, 20.0, 20, h_s, igrid=5)

    # Compute roughness
    r0 = rx0(h_s)
    r1 = rx1(z_w)

    print(f"\nGrid roughness validation ({nx}x{ny}, shelf-slope domain):")
    print(f"  rx0: max={r0.max():.4f}, mean={r0.mean():.4f}, "
          f"median={np.median(r0):.4f}")
    print(f"  rx1: max={r1.max():.4f}, mean={r1.mean():.4f}, "
          f"median={np.median(r1):.4f}")

    # Recommended thresholds
    assert r0.max() < 0.3, f"rx0 too high: {r0.max()}"
    assert r1.max() < 0.3, f"rx1 too high: {r1.max()}"
    print("[PASS] Grid roughness within acceptable bounds")


# ===========================================================================
if __name__ == "__main__":
    test_make_grid_corner_shape()
    test_compute_metrics()
    test_compute_mask()
    test_rx0_flat()
    test_rx0_step()
    test_rx1_flat()
    test_rx1_stepped()
    test_interp_bathymetry()
    test_fill_bathymetry()
    test_shapiro_filter()
    test_smooth_bathymetry_rx0()
    test_create_roms_grid()
    test_grid_roughness_with_vgrid()
    print("\n=== All grid tests passed ===")