import os, sys
sys.path.insert(0, os.path.dirname(__file__))

def make_bry_cmems(grid_file=None):
    """Create ROMS boundary conditions from CMEMS monthly files."""
    from roms_prepro.bc.d_obc_cmems import main as cmems_bry_main
    import roms_prepro.bc.d_obc_cmems as cfg

    if grid_file is None:
        grid_file = r"D:\testFile\windcompare\tropical_5km_grid_629.nc"

    # ---------- edit these parameters ----------
    cfg.CMEMS_FILE = r"E:\Ocean_data\mercator\CMEMS_TRO_2016.nc"
    cfg.GRD_NAME = grid_file
    cfg.BRY_NAME = r'D:\testFile\windcompare\bry_201609.nc'

    cfg.BOUNDARY = [0, 1, 1, 1]  # [W, E, S, N]

    cfg.N_LEVELS = 50
    cfg.VTRANSFORM = 2
    cfg.VSTRETCHING = 4
    cfg.THETA_S = 5.0
    cfg.THETA_B = 2.0
    cfg.TCLINE = 200.0

    cfg.TIME_START = '2016-09-01 00:00:00'
    cfg.TIME_END = '2016-09-09 23:00:00'
    cfg.ROMS_TIME_REF = '1990-01-01 00:00:00'
    # -------------------------------------------

    cmems_bry_main()
if __name__ == '__main__':
    make_bry_cmems()