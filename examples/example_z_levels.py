"""
Example: Interpolate ROMS output to standard depth levels.

Converts ROMS sigma-coordinate fields to fixed z-levels for visualization
and analysis with standard oceanographic depth levels.

Usage:
    python example_z_levels.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roms_prepro.remapping import process_file, DEFAULT_STD_DEPTHS
import numpy as np

# ===================================================================
# Configuration
# ===================================================================

INPUT_FILE = 'ocean_avg_0003.nc'
OUTPUT_FILE = 'ocean_avg_0003_z.nc'

# Custom depth levels (optional, uses DEFAULT_STD_DEPTHS if None)
custom_depths = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]

# ===================================================================
# Run
# ===================================================================

if __name__ == '__main__':
    # Using default depths
    result = process_file(INPUT_FILE, OUTPUT_FILE)

    # Or with custom depths
    # result = process_file(INPUT_FILE, 'output_custom_z.nc',
    #                       std_depths=custom_depths)

    if result:
        print(f'\nSuccess! Output: {result}')
    else:
        print('\nFailed!')
