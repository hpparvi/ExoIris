import os
os.environ["OMP_NUM_THREADS"] =        "1"
os.environ["OPENBLAS_NUM_THREADS"] =   "1"
os.environ["MKL_NUM_THREADS"] =        "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] =    "1"

# I'm forcing the multiprocessing start method to 'fork' on Macs
# because 'spawn' seems to cause errors with the numba code.
from sys import platform
if 'darwin' in platform:
    import multiprocessing as mp
    mp.set_start_method('fork')

from numba import set_num_threads, config
config.THREADING_LAYER = 'safe'
set_num_threads(1)