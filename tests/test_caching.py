import os
import shutil
from pathlib import Path

from rs2simlib.fast.sim import trigger_jit

NUMBA_CACHE_DIR = "pytest_numba_cache"


# TODO: is this test useless?
def test_caching():
    Path(NUMBA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR
    os.environ["NUMBA_DEBUG_CACHE"] = "1"

    assert trigger_jit()
    shutil.rmtree(NUMBA_CACHE_DIR)
    assert trigger_jit()
