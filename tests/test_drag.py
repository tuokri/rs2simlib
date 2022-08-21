import numpy as np

from rs2simlib import drag
from rs2simlib.fast import drag as fastdrag


def test_fast_drag_g1():
    assert fastdrag.drag_g1(np.float64(0.0)) == 0.2629
    assert fastdrag.drag_g1(np.float64(5.1)) == 0.4988


def test_fast_drag_g7():
    assert fastdrag.drag_g7(np.float64(0.0)) == 0.1198
    assert fastdrag.drag_g7(np.float64(5.1)) == 0.1618


def test_drag_g1():
    assert drag.drag_g1(0.0) == 0.2629
    assert drag.drag_g1(5.1) == 0.4988


def test_drag_g7():
    assert drag.drag_g7(0.0) == 0.1198
    assert drag.drag_g7(5.1) == 0.1618
