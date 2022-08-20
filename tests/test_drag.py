from rs2simlib.fast import drag as fast_drag


def test_fast_drag_g1():
    assert fast_drag.drag_g1(0.0) == 0.2629
    assert fast_drag.drag_g1(5.1) == 0.4988


def test_fast_drag_g7():
    assert fast_drag.drag_g7(0.0) == 0.1198
    assert fast_drag.drag_g7(5.1) == 0.1618
