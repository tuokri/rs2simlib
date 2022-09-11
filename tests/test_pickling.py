import pickle

from rs2simlib.fast import drag as fastdrag
from rs2simlib.fast import sim as fastsim


def test_pickle_jitted_functions():
    p_str_g1 = pickle.dumps(fastdrag.drag_g1)
    p_str_g7 = pickle.dumps(fastdrag.drag_g7)
    p_str_sim = pickle.dumps(fastsim.simulate)

    obj_g1 = pickle.loads(p_str_g1)
    obj_g7 = pickle.loads(p_str_g7)
    obj_sim = pickle.loads(p_str_sim)

    assert obj_g1
    assert obj_g7
    assert obj_sim
