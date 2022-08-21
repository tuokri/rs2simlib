import math

import numpy as np

from rs2simlib.fast import sim as fastsim


def test_fast_simulate():
    time_step = 1 / 500
    sim_time = 5.0

    sim_results = fastsim.simulate(
        sim_time=sim_time,
        time_step=time_step,
        drag_func=7,
        ballistic_coeff=0.24,
        aim_dir_x=1.0,
        aim_dir_y=0.0,
        muzzle_velocity=340.0 * 50,
        falloff_x=np.array([241491600.0, 1509322500.0]),
        falloff_y=np.array([0.85, 0.2]),
        bullet_damage=147,
        instant_damage=160,
        pre_fire_trace_len=25 * 50,
        start_loc_x=0.0,
        start_loc_y=0.0,
    )

    arr_len = math.ceil(sim_time / time_step) - 1
    assert np.size(sim_results) == arr_len * 6
    assert np.size(sim_results, axis=0) == 6
    assert np.size(sim_results, axis=1) == arr_len
