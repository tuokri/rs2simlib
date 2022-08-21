import math

import numpy as np
import pytest

from rs2simlib.fast import sim as fastsim

# noinspection DuplicatedCode
sim_params_1 = {
    "time_step": np.float64(1 / 500),
    "sim_time": np.float64(5.0),
    "bc": np.float64(0.24),
    "aim_dir_x": np.float64(1.0),
    "aim_dir_y": np.float64(0.0),
    "muzzle_vel": np.float64(340.0 * 50),
    "falloff_x": np.array([241491600.0, 1509322500.0]),
    "falloff_y": np.array([0.85, 0.2]),
}
# noinspection DuplicatedCode
sim_results_1 = {

}

# noinspection DuplicatedCode
sim_params_2 = {
    "time_step": np.float64(1 / 99),
    "sim_time": np.float64(6.426436),
    "bc": np.float64(0.111185),
    "aim_dir_x": np.float64(1.555),
    "aim_dir_y": np.float64(-2.32334234),
    "muzzle_vel": np.float64(999.5599 * 50),
    "falloff_x": np.array([241491600.0, 1509322500.0]),
    "falloff_y": np.array([0.85, 0.2]),
}
# noinspection DuplicatedCode
sim_results_2 = {

}

sim_test_data = [
    (
        sim_params_1,
        sim_results_1,
        math.ceil(sim_params_1["sim_time"] / sim_params_1["time_step"]) - 1
    ),
    (
        sim_params_2,
        sim_results_2,
        math.ceil(sim_params_2["sim_time"] / sim_params_2["time_step"]) - 1
    ),
]


@pytest.mark.parametrize(
    "sim_params,sim_results,arr_len", sim_test_data)
def test_fast_simulate(sim_params, sim_results, arr_len):
    time_step = sim_params["time_step"]
    sim_time = sim_params["sim_time"]
    bc = sim_params["bc"]
    aim_dir_x = sim_params["aim_dir_x"]
    aim_dir_y = sim_params["aim_dir_y"]
    muzzle_vel = sim_params["muzzle_vel"]
    falloff_x = sim_params["falloff_x"]
    falloff_y = sim_params["falloff_y"]

    results = fastsim.simulate(
        sim_time=sim_time,
        time_step=time_step,
        drag_func=7,
        ballistic_coeff=bc,
        aim_dir_x=aim_dir_x,
        aim_dir_y=aim_dir_y,
        muzzle_velocity=muzzle_vel,
        falloff_x=falloff_x,
        falloff_y=falloff_y,
        bullet_damage=147,
        instant_damage=160,
        pre_fire_trace_len=25 * 50,
        start_loc_x=0.0,
        start_loc_y=0.0,
    )

    assert np.size(results) == arr_len * 6
    assert np.size(results, axis=0) == 6
    assert np.size(results, axis=1) == arr_len
