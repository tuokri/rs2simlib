import math

import numpy as np
import pytest
from rs2simlib.fast import sim as fastsim

# noinspection DuplicatedCode
sim_params_1 = {
    "time_step": np.float64(1 / 500),
    "sim_time": np.float64(5.0),
    "ballistic_coeff": np.float64(0.24),
    "aim_dir_x": np.float64(1.0),
    "aim_dir_y": np.float64(0.0),
    "muzzle_velocity": np.float64(340.0 * 50),
    "falloff_x": np.array([241491600.0, 1509322500.0]),
    "falloff_y": np.array([0.85, 0.2]),
    "bullet_damage": np.int64(147),
    "instant_damage": np.int64(160),
    "pre_fire_trace_len": np.int64(25 * 50),
    "start_loc_x": np.float64(0.0),
    "start_loc_y": np.float64(0.0),
    "drag_func": np.int64(7),
}
# noinspection DuplicatedCode
sim_results_1 = {

}

# noinspection DuplicatedCode
sim_params_2 = {
    "time_step": np.float64(1 / 99),
    "sim_time": np.float64(6.426436),
    "ballistic_coeff": np.float64(0.111185),
    "aim_dir_x": np.float64(1.555),
    "aim_dir_y": np.float64(-2.32334234),
    "muzzle_velocity": np.float64(999.5599 * 50),
    "falloff_x": np.array([241491611.0, 1504422500.0]),
    "falloff_y": np.array([0.55, 0.32]),
    "bullet_damage": np.int64(85),
    "instant_damage": np.int64(101),
    "pre_fire_trace_len": np.int64(50 * 50),
    "start_loc_x": np.float64(-5.0),
    "start_loc_y": np.float64(9.6845),
    "drag_func": np.int64(1),
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
    results = fastsim.simulate(**sim_params)
    assert np.size(results) == arr_len * 6
    assert np.size(results, axis=0) == 6
    assert np.size(results, axis=1) == arr_len
