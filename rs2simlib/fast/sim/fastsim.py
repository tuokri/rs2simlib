from typing import Tuple

import numba as nb
import numpy as np

from rs2simlib.fast.drag import drag_g1
from rs2simlib.fast.drag import drag_g7

SCALE_FACTOR_INVERSE: nb.float32 = 0.065618
SCALE_FACTOR: nb.float32 = 15.24


@nb.njit
def simulate(
        sim_time: nb.float32,
        time_step: nb.float32,
        drag_func: nb.int32,
        ballistic_coeff: nb.float32,
        aim_dir_x: nb.float32,
        aim_dir_y: nb.float32,
        muzzle_velocity: nb.float32,
        start_loc_x: nb.float32 = 0.0,
        start_loc_y: nb.float32 = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    v_size: nb.float32
    mach: nb.float32
    v: nb.float32
    cd: nb.float32
    flight_time: nb.float32 = 0.0
    distance_traveled_uu: nb.float32 = 0.0
    location = np.array([start_loc_x, start_loc_y])
    bc_inverse = 1.0 / ballistic_coeff
    velocity = np.array([aim_dir_x, aim_dir_y])
    velocity /= np.linalg.norm(velocity)
    velocity *= muzzle_velocity
    drag_func = drag_g7 if (drag_func == 7) else drag_g1

    while flight_time < sim_time:
        flight_time += time_step
        v_size = np.linalg.norm(velocity)
        v = v_size * SCALE_FACTOR_INVERSE
        mach = v * 0.0008958245617
        cd = drag_func(mach)
        velocity = (
                0.00020874137882624
                * (cd * bc_inverse) * np.square(v)
                * SCALE_FACTOR
                * (-1 * (velocity / v_size * time_step)))
        velocity[1] -= (490.3325 * time_step)
        loc_change = velocity * time_step
        prev_loc = location.copy()
        location += loc_change
        distance_traveled_uu += abs(
            np.linalg.norm(prev_loc - location))

    return np.zeros(0), np.zeros(0)
