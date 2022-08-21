import math

import numba as nb
import numpy as np
import numpy.typing as npt

from rs2simlib.fast.drag import drag_g1
from rs2simlib.fast.drag import drag_g7

SCALE_FACTOR_INVERSE = nb.float64(0.065618)
SCALE_FACTOR = nb.float64(15.24)


@nb.njit
def clamp(n: nb.float64,
          smallest: nb.float64,
          largest: nb.float64) -> nb.float64:
    return max(smallest, min(n, largest))


@nb.njit
def calc_damage(
        velocity: npt.NDArray[np.float64],
        muzzle_velocity: nb.float64,
        damage: nb.int32,
        falloff_x: npt.NDArray,
        falloff_y: npt.NDArray,
) -> nb.float64:
    v_size_sq = np.linalg.norm(velocity) ** 2
    power_left = v_size_sq / (muzzle_velocity ** 2)
    damage *= power_left
    energy_transfer = np.interp(
        x=v_size_sq,
        xp=falloff_x,
        fp=falloff_y,
    )
    energy_transfer = clamp(
        energy_transfer,
        smallest=falloff_y[0],
        largest=falloff_y[-1],
    )
    return damage * energy_transfer


@nb.njit
def simulate(
        sim_time: nb.float64,
        time_step: nb.float64,
        drag_func: nb.int32,
        ballistic_coeff: nb.float64,
        aim_dir_x: nb.float64,
        aim_dir_y: nb.float64,
        muzzle_velocity: nb.float64,
        falloff_x: npt.NDArray,
        falloff_y: npt.NDArray,
        bullet_damage: nb.int32,
        instant_damage: nb.int32,
        pre_fire_trace_len: nb.int32,
        start_loc_x: nb.float64 = 0.0,
        start_loc_y: nb.float64 = 0.0,
) -> npt.NDArray[npt.NDArray[np.float64]]:
    d: nb.float64
    v_size: nb.float64
    mach: nb.float64
    v: nb.float64
    cd: nb.float64
    damage: nb.float64
    flight_time = nb.float64(0.0)
    location = np.array([start_loc_x, start_loc_y], dtype=np.float64)
    prev_loc = location.copy()
    bc_inverse = 1.0 / ballistic_coeff
    velocity = np.array([aim_dir_x, aim_dir_y], np.float64)
    velocity /= np.linalg.norm(velocity)
    velocity *= muzzle_velocity

    drag_func = drag_g7 if (drag_func == 7) else drag_g1
    num_steps = math.ceil(sim_time / time_step)
    arr_len = num_steps - 1

    trajectory_x = np.empty(arr_len, dtype=np.float64)
    trajectory_y = np.empty(arr_len, dtype=np.float64)
    damage = np.empty(arr_len, dtype=np.float64)
    distance = np.empty(arr_len, dtype=np.float64)
    time_at_flight = np.empty(arr_len, dtype=np.float64)
    bullet_velocity = np.empty(arr_len, dtype=np.float64)

    i = nb.int32(0)
    while (flight_time < sim_time) and (i <= arr_len):
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
        prev_loc[0] = location[0]
        prev_loc[1] = location[1]
        location += loc_change
        d = abs(np.linalg.norm(prev_loc - location))
        distance[i] += d
        if d <= pre_fire_trace_len:
            damage[i] = instant_damage
        else:
            damage[i] = calc_damage(
                velocity,
                muzzle_velocity,
                bullet_damage,
                falloff_x,
                falloff_y)
        trajectory_x[i] = location[0]
        trajectory_y[i] = location[1]
        time_at_flight[i] = flight_time
        bullet_velocity[i] = np.linalg.norm(velocity)
        i += 1

    trajectory_x /= 50
    trajectory_y /= 50
    velocity /= 50
    distance /= 50

    ret = np.empty(shape=(6, arr_len), dtype=np.float64)
    ret[0] = trajectory_x
    ret[1] = trajectory_x
    ret[2] = damage
    ret[3] = distance
    ret[4] = time_at_flight
    ret[5] = bullet_velocity
    return ret
