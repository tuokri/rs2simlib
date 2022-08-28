import math

import numba as nb
import numpy as np
import numpy.typing as npt
from numba.core.types import FunctionType

from rs2simlib.fast.drag import drag_g1
from rs2simlib.fast.drag import drag_g7

_DragFuncType = FunctionType(nb.float64(nb.float64))

SCALE_FACTOR_INVERSE = np.float64(0.065618)
SCALE_FACTOR = np.float64(15.24)
X1 = np.float64(0.0008958245617)
X2 = np.float64(0.00020874137882624)
GRAVITY = np.float64(490.3325)  # 490.3325 UU/s = 9.80665 m/s


@nb.njit(cache=True)
def clamp(n: np.float64,
          smallest: np.float64,
          largest: np.float64) -> np.float64:
    return np.float64(
        max(smallest, min(n, largest)))  # type: ignore[call-overload]


@nb.njit(cache=True)
def calc_damage(
        velocity: npt.NDArray[np.float64],
        muzzle_velocity: np.float64,
        damage: np.int64,
        falloff_x: npt.NDArray[np.float64],
        falloff_y: npt.NDArray[np.float64],
) -> np.float64:
    v_size_sq = np.linalg.norm(velocity) ** 2
    power_left = v_size_sq / (muzzle_velocity ** 2)
    d = damage * power_left
    energy_transfer: np.float64 = np.interp(
        x=v_size_sq,
        xp=falloff_x,
        fp=falloff_y,
    )  # type: ignore[assignment]
    energy_transfer = clamp(
        energy_transfer,
        smallest=falloff_y[0],
        largest=falloff_y[-1],
    )
    return d * energy_transfer


# @nb.njit(cache=True)
# def simulate(
#         sim_time: np.float64,
#         time_step: np.float64,
#         drag_func: np.int64,
#         ballistic_coeff: np.float64,
#         aim_dir_x: np.float64,
#         aim_dir_y: np.float64,
#         muzzle_velocity: np.float64,
#         falloff_x: nb.typed.List,
#         falloff_y: nb.typed.List,
#         bullet_damage: np.int64,
#         instant_damage: np.int64,
#         pre_fire_trace_len: np.int64,
#         start_loc_x=np.float64(0.0),
#         start_loc_y=np.float64(0.0),
# ) -> npt.ArrayLike:
#     # Workaround for numba caching failing when
#     # dynamically choosing the drag function.
#     if drag_func == 1:
#         return _simulate_g1(
#             sim_time=sim_time,
#             time_step=time_step,
#             ballistic_coeff=ballistic_coeff,
#             aim_dir_x=aim_dir_x,
#             aim_dir_y=aim_dir_y,
#             muzzle_velocity=muzzle_velocity,
#             falloff_x=falloff_x,
#             falloff_y=falloff_y,
#             bullet_damage=bullet_damage,
#             instant_damage=instant_damage,
#             pre_fire_trace_len=pre_fire_trace_len,
#             start_loc_x=start_loc_x,
#             start_loc_y=start_loc_y,
#         )
#     elif drag_func == 7:
#         return _simulate_g7(
#             sim_time=sim_time,
#             time_step=time_step,
#             ballistic_coeff=ballistic_coeff,
#             aim_dir_x=aim_dir_x,
#             aim_dir_y=aim_dir_y,
#             muzzle_velocity=muzzle_velocity,
#             falloff_x=falloff_x,
#             falloff_y=falloff_y,
#             bullet_damage=bullet_damage,
#             instant_damage=instant_damage,
#             pre_fire_trace_len=pre_fire_trace_len,
#             start_loc_x=start_loc_x,
#             start_loc_y=start_loc_y,
#         )
#     else:
#         raise ValueError("invalid drag function")


@nb.njit(cache=True)
def simulate(
        sim_time: np.float64,
        time_step: np.float64,
        drag_func: np.int64,
        ballistic_coeff: np.float64,
        aim_dir_x: np.float64,
        aim_dir_y: np.float64,
        muzzle_velocity: np.float64,
        falloff_x: nb.typed.List,
        falloff_y: nb.typed.List,
        bullet_damage: np.int64,
        instant_damage: np.int64,
        pre_fire_trace_len: np.int64,
        start_loc_x=np.float64(0.0),
        start_loc_y=np.float64(0.0),
) -> npt.ArrayLike:
    d = np.float64(0.0)
    flight_time = np.float64(0.0)
    location = np.array([start_loc_x, start_loc_y], dtype=np.float64)
    bc_inverse = 1.0 / ballistic_coeff
    velocity = np.array([aim_dir_x, aim_dir_y], np.float64)
    velocity /= np.linalg.norm(velocity)
    velocity *= muzzle_velocity

    num_steps = math.ceil(sim_time / time_step)
    arr_len = num_steps - 1

    trajectory_x = np.empty(arr_len, dtype=np.float64)
    trajectory_y = np.empty(arr_len, dtype=np.float64)
    damage = np.empty(arr_len, dtype=np.float64)
    distance = np.empty(arr_len, dtype=np.float64)
    time_at_flight = np.empty(arr_len, dtype=np.float64)
    bullet_velocity = np.empty(arr_len, dtype=np.float64)

    i = np.int64(0)
    while (flight_time < sim_time) and (i <= arr_len):
        flight_time += time_step
        # noinspection PyTypeChecker
        v_size: np.float64 = np.linalg.norm(velocity)
        # noinspection PyTypeChecker
        v: np.float64 = v_size * SCALE_FACTOR_INVERSE
        mach = v * X1

        # Numba caching fails when trying to choose
        # the drag function dynamically outside the loop.
        if drag_func == 1:
            # noinspection PyTypeChecker
            cd = drag_g1(mach)
        elif drag_func == 7:
            # noinspection PyTypeChecker
            cd = drag_g7(mach)
        else:
            raise ValueError("invalid drag function")

        velocity += (
                X2 * (cd * bc_inverse) * np.square(v)
                * SCALE_FACTOR
                * (-1 * ((velocity / v_size) * time_step)))
        velocity[1] -= (GRAVITY * time_step)
        loc_change = velocity * time_step
        prev_loc = location.copy()
        location += loc_change
        d += np.float64(abs(np.linalg.norm(prev_loc - location)))
        distance[i] = d
        if d <= pre_fire_trace_len:
            damage[i] = instant_damage
        else:
            damage[i] = calc_damage(
                velocity,
                muzzle_velocity,
                bullet_damage,
                falloff_x,
                falloff_y)
        trajectory_x[i] = np.float64(location[0])
        trajectory_y[i] = np.float64(location[1])
        time_at_flight[i] = np.float64(flight_time)
        bullet_velocity[i] = np.linalg.norm(velocity)
        i += 1

    # UU to m.
    trajectory_x /= 50
    trajectory_y /= 50
    velocity /= 50
    distance /= 50
    bullet_velocity /= 50

    ret = np.empty(shape=(6, arr_len), dtype=np.float64)
    ret[0] = trajectory_x
    ret[1] = trajectory_y
    ret[2] = damage
    ret[3] = distance
    ret[4] = time_at_flight
    ret[5] = bullet_velocity
    return ret


def trigger_jit():
    """Convenience function to trigger JIT compilation
    for all simulation functions.
    """
    drag_g1(np.float64(0.1))
    drag_g7(np.float64(0.1))

    simulate(
        sim_time=np.float64(0.21),
        time_step=np.float64(0.1),
        drag_func=7,
        ballistic_coeff=np.float64(0.15),
        aim_dir_x=np.float64(0.0),
        aim_dir_y=np.float64(0.0),
        muzzle_velocity=np.float64(15000.0),
        falloff_x=np.array([1.0, 1.0]),
        falloff_y=np.array([0.1, 0.1]),
        bullet_damage=np.int64(100),
        instant_damage=np.int64(101),
        pre_fire_trace_len=np.int64(1),
        start_loc_x=np.float64(0.0),
        start_loc_y=np.float64(0.0),
    )
