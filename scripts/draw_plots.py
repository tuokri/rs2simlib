import matplotlib.pyplot as plt
import numpy as np

from rs2simlib.fast.sim import simulate

# noinspection DuplicatedCode
sim_params = {
    "time_step": np.float64(1 / 500),
    "sim_time": np.float64(5.0),
    "ballistic_coeff": np.float64(0.24),
    "aim_dir_x": np.float64(1.0),
    "aim_dir_y": np.float64(0.0),
    "muzzle_velocity": np.float64(340.0 * 50),
    "falloff_x": np.array([241491600.0, 1509322500.0]),
    "falloff_y": np.array([0.85, 0.2]),
    "bullet_damage": np.int64(147),
    "instant_damages": np.int64(160),
    "pre_fire_trace_len": np.int64(25 * 50),
    "start_loc_x": np.float64(0.0),
    "start_loc_y": np.float64(0.0),
    "drag_func": np.int64(7),
}


def main():
    res = simulate(**sim_params)

    # trajectory (x, y)
    plt.plot(res[0], res[1])
    plt.title("trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # distance, damage
    plt.figure()
    plt.plot(res[3], res[2])
    plt.axvline(sim_params["pre_fire_trace_len"] / 50, color="red")
    plt.xlabel("distance [m]")
    plt.ylabel("damage")
    plt.title("damage vs. distance")

    # distance, speed
    plt.figure()
    plt.plot(res[3], res[5])
    plt.xlabel("distance [m]")
    plt.ylabel(r"speed [$\frac{m}{s}$]")
    plt.title("speed vs. distance")

    # time_at_flight, speed
    plt.figure()
    plt.plot(res[4], res[5])
    plt.xlabel("flight time [s]")
    plt.ylabel(r"speed [$\frac{m}{s}$]")
    plt.title("speed vs. flight time")

    # time_at_flight, damage
    plt.figure()
    plt.plot(res[4], res[2])
    plt.xlabel("flight time [s]")
    plt.ylabel("damage")
    plt.title("damage vs. flight time")

    plt.show()


if __name__ == "__main__":
    main()
