from rs2simlib.fast import sim as fastsim


def test_fast_simulate():
    sim_x, sim_y = fastsim.simulate(
        sim_time=5.0,
        time_step=1 / 500,
        drag_func=7,
        ballistic_coeff=0.25,
        aim_dir_x=1.0,
        aim_dir_y=0.0,
        muzzle_velocity=700.0 * 50,
        start_loc_x=0.0,
        start_loc_y=0.0,
    )
