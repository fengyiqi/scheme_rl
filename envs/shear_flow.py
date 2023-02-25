import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
from .utils import normalize, get_scale_coefs


_zero_mean = True


def _get_states(data_obj, layers=None, zero_mean=_zero_mean, ave_pool=None):
    if layers is None:
        layers = ["density", "velocity_x", "velocity_y", "pressure"]
    state_matrix = []
    for state in layers:
        state_dist = data_obj.result[state]
        if ave_pool is not None and state_dist.shape != (64, 64):
            state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
        
        state_matrix.append(state_dist)
    return np.array(state_matrix)

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 64, 64), dtype=np.float32)

class FreeShearEnv(AlpacaEnv):

    def __init__(self):
        super(FreeShearEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_64",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear/shear_64_weno5",
            high_res=(False, None),
            cpu_num=4,
            get_state_func=_get_states,
        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/shear_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return self.compute_reward(end_time=end_time, coef_dict=self.scale_coef, scale=100)



class FreeShearHighRes128Env(AlpacaEnv):
    def __init__(self):
        super(FreeShearHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_128",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear/shear_64_weno5",
            high_res=(True, 2),
            cpu_num=6,
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0


class FreeShearHighRes256Env(AlpacaEnv):

    def __init__(self):
        super(FreeShearHighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_256",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear/shear_64_weno5",
            high_res=(True, 4),
            cpu_num=6,
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0