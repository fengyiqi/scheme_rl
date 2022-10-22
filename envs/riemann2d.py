import numpy as np
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
import torch
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
        if np.max(state_dist) - np.min(state_dist) < 1e-6:
            value = np.zeros_like(state_dist) if zero_mean else np.zeros_like(state_dist) + 0.5
        else:
            value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
            value = value - 0.5 if zero_mean else value
        state_matrix.append(value)
    return np.array(state_matrix)


class RiemannConfig3Env(AlpacaEnv):

    def __init__(self):
        
        super(RiemannConfig3Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="config3_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/config3_64_weno5",
            high_res=(False, None),
            get_state_func=_get_states,
            cpu_num=4,
        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/config3_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return self.compute_reward(end_time=end_time, coef_dict=self.scale_coef)


class RiemannConfig3HighRes128Env(AlpacaEnv):

    def __init__(self):
        super(RiemannConfig3HighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="config3_128",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/config3_64_weno5",
            high_res=(True, 2),
            get_state_func=_get_states,
            cpu_num=6,
        )

    def get_reward(self, end_time):
        return 0


class RiemannConfig3HighRes256Env(AlpacaEnv):

    def __init__(self):
        super(RiemannConfig3HighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="config3_256",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/config3_64_weno5",
            high_res=(True, 4),
            get_state_func=_get_states,
            cpu_num=6,
        )

    def get_reward(self, end_time):
        return 0


class RiemannConfig3HighRes512Env(AlpacaEnv):

    def __init__(self):
        super(RiemannConfig3HighRes512Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="config3_512",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/config3_64_weno5",
            high_res=(True, 8),
            get_state_func=_get_states,
            cpu_num=6,
        )

    def get_reward(self, end_time):
        return 0
