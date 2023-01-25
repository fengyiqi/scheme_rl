import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
from .utils import normalize, get_scale_coefs


_zero_mean = True


# def _get_states(data_obj, layers=None, zero_mean=_zero_mean, ave_pool=None):
#     assert zero_mean, "Non-zeromean has not been implemented"
#     if layers is None:
#         layers = ["density", "velocity_x", "velocity_y", "pressure"]
#     state_matrix = []
#     for state in layers:
#         state_dist = data_obj.result[state]
#         if ave_pool is not None and state_dist.shape != (64, 64):
#             state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
#         if round(np.max(state_dist) - np.min(state_dist), 6) < 1e-6:
#             value = np.zeros_like(state_dist) if zero_mean else np.zeros_like(state_dist) + 0.5
#         else:
#             value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
#             value = value - 0.5 if zero_mean else value
#         state_matrix.append(value)
#     return np.array(state_matrix)

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

class FreeShearThinEnv(AlpacaEnv):

    def __init__(self):
        super(FreeShearThinEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_thin_64",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear_thin/shear_thin_64_weno5",
            high_res=(False, None),
            cpu_num=4,
            get_state_func=_get_states,
        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/shear_thin_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return self.compute_reward(end_time=end_time, coef_dict=self.scale_coef, scale=100)



class FreeShearThinHighRes128Env(AlpacaEnv):
    def __init__(self):
        super(FreeShearThinHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_thin_128",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear_thin/shear_thin_64_weno5",
            high_res=(True, 2),
            cpu_num=6,
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0

class FreeShearThinHighRes184Env(AlpacaEnv):
    def __init__(self):
        super(FreeShearThinHighRes184Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_thin_184",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear_thin/shear_thin_64_weno5",
            high_res=(True, 3),
            cpu_num=6,
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0

class FreeShearThinHighRes256Env(AlpacaEnv):

    def __init__(self):
        super(FreeShearThinHighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="shear_thin_256",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/shear_thin/shear_thin_64_weno5",
            high_res=(True, 4),
            cpu_num=6,
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0