import numpy as np
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
import torch
from .utils import normalize, get_scale_coefs

_zero_mean = True


# def _get_states(data_obj, subdomain, layers=None, zero_mean=_zero_mean, ave_pool=None):
#     if layers is None:
#         layers = ["density", "velocity_x", "velocity_y", "pressure"]
#     state_matrix = []
#     for state in layers:
#         state_dist = data_obj.result[state][subdomain[0], subdomain[1]]
#         if ave_pool is not None and state_dist.shape != (32, 128):
#             state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
#         if np.max(state_dist) - np.min(state_dist) < 1e-6:
#             value = np.zeros_like(state_dist) if zero_mean else np.zeros_like(state_dist) + 0.5
#         else:
#             value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
#             value = value - 0.5 if zero_mean else value
#         state_matrix.append(value)
#     return np.array(state_matrix)

def _get_states(data_obj, subdomain, layers=None, zero_mean=_zero_mean, ave_pool=None):
    if layers is None:
        layers = ["density", "velocity_x", "velocity_y", "pressure"]
    state_matrix = []
    for state in layers:
        state_dist = data_obj.result[state][subdomain[0], subdomain[1]]
        if ave_pool is not None and state_dist.shape != (32, 128):
            state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
        
        state_matrix.append(state_dist)
    return np.array(state_matrix)

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 32, 128), dtype=np.float32)


class DoubleMachReflectionEnv(AlpacaEnv):

    def __init__(self):
        
        super(DoubleMachReflectionEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_32",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_32_weno5",
            high_res=(False, None),
            get_state_func=_get_states,
            cpu_num=4,
            # scheme_parameters=["eta"],
            config={"subdomain": (32, 128)}

        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/doublemach_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        reward = self.compute_reward(end_time=end_time, bias=0.004, coef_dict=self.scale_coef, scale=100)
        return reward


class DoubleMachReflectionHighRes64Env(AlpacaEnv):

    def __init__(self):
        
        super(DoubleMachReflectionHighRes64Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE_NORMAL",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 128), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_64_weno5",
            high_res=(True, 2),
            get_state_func=_get_states,
            cpu_num=6,
            # scheme_parameters=["cq", "eta"],
            config={"subdomain": (64, 256)}
        )
        # self.scale_coef = get_scale_coefs("scheme_rl/data/moving_gresho_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return 0

class DoubleMachReflectionHighRes128Env(AlpacaEnv):

    def __init__(self):
        
        super(DoubleMachReflectionHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE_NORMAL",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_128",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_128_weno5",
            high_res=(True, 4),
            get_state_func=_get_states,
            cpu_num=6,
            # scheme_parameters=["cq", "eta"],
            config={"subdomain": (128, 512)}
        )
        # self.scale_coef = get_scale_coefs("scheme_rl/data/moving_gresho_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return 0

