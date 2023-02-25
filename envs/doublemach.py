import numpy as np
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
import torch
from .utils import get_scale_coefs

_zero_mean = True


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
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE_ROEM",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_32",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_32_weno5",
            high_res=(False, None),
            get_state_func=_get_states,
            cpu_num=2,
            # scheme_parameters=["eta"],
            config={"subdomain": (32, 128)}

        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/doublemach_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        reward = self.compute_reward(end_time=end_time, bias=0.002, coef_dict=self.scale_coef, scale=1000)
        return reward

class DoubleMachReflectionHighRes64Env(AlpacaEnv):

    def __init__(self):
        
        super(DoubleMachReflectionHighRes64Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE_ROEM",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_64",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_64_weno5",
            high_res=(True, 2),
            get_state_func=_get_states,
            cpu_num=6,
            # scheme_parameters=["eta"],
            config={"subdomain": (64, 256)}
        )
        self.states = []
    def get_reward(self, end_time):
        self.states.append(self.obj.get_state(end_time=end_time))
        return 0

class DoubleMachReflectionHighRes128Env(AlpacaEnv):

    def __init__(self):
        
        super(DoubleMachReflectionHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE_ROEM",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_128",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_128_teno5lin",
            high_res=(True, 4),
            get_state_func=_get_states,
            cpu_num=6,
            # scheme_parameters=["eta"],
            config={"subdomain": (128, 512)}
        )
        self.states = []
    def get_reward(self, end_time):
        self.states.append(self.obj.get_state(end_time=end_time))
        return 0

class DoubleMachReflectionHighRes160Env(AlpacaEnv):

    def __init__(self):
        
        super(DoubleMachReflectionHighRes160Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/doublemach/ALPACA_32_TENO5RL_ETA_DOUBLE_ROEM",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="doublemach_160",
            observation_space=observation_space,
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.002,
            time_span=0.2,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/doublemach/doublemach_160_teno5lin",
            high_res=(True, 5),
            get_state_func=_get_states,
            cpu_num=6,
            # scheme_parameters=["eta"],
            config={"subdomain": (160, 640)}
        )
        self.states = []
    def get_reward(self, end_time):
        self.states.append(self.obj.get_state(end_time=end_time))
        return 0

