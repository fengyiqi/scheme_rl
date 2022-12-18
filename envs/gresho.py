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
        if ave_pool is not None and state_dist.shape != (32, 128):
            state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
        if np.max(state_dist) - np.min(state_dist) < 1e-6:
            value = np.zeros_like(state_dist) if zero_mean else np.zeros_like(state_dist) + 0.5
        else:
            value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
            value = value - 0.5 if zero_mean else value
        state_matrix.append(value)
    return np.array(state_matrix)


class MovingGreshoEnv(AlpacaEnv):

    def __init__(self):
        
        super(MovingGreshoEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="moving_gresho_32",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 128), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=3.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/moving_gresho_32_weno5",
            high_res=(False, None),
            get_state_func=_get_states,
            cpu_num=4,
            shape=(32, 128)
        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/moving_gresho_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        reward = self.compute_reward(end_time=end_time, bias=0.0, coef_dict=self.scale_coef, scale=10000)
        return reward
    
    def l2_error(self, end_time):
        initial = self.obj.baseline_data_obj.get_raw_state("vorticity")
        final = self.obj.get_raw_state(end_time, "vorticity")
        errors = np.linalg.norm(initial[:, :32].flatten() - final[:, -32:].flatten(), ord=2)
        return errors


class MovingGreshoHighRes64Env(AlpacaEnv):

    def __init__(self):
        
        super(MovingGreshoHighRes64Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="moving_gresho_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 128), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=3.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/moving_gresho_64_weno5",
            high_res=(True, 2),
            get_state_func=_get_states,
            cpu_num=6,
            shape=(64, 256)
        )
        # self.scale_coef = get_scale_coefs("scheme_rl/data/moving_gresho_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return 0


class MovingGreshoHighRes128Env(AlpacaEnv):

    def __init__(self):
        
        super(MovingGreshoHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_VOR",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="moving_gresho_128",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 128), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=3.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/moving_gresho_128_weno5",
            high_res=(True, 4),
            get_state_func=_get_states,
            cpu_num=6,
            shape=(128, 512)
        )
        # self.scale_coef = get_scale_coefs("scheme_rl/data/moving_gresho_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        return 0
# class ViscousShockTubeHighRes128Env(AlpacaEnv):

#     def __init__(self):
#         super(ViscousShockTubeHighRes128Env, self).__init__(
#             executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_TEMP",
#             schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
#             inputfile="viscous_shock_128",
#             observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 64), dtype=np.float32),
#             action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
#             timestep_size=0.01,
#             time_span=1.0,
#             baseline_data_loc="/media/yiqi/Elements/RL/baseline/viscous_shock_128_weno5",
#             high_res=(True, 2),
#             get_state_func=_get_states,
#             cpu_num=4,
#             shape=(64, 128)
#         )

#     def get_reward(self, end_time):
#         return 0


# class ViscousShockTubeHighRes256Env(AlpacaEnv):

#     def __init__(self):
#         super(ViscousShockTubeHighRes256Env, self).__init__(
#             executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_TEMP",
#             schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
#             inputfile="viscous_shock_256",
#             observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 64), dtype=np.float32),
#             action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
#             timestep_size=0.01,
#             time_span=1.0,
#             baseline_data_loc="/media/yiqi/Elements/RL/baseline/viscous_shock_256_weno5",
#             high_res=(True, 4),
#             get_state_func=_get_states,
#             cpu_num=6,
#             shape=(128, 256)
#         )

#     def get_reward(self, end_time):
#         return 0

# class ViscousShockTubeHighRes512Env(AlpacaEnv):

#     def __init__(self):
#         super(ViscousShockTubeHighRes512Env, self).__init__(
#             executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_TEMP",
#             schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
#             inputfile="viscous_shock_512",
#             observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 32, 64), dtype=np.float32),
#             action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
#             timestep_size=0.01,
#             time_span=1.0,
#             baseline_data_loc="/media/yiqi/Elements/RL/baseline/viscous_shock_512_weno5",
#             high_res=(True, 8),
#             get_state_func=_get_states,
#             cpu_num=6,
#             shape=(256, 512)
#         )

#     def get_reward(self, end_time):
#         return 0