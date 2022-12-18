import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
from .utils import normalize, get_scale_coefs

_zero_mean = True


def _get_states(data_obj, layers=None, zero_mean=_zero_mean, ave_pool=None):
    assert zero_mean, "Non-zeromean has not been implemented"
    if layers is None:
        layers = ["density", "velocity_x", "velocity_y", "pressure"]
    state_matrix = []
    for state in layers:
        state_dist = data_obj.result[state]
        if ave_pool is not None and state_dist.shape != (64, 64):
            state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
    
        if round(np.max(state_dist) - np.min(state_dist), 6) < 1e-6:
            value = np.zeros_like(state_dist) if zero_mean else np.zeros_like(state_dist) + 0.5
        else:
            value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
            value = value - 0.5 if zero_mean else value
        state_matrix.append(value)
    return np.array(state_matrix)

class RTIEnv(AlpacaEnv):

    def __init__(self):
        super(RTIEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_G",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="rti_32",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 128, 32), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/rti_32_weno5",
            high_res=(False, None),
            cpu_num=4,
            get_state_func=_get_states,
            shape=(128, 32),
        )
        self.scale_coef = get_scale_coefs("scheme_rl/data/rti_teno5_to_weno5.csv", self.end_time, self.timestep_size)

    def get_reward(self, end_time):
        if self.obj.is_crashed:
            return -50
        else:
            # compute the kinetic energy improvement
            reward_ke = self.obj.get_ke_reward(end_time=end_time)
            # compute the anti-diffusion improvement
            reward_si = self.obj.get_dispersive_to_highorder_baseline_penalty(end_time=end_time)
            if self.scale_coef[end_time] < 0 and reward_si > 0:
                self.obj.done = True
                return -50
            si_penalty = np.max((reward_si, 0)) * self.scale_coef[end_time]
            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution
            total_reward = (reward_ke - si_penalty + 0.05) * 10
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)} -> ")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"disper: {round(reward_si, 3):<6} ")
                self.debug.collect_info(f"disper_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"coef: {round(self.scale_coef[end_time], 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<6} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
            return total_reward


class RTIHighRes64Env(AlpacaEnv):

    def __init__(self):
        super(RTIHighRes64Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D2/solvers/ALPACA_32_TENO5RL_ETA_G",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="rti_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 128, 32), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/rti_64_weno5",
            high_res=(True, 2),
            cpu_num=6,
            shape=(256, 64),
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0

class RTIHighRes96Env(AlpacaEnv):

    def __init__(self):
        super(RTIHighRes96Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D2/solvers/ALPACA_32_TENO5RL_ETA_G",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="rti_96",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 128, 32), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/rti_64_weno5",
            high_res=(True, 3),
            cpu_num=6,
            shape=(384, 96),
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0

class RTIHighRes128Env(AlpacaEnv):

    def __init__(self):
        super(RTIHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_G",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="rti_128",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 128, 32), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/rti_128_weno5",
            high_res=(True, 4),
            cpu_num=6,
            shape=(512, 128),
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0


class RTIHighRes256Env(AlpacaEnv):

    def __init__(self):
        super(RTIHighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D2/solvers/ALPACA_32_TENO5RL_ETA_G",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="rti_256",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 256, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.0,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/rti_256_weno5",
            high_res=(True, 4),
            cpu_num=7,
            shape=(1024, 256),
            get_state_func=_get_states,
        )

    def get_reward(self, end_time):
        return 0