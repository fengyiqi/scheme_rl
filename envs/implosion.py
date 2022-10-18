import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
from .utils import normalize


_zero_mean = True


def _get_states(data_obj, layers, zero_mean=_zero_mean, ave_pool=None):
    assert zero_mean, "Non-zeromean has not been implemented"
    state_matrix = []
    for state in layers:
        state_dist = data_obj.result[state]
        if ave_pool is not None and state_dist.shape != (64, 64):
            state_dist = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
        if state == "velocity_x" or state == "velocity_y":
            if round(np.max(state_dist) - np.min(state_dist), 6) < 1e-6:
                value = np.zeros_like(state_dist) if zero_mean else np.zeros_like(state_dist) + 0.5
            else:
                value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
                value = value - value.mean() if zero_mean else value
        else:
            value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
            value = value - 0.5 if zero_mean else value
        state_matrix.append(value)
    return np.array(state_matrix)

 
class ImplosionEnv(AlpacaEnv):

    def __init__(self):
        super(ImplosionEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="implosion_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(3, ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.5,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/implosion_64_weno5",
            high_res=(False, None),
            get_state_func=_get_states,
            cpu_num=4,
        )

    def get_reward(self, end_time):
        if self.obj.is_crashed:
            return -100
        else:
            # kinetic energy improvement
            reward_ke = self.obj.get_ke_reward(end_time=end_time)
            # anti-diffusion increase
            reward_si = self.obj.get_dispersive_to_highorder_baseline_penalty(end_time=end_time)
            si_penalty = np.max((reward_si, 0)) * 0.005216
            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution
            total_reward = reward_ke - si_penalty
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)} -> ")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"disper: {round(reward_si, 3):<6} ")
                self.debug.collect_info(f"disper_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<6} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
            return total_reward


class ImplosionHighRes128Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "velocity_x", "velocity_y", "pressure"]
        paras = ("q", "cq", "eta")
        super(ImplosionHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="implosion_128",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.5,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/implosion_64_weno5",
            linked_reset=False,
            high_res=(True, 2),
            get_state_func=_get_states,
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0


class ImplosionHighRes256Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "velocity_x", "velocity_y", "pressure"]
        paras = ("q", "cq", "eta")
        super(ImplosionHighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="implosion_256",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.01,
            time_span=2.5,
            baseline_data_loc="/media/yiqi/Elements/RL/baseline/implosion_64_weno5",
            linked_reset=False,
            high_res=(True, 4),
            get_state_func=_get_states,
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0
