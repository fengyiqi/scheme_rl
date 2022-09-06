import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound
from .utils import normalize
import os


_zero_mean = True


def _get_states(data_obj, layers=None, zero_mean=_zero_mean, ave_pool=None):
    assert zero_mean, "Non-zeromean has not been implemented"
    if layers is None:
        layers = ["density", "velocity_x", "velocity_y", "velocity_z", "pressure"]
    state_matrix = []
    for state in layers:
        state_dist = data_obj.result[state]
        if ave_pool is not None and state_dist.shape != (16, 16, 16):
            state_dist = torch.nn.AvgPool3d(ave_pool)(torch.tensor(np.expand_dims(state_dist, axis=0)))[0].numpy()
        if state == "density" or state == "velocity_z":
            if round(np.max(state_dist) - np.min(state_dist), 6) < 1e-6:
                value = np.zeros_like(state_dist)
            else:
                value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
                value = value - 0.5
        else:
            value = normalize(value=state_dist, bounds=(state_dist.min(), state_dist.max()))
            value = value - 0.5
        state_matrix.append(value)

    return state_matrix


class TaylorGreenVortexEnv(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "velocity_x", "velocity_y", "velocity_z", "pressure"]
        paras = ("q", "cq", "eta")
        super(TaylorGreenVortexEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_3D8_TENO5RL_ETA",  # Note that we use HLLC
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="tgv_16",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 16, 16, 16), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.05,
            time_span=10.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/tgv_16_weno5",
            linked_reset=False,
            high_res=(False, None),
            get_state_func=_get_states,
            cpu_num=4,
            dimension=3,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        if self.obj.is_crashed:
            return -100
        else:
            # kinetic energy improvement
            reward_ke = self.obj.get_ke_reward(end_time=end_time) / 5
            ke_improve = True if reward_ke > 0 else False
            # dispersion improvement
            reward_si = self.obj.get_dispersive_penalty(end_time=end_time)
            # si_penalty = abs(np.min((reward_si / 10, 0)))**4
            si_penalty = 0
            si_improve = True if reward_si > 0 else False
            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution

            quality = reward_ke - si_penalty
            self.cumulative_quality += quality
            total_reward = (quality + 0.0)
            # if end_time == "10.000":
            #     obj_iles = self.obj.get_tke_reward()
            #     total_reward += -10 * obj_iles
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)} -> ")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"disper: {round(reward_si, 3):<6} ")
                self.debug.collect_info(f"disper_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<6} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
                self.debug.collect_info(f"improve (si, vor): {si_improve:<1}, {ke_improve:<1} ")
                self.debug.collect_info(f"quality: {round(quality, 3):<6}  ")
            return total_reward

    def get_infos(self, end_time):
        if end_time == "10.000":
            tke_ori = self.obj.get_tke_reward()
            obj_iles = - (tke_ori - 3)*1.5
            if self.evaluation:
                print("tke error: ", tke_ori)
            return {"final_tke": obj_iles}
        else:
            return {}


class TaylorGreenVortexHighRes32Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "velocity_x", "velocity_y", "velocity_z", "pressure"]
        paras = ("q", "cq", "eta")
        super(TaylorGreenVortexHighRes32Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_3D8_TENO5RL_ETA",  # Note that we use HLLC
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="tgv_32",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 16, 16, 16), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.05,
            time_span=10.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/tgv_16_weno5",
            linked_reset=False,
            high_res=(True, 2),
            get_state_func=_get_states,
            cpu_num=6,
            dimension=3,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0


class TaylorGreenVortexHighRes64Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "velocity_x", "velocity_y", "velocity_z", "pressure"]
        paras = ("q", "cq", "eta")
        super(TaylorGreenVortexHighRes64Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_3D8_TENO5RL_ETA",  # Note that we use HLLC
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="tgv_64",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 16, 16, 16), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.05,
            time_span=10.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/tgv_16_weno5",
            linked_reset=False,
            high_res=(True, 4),
            get_state_func=_get_states,
            cpu_num=7,
            dimension=3,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        if float(end_time) > 0.1:
            last_last_time = float(end_time) - 0.1
            os.system(f"rm -rf ~/PycharmProjects/RL2D/runtime_data/tgv_64_{format(last_last_time, '.3f')}")
        return 0
