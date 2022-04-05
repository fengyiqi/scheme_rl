import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound

class ImplosionEnv(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(ImplosionEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="implosion_64",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.1,
            time_span=2.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64_weno5",
            linked_reset=True,
            high_res=(False, None),
            cpu_num=4,
            layers=layers,
            config=config
        )

    def _get_reward(self, end_time):
        if self.obj.is_crashed:
            return -100
        else:
            # truncation error improvement
            reward_ke = self.obj.get_ke_reward(end_time=end_time)
            ke_improve = True if reward_ke > 0 else False
            # smoothness improvement
            reward_si = self.obj.get_smoothness_reward(end_time=end_time)
            si_improve = True if reward_si > 0 else False
            # smoothness indicator adaptive weight
            si_penalty = abs(np.min((reward_si, 0))) ** 1.3

            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution
            quality = (reward_ke - si_penalty)
            self.cumulative_quality += quality
            total_reward = 10 * reward_si
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)}->")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"si_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<5} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<5} ")
                self.debug.collect_info(f"improve (si, ke): {si_improve:<2}, {ke_improve:<2} ")
                self.debug.collect_info(f"quality: {round(self.cumulative_quality, 3):<6}  ")
            return total_reward

    def get_reward(self, end_time):
        if self.obj.is_crashed:
            return -100
        else:
            # truncation error improvement
            reward_ke = self.obj.get_ke_reward(end_time=end_time)
            ke_improve = True if reward_ke > 0 else False
            # smoothness improvement
            # reward_si = self.obj.get_smoothness_reward(end_time=end_time)
            reward_si = self.obj.get_cutoff_tke_reward(end_time=end_time)
            si_improve = True if reward_si > 0 else False
            # reward_si = self.obj.current_data._create_spectrum()[32:, 1].sum()
            # si_improve =
            # smoothness indicator adaptive weight
            si_penalty = abs(np.min((reward_si, 0))) ** 1.0

            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution

            quality = (reward_ke - si_penalty)
            self.cumulative_quality += quality
            total_reward = 10 * reward_si
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)}->")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"si_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<6} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
                self.debug.collect_info(f"improve (si, ke): {si_improve:<1}, {ke_improve:<1} ")
                self.debug.collect_info(f"quality: {round(self.cumulative_quality, 3):<6}  ")
            return total_reward

class ImplosionHighResEnv(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(ImplosionHighResEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="implosion_128",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.1,
            time_span=2.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_128",
            linked_reset=False,
            high_res=True,
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0