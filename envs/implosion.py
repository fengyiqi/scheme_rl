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
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64",
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
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
            # action_penalty = self.obj.get_action_penalty()
            action_penalty = 0

            self.quality += (reward_ke - si_penalty)
            total_reward = 10 * (reward_ke - si_penalty - action_penalty)
            if self.evaluation:
                self.debug.collect_info(f"si_penalty: {round(si_penalty, 3):<5} ")
                # self.debug.collect_info(f"a_penalty: {round(action_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<5} ")
                self.debug.collect_info(f"improve (si, ke): {si_improve:<2}, {ke_improve:<2} ")
                self.debug.collect_info(f"quality: {round(self.quality, 3):<6}  ")
            return total_reward