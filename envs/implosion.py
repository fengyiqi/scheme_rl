import numpy as np
import torch
from .env_base import AlpacaEnv
from gym import spaces


class ImplosionEnv(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="implosion_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32),
            timestep_size=0.1,
            time_span=2.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64",
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        if self.is_crashed:
            return 0
        else:
            # truncation error improvement
            reward_tr = self.obj.get_truncation_reward(end_time=end_time)
            tr_improve = True if reward_tr > 0 else False
            # smoothness improvement
            reward_si = self.obj.get_smoothness_reward(end_time=end_time)
            si_improve = True if reward_si > 0 else False
            # smoothness indicator adaptive weight
            si_penalty = abs(np.min((reward_si, 0))) ** 1.1

            self.quality += (reward_tr - si_penalty)
            total_reward = 10 * self.quality
            if self.evaluation:
                self.debug.collect_info(f"si_penalty: {round(si_penalty, 3):<3} ")
                self.debug.collect_info(f"improvement (tr, si): {tr_improve:<2}, {si_improve:<2} ")
                self.debug.collect_info(f"quality: {self.quality}")
            return total_reward