import numpy as np
import torch
from .base import AlpacaEnv, fmt, eta_bound, ct_bound
from .data_handler import normalize
from gym import spaces
import xml.etree.ElementTree as ET



class RiemannConfig3(AlpacaEnv):

    def __init__(self, config=None):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        class_config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_weno5/domain",
            scheme_file="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            smoothness_threshold=0.15
        )
        if config is not None:
            class_config.update(config)
        layers = ["density", "kinetic_energy", "pressure"]
        super(RiemannConfig3, self).__init__(
            executable=executable,
            inputfile="config3_64",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.05,
            end_time=1.0,
            layers=layers,
            config=class_config
        )
        if self.baseline_data_loc is not None:
            self.ke_baseline = self.baseline_data_handler.get_baseline_reward(prop="kinetic_energy")
            self.si_baseline = self.baseline_data_handler.get_baseline_reward(prop="smoothness_indicator")
            self.nu_baseline = self.baseline_data_handler.get_baseline_reward(prop="numerical_dissipation_rate")

    def get_reward(self, end_time):
        if self.is_crashed:
            return -10
        else:
            # smoothness improvement
            _, reward = self.current_data.smoothness(threshold=self.smoothness_threshold)
            reward_si = reward / self.si_baseline[end_time] - 1
            self.si_improve = True if reward_si > 0 else False

            # kinetic energy improvement (dissipation reduce)
            # ke = self.current_data.result["kinetic_energy"]
            # reward_ke = ke.sum() / self.ke_baseline[end_time] - 1
            # self.ke_improve = True if reward_ke > 0 else False
            _, _, _, nu = self.current_data.truncation_errors()
            reward_nu = nu.sum() / self.nu_baseline[end_time] - 1
            si_penalty = abs(np.min((reward_si, 0))) ** 1.3
            self.ke_improve = True if reward_nu > 0 else False
            # smoothness indicator adaptive weight
            current_quality = reward_nu - si_penalty
            self.quality += current_quality
            total_reward = 10 * np.log( current_quality + 1 )
            self.runtime_info += f"si_penalty: {fmt(si_penalty)}  reward_si: {fmt(reward_si):<6} "
            self.runtime_info += f"Improve (si, ke)={self.si_improve:<2}, {self.ke_improve:<2} Reward: {fmt(total_reward):<6}   "
            self.runtime_info += f"Quality: {round(current_quality, 3):<5}"
            return total_reward


class RiemannConfig3HighRes(AlpacaEnv):
    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_weno5/domain",
            scheme_file="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            smoothness_threshold=0.15
        )
        layers = ["density", "kinetic_energy", "pressure"]
        super(RiemannConfig3HighRes, self).__init__(
            executable=executable,
            inputfile="config3_128",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.05,
            end_time=1.0,
            layers=layers,
            config=config
        )

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )

        state = []
        for i, layer in enumerate(self.layers):
            value = self.current_data.result[layer]
            value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy()
            value = normalize(
                value=value,
                bounds=self.bounds[self.layers[i]],
            )

            state.append(value)
        return np.array(state)

    def get_reward(self, end_time):
        return 1

# class ImplosionHighResEnv(AlpacaEnv):
#     def __init__(self):
#         executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
#         config = dict(
#             baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64/domain"
#         )
#         layers = ["density", "kinetic_energy", "pressure"]
#         super(ImplosionHighResEnv, self).__init__(
#             executable=executable,
#             inputfile="implosion_128",
#             observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
#             timestep_size=0.1,
#             end_time=2.5,
#             layers=layers,
#             config=config
#         )
#
#     def get_state(self, end_time):
#         self.current_data = self.objective(
#             results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
#             result_filename=f"data_{end_time}0*.h5"
#         )
#         if not self.current_data.result_exit:
#             self.is_crashed = True
#             self.done = True
#             self.current_data = self.objective(
#                 results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
#                 result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
#             )
#
#         state = []
#         for i, layer in enumerate(self.layers):
#             value = self.current_data.result[layer]
#             value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy()
#             value = normalize(
#                 value=value,
#                 bounds=self.bounds[self.layers[i]],
#             )
#
#             state.append(value)
#         return np.array(state)
#
#     def get_reward(self, end_time):
#         return 1