import numpy as np
import torch
from .env_base import AlpacaEnv, fmt
from .data_handler import normalize
from gym import spaces


class ShearFlowEnv(AlpacaEnv):

    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        layers = ["schlieren", "vorticity", "numerical_dissipation_rate"]
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/shear_64/domain",
            smoothness_threshold=None
        )
        super(ShearFlowEnv, self).__init__(
            executable=executable,
            inputfile="shear_64",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.05,
            end_time=1.2,
            layers=layers,
            config=config
        )
        if self.baseline_data_loc is not None:
            self.nu_baseline = self.baseline_data_handler.get_baseline_reward(prop="numerical_dissipation_rate")
        self.nu_improve = False

    def get_reward(self, end_time):
        if self.is_crashed:
            return -100
        else:
            # numerical errors improvement
            _, _, reward = self.current_data.truncation_errors()
            # reward = 1 - reward / self.nu_baseline[end_time]
            reward = -reward + 5
            self.nu_improve = True if reward > 0 else False
            total_reward = reward * 1
            self.runtime_info += f"Improve nu = {self.nu_improve}   Reward: {fmt(total_reward)}"
            return total_reward


class ShearFlowHighResEnv(AlpacaEnv):

    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        layers = ["schlieren", "vorticity", "numerical_dissipation_rate"]
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/shear_64/domain",
            smoothness_threshold=None
        )
        super(ShearFlowHighResEnv, self).__init__(
            executable=executable,
            inputfile="shear_128",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.05,
            end_time=1.2,
            layers=layers,
            config=config
        )
        if self.baseline_data_loc is not None:
            self.nu_baseline = self.baseline_data_handler.get_baseline_reward(prop="numerical_dissipation_rate")
        self.nu_improve = False

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            self.current_data = self.objective(
                results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
                result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
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
#             baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/full_implosion/full_implosion_64/domain"
#         )
#         super(ImplosionHighResEnv, self).__init__(
#             executable=executable,
#             inputfile="implosion_128",
#             timestep_size=0.1,
#             end_time=2.5,
#             layers=["density", "kinetic_energy", "pressure"],
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
#             value = normalize_state(
#                 value=value,
#                 bounds=self.state_bounds[layer],
#                 layer=layer
#             )
#             value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy().tolist()
#             state.append(value)
#         return np.array(state)
#
#     def get_reward(self, end_time):
#         return 1
