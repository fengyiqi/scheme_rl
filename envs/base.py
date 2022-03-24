import os
from abc import ABC

import gym
from gym import spaces
import numpy as np
import xml.etree.ElementTree as ET
from boiles.objective.simulation2d import Simulation2D
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from .data_handler import normalize, BaselineDataHandler
else:
    from .data_handler import normalize, BaselineDataHandler

q_bound = (1, 10)
cq_bound = (1, 100)
eta_bound = (0.4, 0.9)
ct_bound = (3, 15)


def fmt(value, dig=".3f"):
    return format(value, dig)

class TimeStepsController:
    def __init__(self, time_span, timestep_size):
        self._time_span = time_span
        self._timestep_size = timestep_size
    def get_time_span(self):
        return self._time_span

    def get_timestep_size(self):
        return self._timestep_size

    def get_end_time_float(self, counter):
        return counter * self._timestep_size

    def get_end_time_string(self, counter):
        return format(self.get_end_time_float(counter), ".3f")



class SchemeParametersWriter:
    def __init__(self, file_loc):
        self.file_loc = file_loc


class DebugProfileHandler:
    def __init__(self):
        pass

class SimulaitonDataHandler:
    def __init__(self, objective):
        self.objective = objective

class GymIOSpaceHandler:
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    def get_io_space(self):
        return self._observation_space, self._action_space

class AlpacaExecutor:
    def __init__(self, executable, inputfile, cpu_num):
        self.executable = executable
        self.inputfile = inputfile
        self.cpu_num = cpu_num

class AlpacaEnv(gym.Env, ABC):

    def __init__(
            self,
            executable: str,
            inputfile: str,
            observation_space: spaces.Box,
            action_space: spaces.Box,
            timestep_size: float = 0.1,
            time_span: float = 2.5,
            cpu_num: int = 1,
            layers: list = None,
            config: dict = None
    ):
        if layers is None:
            layers = ["density", "kinetic_energy", "pressure"]
        if config is None:
            config = dict()
        self.timestep_controller = TimeStepsController(
            time_span=time_span,
            timestep_size=timestep_size
        )
        self.gym_io_space_handler = GymIOSpaceHandler(
            observation_space=observation_space,
            action_space=action_space
        )
        self.observation_space, self.action_space = self.gym_io_space_handler.get_io_space()
        self.alpaca = AlpacaExecutor(
            executable=executable,
            inputfile=inputfile,
            cpu_num=cpu_num
        )
        self.schemefile = SchemeParametersWriter(
            file_loc = config.get(
                "scheme_file", "/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml"
            )
        )
        self.objective = SimulaitonDataHandler(
            objective=config.get("objective", Simulation2D)
        )
        self.baseline_data_handler = BaselineDataHandler(
            timestep_controler=self.timestep_controller,
            data_loc=config.get("baseline_data_loc"),
            layers=self.layers,
            objective=self.objective,
            config=config
        )
        self.bounds = self.baseline_data_handler.bounds
        self.initial_state = self.baseline_data_handler.get_initial_state()
        self.smoothness_threshold = config.get("smoothness_threshold", 0.33)
        self.layers = layers

        self.done, self.si_improve, self.ke_improve, self.evaluation, self.is_crashed = False, False, False, False, False
        self.counter = 0
        self.quality = 0
        self.current_data = None
        self.runtime_info = ""
        self.reset_from_crased = config.get("reset_from_crased", False)

    def _reset_from_crashed(self):
        self.counter += 1
        end_time = self.timestep_controller.get_end_time_string(self.counter)
        state = self._get_restart_state(end_time)


    def _get_restart_state(self, end_time) -> np.array:
        # return the restart file of end time
        pass

    def _reset_flags_and_buffers(self):
        # reset flags and buffers, e.g. self.counter, self.is_crashed
        pass


    def reset(self, print_info=False, evaluate=False):
        self.done, self.evaluation= False, evaluate
        if self.reset_from_crased and self.is_crashed and self.counter < self.end_time / self.timestep_size - 1 and not self.evaluation:
            self.counter += 1
            end_time = format(self.counter * self.timestep_size, ".3f")
            self.current_data = self.objective(
                results_folder=f"/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_teno5/domain",
                result_filename=f"data_{end_time}*.h5"
            )
            self.quality = 0
            self.runtime_info = ""
            state = []
            for i, layer in enumerate(self.layers):
                value = self.current_data.result[layer]
                value = normalize(
                    value=value,
                    bounds=self.bounds[self.layers[i]],
                )
                state.append(value)
            # self.is_crashed = False
            os.system(f"rm -rf runtime_data/{self.inputfile}_*")
            return np.array(state)

        self.counter = 1
        self.quality = 0
        self.is_crashed = False
        self.action_trajectory = []
        self.runtime_info = ""
        # if evaluate:
        #     self.evaluation = True
        os.system(f"rm -rf runtime_data/{self.inputfile}_*")
        return self.initial_state

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action! {action}"
        # action = [np.tanh(a) for a in action]
        self.configure_scheme_xml(action=action)
        end_time = format(self.counter * self.timestep_size, ".3f")
        self.advance_inputfile(end_time=end_time)
        self.run_alpaca(inputfile=f"inputfiles/{self.inputfile}_{end_time}.xml")
        current_state = self.get_state(end_time=end_time)
        reward = self.get_reward(end_time=end_time)
        if self.evaluation:
            print(self.runtime_info)
            self.runtime_info = ""
        if end_time == format(self.end_time, ".3f"):
            self.done = True
        self.counter += 1
        return current_state, reward, self.done, {}

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )

        if not self.current_data.result_exit:

            self.is_crashed = True
            self.done = True
            # self.current_data = self.objective(
            #     results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            #     result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
            # )
            return self.initial_state

        state = []
        for i, layer in enumerate(self.layers):
            value = self.current_data.result[layer]
            value = normalize(
                value=value,
                bounds=self.bounds[self.layers[i]],
            )
            state.append(value)
        return np.array(state)

    def get_reward(self, end_time):
        return NotImplemented

    def configure_scheme_xml(self, action):
        tree = ET.ElementTree(file=self.schemefile)
        root = tree.getroot()
        root[0].text = "0"
        q, cq, eta = action[0], action[1], action[2]

        eta = np.round((eta + 1) / 2 * (eta_bound[1] - eta_bound[0]) + eta_bound[0], 6)
        d1, d2 = np.round((2 + eta) / 4, 4), np.round((1 - eta) / 2, 4)
        q = round((q + 1) / 2 * (q_bound[1] - q_bound[0]) + q_bound[0])
        cq = round((cq + 1) / 2 * (cq_bound[1] - cq_bound[0]) + cq_bound[0])
        ct = 1e-5

        for i, para in enumerate([q, cq, d1, d2, ct]):
            root[i + 2].text = str(para)

        tree.write(self.schemefile)
        if self.evaluation:
            self.action_trajectory.append((q, cq, d1, d2, eta))
            self.runtime_info += f"q, cq, d1, d2, eta: ({q:<2}, {cq:<3}, {fmt(d1)}, {fmt(d2)}, {fmt(eta)})   "

    def advance_inputfile(self, end_time: str):

        new_file = self.rename_inputfile(end_time)
        # starting from initial condition, no restart
        if self.counter == 1:
            restore_mode = "Off"
            restart_file = "None"
        elif self.is_crashed and self.reset_from_crased:
            self.is_crashed = False
            restore_mode = "Forced"
            restart_time = format(float(end_time) - self.timestep_size, ".3f")
            restart_file = f"/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_teno5/restart/restart_{format(float(restart_time), '.3f')}.h5"
        else:
            restore_mode = "Forced"
            restart_time = format(float(end_time) - self.timestep_size, ".3f")
            # restart file needs the exact file name
            restart_file = f"{self.inputfile}_{restart_time}/restart/restart_{format(float(restart_time), '.6f')}.h5"
        configure_input_xml(
            new_file,
            endtime=end_time,
            restore_mode=restore_mode,
            restart_file=restart_file,
            snapshots_type="Stamps"
        )

    def rename_inputfile(self, end_time):
        old_file = f"runtime_data/inputfiles/{self.inputfile}_*.xml"
        new_file = f"runtime_data/inputfiles/{self.inputfile}_{end_time}.xml"

        os.system(f"mv {old_file} {new_file}")
        return new_file

    def run_alpaca(self, inputfile):
        # TODO run a time series
        os.system(f"cd runtime_data; mpiexec -n {self.cpu_num} {self.executable} {inputfile}")

    def test_functionality(self):

        print("Running test function")
        print("Bounds: \n", self.state_bounds)
        print("Baseline reward (kinetic_energy): \n", self.ke_baseline)
        print("Baseline reward (numerical_dissipation_rate): \n", self.nu_baseline)
        print("Baseline reward (smoothness_indicator): \n", self.si_baseline)

        plt.figure(figsize=(4 * len(self.layers), 3))
        plt.suptitle("Initial state")
        initial_state = self.initial_state
        for i in range(len(self.layers)):
            plt.subplot(1, len(self.layers), i + 1)
            plt.imshow(initial_state[i])
            plt.title(self.layers[i])
            plt.colorbar()
        plt.show()
        plt.close()

    def render(self, mode="human"):
        return NotImplemented

    def __str__(self):
        info = self.__class__.__name__ + " Summary:\n\n"
        info += f"\tInputfile: {self.inputfile}\n"
        info += f"\tTimespan: (0, {self.end_time}); Timestep size: {self.timestep_size}\n"
        info += f"\tParameters: q {q_bound}; Cq {cq_bound}; Eta {eta_bound}\n"
        info += f"\tLayers: {self.layers}\n"
        info += f"\tSmoothness: {self.smoothness_threshold}\n"
        info += f"\tBaseline data: {self.baseline_data_loc}\n"
        info += "\n"
        info += f"\tExecutable: {self.executable}\n"
        info += f"\tCore num: {self.cpu_num}"
        # print("inputfile: ", self.inputfile)
        return info


def configure_input_xml(file: str, endtime: str, restore_mode: str, restart_file: str, snapshots_type: str):
    # All arguments should be string
    tree = ET.ElementTree(file=file)
    root = tree.getroot()

    root[4][1].text = endtime  # timeControl -> endTime
    root[6][0][0].text = restore_mode  # restart -> restore -> mode
    root[6][0][1].text = restart_file  # restart -> restore -> fileName
    root[6][1][0].text = snapshots_type  # restart -> snapshots -> type
    root[6][1][3][0].text = endtime  # restart -> snapshots -> stamps -> ts1

    tree.write(file)
    return True


def get_crashed_time(folder):
    files = [file[-11:-3] for file in glob.glob(os.path.join(folder, "*.h5"))]
    time = np.array(files).astype(float)
    try:
        end_time = np.sort(time)[-2]
    except:
        end_time = np.sort(time)[-1]
    return end_time


# def main():
#     env = AlpacaEnv()
#     env.reset()
#     env.test_functionality()
#
#
# if __name__ == '__main__':
#     main()
