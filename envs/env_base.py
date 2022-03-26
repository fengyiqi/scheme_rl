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
ct_power_bound = (3, 15)
action_bound = (-1, 1)

def fmt(value, dig=".3f"):
    return format(value, dig)




class AlpacaEnv(gym.Env, ABC):

    def __init__(
            self,
            executable: str,
            inputfile: str,
            observation_space: spaces.Box,
            action_space: spaces.Box,
            timestep_size: float,
            time_span: float,
            baseline_data_loc: str,
            cpu_num: int = 1,
            layers: list = None,
            config: dict = None
    ):
        if layers is None:
            layers = ["density", "kinetic_energy", "pressure"]
        if config is None:
            config = dict()

        self.gym_io_space_handler = GymIOSpaceHandler(
            observation_space=observation_space,
            action_space=action_space
        )
        self.observation_space, self.action_space = self.gym_io_space_handler.get_io_space()

        self.schemefile = SchemeParametersWriter(
            file = config.get(
                "scheme_file", "/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml"
            )
        )
        alpaca = AlpacaExecutor(
            executable=executable,
            inputfile=inputfile,
            cpu_num=cpu_num
        )
        self.timestep_controller = TimeStepsController(
            time_span=time_span,
            timestep_size=timestep_size
        )
        baseline_data_obj = BaselineDataHandler(
            timestep_size=timestep_size,
            time_span=time_span,
            data_loc=baseline_data_loc,
            layers=layers,
            config=config
        )
        self.objective = SimulaitonHandler(
            solver=alpaca,
            time_controller=self.timestep_controller,
            baseline_data_obj=baseline_data_obj,
            linked_reset=True
        )

        self.bounds = self.baseline_data_handler.bounds
        self.initial_state = self.baseline_data_handler.get_initial_state()
        self.smoothness_threshold = config.get("smoothness_threshold", 0.33)
        self.layers = layers

        self.done, self.si_improve, self.ke_improve, self.evaluation = False, False, False, False
        self.quality = 0
        self.current_data = None
        self.runtime_info = ""
        self.linked_reset = config.get("linked_reset", False)
        self._build_folders()

    def _build_folders(self):
        if not os.path.exists("runtime/inputfiles"):
            os.makedirs("runtime/inputfiles")
        else:
            os.system("rm -rf runtime/inputfiles/*")
        os.system(f"mv scheme_rl/xml/{self.objective.inputfile}.xml runtime/inputfiles/{self.objective.inputfile}")
        os.system(f"mv scheme_rl/xml/{self.schemefile.file}.xml runtime/{self.schemefile.file}")

    def _reset_flags_and_buffers(self):
        # reset flags and buffers, e.g. self.counter, self.is_crashed
        self.is_crashed = False
        self.done = False
        os.system(f"rm -rf runtime_data/{self.objective.inputfile}_*")

    def reset_from_crashed(self):
        conditions = (
            self.linked_reset,
            self.objective.is_crashed,
            self.timestep_controller.counter < self.timestep_controller.get_total_steps() - 1,
            not self.evaluation
        )
        return False not in conditions

    def reset(self, print_info=False, evaluate=False):
        if self.reset_from_crashed():
            self.timestep_controller.counter += 1
            end_time = format(self.timestep_controller.counter * self.timestep_controller.get_timestep_size(), ".3f")
            states = self.objective.baseline_data_obj.get_baseline_state(end_time=end_time)
            self._reset_flags_and_buffers()
            return np.array(states)
        self._reset_flags_and_buffers()
        return self.initial_state

    def step(self, action: list):
        assert self.action_space.contains(action), f"Invalid action! {action}"
        # action = [np.tanh(a) for a in action]
        self.schemefile.configure_scheme_xml(action)
        end_time = self.timestep_controller.get_end_time_string(self.timestep_controller.counter)

        self.advance_inputfile(end_time=end_time)
        self.run_alpaca(inputfile=f"inputfiles/{self.objective.inputfile}_{end_time}.xml")
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
