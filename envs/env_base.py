import os
from abc import ABC, abstractmethod

import gym
from gym import spaces
import numpy as np
import xml.etree.ElementTree as ET
from boiles.objective.simulation2d import Simulation2D
import glob
import matplotlib.pyplot as plt
from .sim_base import (
    GymIOSpaceHandler,
    SchemeParametersWriter,
    TimeStepsController,
    AlpacaExecutor,
    SimulationHandler,
    DebugProfileHandler
)

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
        self.objective = SimulationHandler(
            solver=alpaca,
            time_controller=self.timestep_controller,
            baseline_data_obj=baseline_data_obj,
            scheme_writer=self.schemefile,
            linked_reset=True
        )
        self.debug = DebugProfileHandler(
            objective=self.objective
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

    def _reset_from_crashed(self):
        conditions = (
            self.linked_reset,
            self.objective.is_crashed,
            self.timestep_controller.counter < self.timestep_controller.get_total_steps() - 1,
        )
        return False not in conditions

    def reset(self, print_info=False, evaluate=False):
        if self._reset_from_crashed():
            self.timestep_controller.counter += 1
            end_time = format(self.timestep_controller.counter * self.timestep_controller.get_timestep_size(), ".3f")
            states = self.objective.baseline_data_obj.get_baseline_state(end_time=end_time)
            self._reset_flags_and_buffers()
            return np.array(states)
        self._reset_flags_and_buffers()
        if evaluate:
            self.debug.collect_scheme_paras()
            self.debug.flush_info()
        return self.initial_state

    def step(self, action: list):
        assert self.action_space.contains(action), f"Invalid action! {action}"
        # action = [np.tanh(a) for a in action]
        self.schemefile.configure_scheme_xml(action)
        end_time = self.timestep_controller.get_end_time_string(self.timestep_controller.counter)
        inputfile = self.objective.configure_inputfile(end_time=end_time)
        self.objective.run(inputfile=inputfile)
        current_state = self.objective.get_states(end_time=end_time)
        reward = self.get_reward(end_time=end_time)

        if end_time == self.timestep_controller.get_time_span_string():
            self.done = True
        self.timestep_controller.counter += 1
        return current_state, reward, self.done, {}

    @abstractmethod
    def get_reward(self, end_time):
        return NotImplemented


    def render(self, mode="human"):
        return NotImplemented

    def __str__(self):
        info = self.__class__.__name__ + " Summary:\n\n"
        info += f"\tInputfile: {self.objective.inputfile}\n"
        info += f"\tTimespan: (0, {self.timestep_controller.get_time_span_string()}); " \
                f"Timestep size: {self.timestep_controller.get_timestep_size()}\n"
        info += f"\tParameters: q {q_bound}; Cq {cq_bound}; Eta {eta_bound}; Ct(power) {ct_power_bound}\n"
        info += f"\tLayers: {self.layers}\n"
        info += f"\tSmoothness: {self.smoothness_threshold}\n"
        info += f"\tBaseline data: {self.objective.baseline_data_obj.state_data_loc}\n"
        info += "\n"
        info += f"\tExecutable: {self.objective.solver.executable}\n"
        info += f"\tCore num: {self.objective.solver.cpu_num}"
        # print("inputfile: ", self.inputfile)
        return info


